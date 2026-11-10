import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import json
import os
from pathlib import Path
from jax import random, vmap
from tqdm import tqdm
import itertools
import random as random_py

from structflow.pgms.pgm_compiler import PGMCompiler
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.learner_factory import create_learner, DEFAULT_COUPLING
from pgm_paper_utils.infrastructure import count_total_params, get_hidden_dims
from pgm_paper_utils.utils import generate_run_name, make_structure
from structflow.pgms.utilities import format_dist_str

# --- Training Logic ---
def _make_chunk_runner(pgm, step_fn, model, normaliser, batch_size, length):
    """jit(lax.scan) over `length` fused (sample -> step) iterations.

    Uses exactly the same key-splitting as the original per-step loop, so the
    RNG stream (data, noise, time, coupling keys) is unchanged; only XLA fusion
    differs, giving float32-rounding-level differences in the losses.
    """
    m_static = eqx.partition(model, eqx.is_array)[1]
    n_static = eqx.partition(normaliser, eqx.is_array)[1]

    def body(carry, _):
        m_params, opt_state, n_params, key = carry
        m = eqx.combine(m_params, m_static)
        n = eqx.combine(n_params, n_static)
        key, step_key, data_key = random.split(key, 3)
        batch = vmap(pgm.sample)(random.split(data_key, batch_size))
        m, opt_state, n, loss = step_fn(m, opt_state, n, batch, step_key)
        return (eqx.filter(m, eqx.is_array), opt_state, eqx.filter(n, eqx.is_array), key), loss

    @jax.jit
    def run(m_params, opt_state, n_params, key):
        return jax.lax.scan(body, (m_params, opt_state, n_params, key), None, length=length)

    return run


def train_run(config, save_dir, pgm, structure, n_checkpoints=3, steps_per_call=None):
    """Train one run.

    steps_per_call (default from $SWEEP_STEPS_PER_CALL, else 100) fuses that
    many sample+step iterations into one jitted lax.scan call and keeps the
    losses on device.  It is deliberately NOT a config key, so run names /
    config hashes are unchanged.  steps_per_call=0 runs the original
    per-step loop.  Checkpoint steps and file names are identical to before.
    """
    if steps_per_call is None:
        steps_per_call = int(os.environ.get('SWEEP_STEPS_PER_CALL', 100))
    if steps_per_call <= 0:
        return _train_run_per_step(config, save_dir, pgm, structure, n_checkpoints)

    key = random.PRNGKey(config['learner_seed'])
    model, normaliser, step_fn, optimizer = create_learner(config)
    n_params = count_total_params(model)
    print(f'n_params: {n_params}')
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    T = config['train_steps']
    # Iteration indices i (0..T) after which the original loop saved a file.
    ends = []
    if n_checkpoints is not None:
        ends = [i for i in range(T + 1) if i % (T // n_checkpoints) == 0 and i != T]
    ends.append(T)

    K = steps_per_call
    run_k = _make_chunk_runner(pgm, step_fn, model, normaliser, config['batch_size'], K)
    run_1 = _make_chunk_runner(pgm, step_fn, model, normaliser, config['batch_size'], 1)
    m_static = eqx.partition(model, eqx.is_array)[1]
    n_static = eqx.partition(normaliser, eqx.is_array)[1]
    m_params = eqx.filter(model, eqx.is_array)
    n_params_ = eqx.filter(normaliser, eqx.is_array)

    loss_chunks = []
    pbar = tqdm(total=T + 1, desc=f"{config['mode']} (H={config['hidden_dim']})")
    i = 0  # next iteration index to run
    for e in ends:
        L = e - i + 1
        for runner, n_calls, n in ((run_k, L // K, K), (run_1, L % K, 1)):
            for _ in range(n_calls):
                (m_params, opt_state, n_params_, key), losses = runner(m_params, opt_state, n_params_, key)
                loss_chunks.append(losses)
                pbar.update(n)
        i = e + 1
        last = float(loss_chunks[-1][-1])  # one host sync per segment
        pbar.set_postfix({'loss': f"{last:.4f}"})
        if e != T:
            eqx.tree_serialise_leaves(os.path.join(save_dir, f"checkpoint_{e}.eqx"),
                                      (eqx.combine(m_params, m_static), eqx.combine(n_params_, n_static)))
    pbar.close()

    model = eqx.combine(m_params, m_static)
    normaliser = eqx.combine(n_params_, n_static)
    jnp.save(os.path.join(save_dir, "losses.npy"), jnp.concatenate(loss_chunks))
    eqx.tree_serialise_leaves(os.path.join(save_dir, f"model_{T}.eqx"), (model, normaliser))
    print(f"Saved to {save_dir}")


def _train_run_per_step(config, save_dir, pgm, structure, n_checkpoints=3):
    key = random.PRNGKey(config['learner_seed'])
    
    # 3. Initialize Model via Factory
    model, normaliser, step_fn, optimizer = create_learner(config)

    n_params = count_total_params(model)
    print(f'n_params: {n_params}')
    
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # 1. Data Generator
    @eqx.filter_jit
    def data_generator(k, batch_size):
        subkeys = random.split(k, batch_size)
        return vmap(pgm.sample)(subkeys)

    losses = []
    pbar = tqdm(range(config['train_steps']+1), desc=f"{config['mode']} (H={config['hidden_dim']})")
    
    for i in pbar:
        key, step_key, data_key = random.split(key, 3)
        
        # Data Sampling
        batch_data_dict = data_generator(data_key, config['batch_size'])
        
        # Optimization Step
        model, opt_state, normaliser, loss = step_fn(model, opt_state, normaliser, batch_data_dict, step_key)
        losses.append(float(loss))

        if i % 100 == 0:
            pbar.set_postfix({'loss': f"{loss:.4f}"})

        if n_checkpoints is not None:
            if i % (config['train_steps'] // n_checkpoints) == 0:
                if i != config['train_steps']:
                    # checkpoint
                    eqx.tree_serialise_leaves(os.path.join(save_dir, f"checkpoint_{i}.eqx"), (model, normaliser))

    jnp.save(os.path.join(save_dir, "losses.npy"), jnp.array(losses))
    eqx.tree_serialise_leaves(os.path.join(save_dir, f"model_{i}.eqx"), (model, normaliser))
    print(f"Saved to {save_dir}")


def train_runs_seed_batched(configs, save_dirs, pgm, structure, n_checkpoints=3, steps_per_call=None):
    """Train several learner seeds of one (mode, pgm_seed) at once with jax.vmap.

    Each seed keeps its own init and RNG stream (key = PRNGKey(learner_seed)),
    so every run is the same computation as train_run up to float32 rounding;
    outputs (losses.npy, checkpoints, model file) go to each seed's own dir.
    Configs must differ only in learner_seed.
    """
    if steps_per_call is None or steps_per_call <= 0:
        steps_per_call = int(os.environ.get('SWEEP_STEPS_PER_CALL', 100)) or 100
    strip = lambda c: {k: v for k, v in c.items() if k != 'learner_seed'}
    assert all(strip(c) == strip(configs[0]) for c in configs), "configs must differ only in learner_seed"
    models, norms, opts, keys = [], [], [], []
    for c in configs:
        m, n, step_fn, optimizer = create_learner(c)
        models.append(m); norms.append(n)
        opts.append(optimizer.init(eqx.filter(m, eqx.is_array)))
        keys.append(random.PRNGKey(c['learner_seed']))
    print(f'n_params: {count_total_params(models[0])} x {len(configs)} seeds')

    def stack(xs):  # stack array leaves; treedef (static fields) from the first element
        td = jax.tree_util.tree_structure(xs[0])
        return jax.tree_util.tree_unflatten(td, [jnp.stack(a) for a in zip(*[jax.tree_util.tree_leaves(x) for x in xs])])

    def unstack(x, i):
        return jax.tree_util.tree_map(lambda a: a[i], x)

    m_static = eqx.partition(models[0], eqx.is_array)[1]
    n_static = eqx.partition(norms[0], eqx.is_array)[1]
    m_params = stack([eqx.filter(m, eqx.is_array) for m in models])
    n_params_ = stack([eqx.filter(n, eqx.is_array) for n in norms])
    opt_state = stack(opts)
    key = jnp.stack(keys)
    B = configs[0]['batch_size']

    def one(m_p, o, n_p, k):
        m = eqx.combine(m_p, m_static); n = eqx.combine(n_p, n_static)
        k, step_key, data_key = random.split(k, 3)
        batch = vmap(pgm.sample)(random.split(data_key, B))
        m, o, n, loss = step_fn(m, o, n, batch, step_key)
        return eqx.filter(m, eqx.is_array), o, eqx.filter(n, eqx.is_array), k, loss
    v_one = vmap(one)

    def make_runner(length):
        @jax.jit
        def run(carry):
            def body(c, _):
                *c, loss = v_one(*c)
                return tuple(c), loss
            return jax.lax.scan(body, carry, None, length=length)
        return run

    T = configs[0]['train_steps']
    ends = []
    if n_checkpoints is not None:
        ends = [i for i in range(T + 1) if i % (T // n_checkpoints) == 0 and i != T]
    ends.append(T)
    K = steps_per_call
    run_k, run_1 = make_runner(K), make_runner(1)
    carry = (m_params, opt_state, n_params_, key)
    loss_chunks = []

    def save(fname, carry):
        for s_i, d in enumerate(save_dirs):
            eqx.tree_serialise_leaves(os.path.join(d, fname),
                                      (eqx.combine(unstack(carry[0], s_i), m_static),
                                       eqx.combine(unstack(carry[2], s_i), n_static)))

    pbar = tqdm(total=T + 1, desc=f"{configs[0]['mode']} x{len(configs)} (H={configs[0]['hidden_dim']})")
    i = 0
    for e in ends:
        L = e - i + 1
        for runner, n_calls, n in ((run_k, L // K, K), (run_1, L % K, 1)):
            for _ in range(n_calls):
                carry, losses = runner(carry)
                loss_chunks.append(losses)  # (n, n_seeds)
                pbar.update(n)
        i = e + 1
        pbar.set_postfix({'loss': f"{float(jnp.mean(loss_chunks[-1][-1])):.4f}"})
        if e != T:
            save(f"checkpoint_{e}.eqx", carry)
    pbar.close()
    all_losses = jnp.concatenate(loss_chunks)
    for s_i, d in enumerate(save_dirs):
        jnp.save(os.path.join(d, "losses.npy"), all_losses[:, s_i])
    save(f"model_{T}.eqx", carry)
    print(f"Saved to {save_dirs}")


def main(basedir,
        n_learner_seeds=5,
        n_pgm_seeds=1,
        pgm_type='random',
        mechanism_type='universal',
        n_nodes=None,
        node_dim=1,
        budget_params=100000,
        depth=8,
        batch_size=128,
        lr=1e-4,
        train_steps=5000,
        sparsity=0.3,
        weight_scale=None,     # linear-mechanism A scale (strong-coupling chains)
        noise_scale=None,      # linear-mechanism B scale
        root_scale=None,       # root B scale, decoupled from noise_scale (headroom chains)
        color_correlation=None,  # colored_mnist rho (defaults to 0.8 in the PGM)
        task='inference',  # versus 'causal'
        use_random_order=True,
        use_conv=False,        # conv (U-Net) blocks for image-shaped variables (static mode)
        conv_channels=64,
        modes=None,            # subset of modes to run, e.g. --modes static or "static,dynamic"
        seed_batch=False,      # train all learner seeds of a (mode, pgm_seed) together (vmap); not a config key
        steps_per_call=None,   # fused steps per jit call (default $SWEEP_STEPS_PER_CALL or 100; 0 = old loop)
        coupling=None,         # per-mode use_coupling, e.g. "static=c2ot,dynamic=c2ot,guided=hungarian"
        c2ot_target_ratio=None,  # C2OT adaptive-weight target (recorded only for modes using 'c2ot')
        ):
    BASE_DIR = Path(basedir)
    
    # Common Config
    base_config = {
        'batch_size': batch_size,
        'lr': lr,
        'budget_params': budget_params,
        'train_steps': train_steps,
        'depth': depth,
        'pgm_type': pgm_type,
        'sparsity': sparsity,
        'node_dim': node_dim,
        'task': task,
        'mechanism_type': mechanism_type,
        'weight_scale': weight_scale,
        'noise_scale': noise_scale,
    }
    # Only recorded when set: keeps config hashes of existing runs unchanged.
    if color_correlation is not None:
        base_config['color_correlation'] = color_correlation
    if root_scale is not None:
        base_config['root_scale'] = root_scale

    # Define Sweep
    learner_seeds = range(n_learner_seeds)
    pgm_seeds = range(n_pgm_seeds)
    all_modes = ['carefl', 'static', 'dynamic', 'guided', 'universal', 'dcm']
    if modes is None:
        modes = all_modes
    else:
        if isinstance(modes, str):
            modes = [m.strip() for m in modes.split(',')]
        modes = list(modes)
        unknown = set(modes) - set(all_modes)
        if unknown:
            raise ValueError(f"Unknown modes {unknown}; valid: {all_modes}")

    print(f'Running {len(modes)} modes with {len(learner_seeds)} learner seeds and {len(pgm_seeds)} pgm seeds, total {len(modes) * len(learner_seeds) * len(pgm_seeds)} runs.')

    coupling_by_mode = {}
    if coupling:
        for item in (coupling.split(',') if isinstance(coupling, str) else coupling):
            m, c = item.split('=')
            coupling_by_mode[m.strip()] = {'false': False, 'none': False, 'true': True}.get(c.strip().lower(), c.strip())

    def prepare(mode, l_seed, pgm_seed):
        """Build config + run dir for one run; None if the run dir already exists."""
        config = base_config.copy()
        config['mode'] = mode
        config['learner_seed'] = l_seed
        config['pgm_seed'] = pgm_seed
        config['task'] = task
        # Always recorded for the coupled modes, so a run's name says which coupling trained it
        # (runs from before the per-mode defaults, which used Sinkhorn OT, never collide).
        cpl = coupling_by_mode.get(mode, DEFAULT_COUPLING.get(mode))
        if cpl is not None:
            config['use_coupling'] = cpl
            if cpl == 'c2ot' and c2ot_target_ratio is not None:
                config['c2ot_target_ratio'] = c2ot_target_ratio

        # 1. Setup Data
        if n_nodes is not None:
            config['n_nodes'] = n_nodes
        if node_dim is not None:
            config['node_dim'] = node_dim

        pgm_seed = config['pgm_seed']
        pgm_type = config['pgm_type']

        pgm = create_pgm(pgm_type, config, random.PRNGKey(pgm_seed))
        structure = make_structure(pgm, config)

        # Conv blocks for image-shaped variables (consumed by static mode only).
        if use_conv:
            if not hasattr(pgm, 'variable_shapes'):
                raise ValueError(f"--use_conv requires the PGM to define variable_shapes; "
                                 f"{pgm_type} does not.")
            config['block_shapes'] = {k: list(v) for k, v in pgm.variable_shapes.items()}
            config['conv_channels'] = conv_channels

        print(structure)
        print('Tractable Distributions:')
        print([format_dist_str(targets=tc['targets'], conditions=tc['conditions']) for tc in structure.get_tractable_distributions()])

        # Add structure to config for saving/loading
        config['structure'] = {
            'dependencies': structure.dependencies,
            'block_order': structure.block_order,
            'dims': structure.dims
        }

        # Find hidden_dim that fits budget
        hidden_dim = get_hidden_dims(
            config,
            budget_params=config['budget_params'],
            structure=structure,
        )
        config['hidden_dim'] = hidden_dim # Record actual dim used
        print(f"Using hidden_dim: {hidden_dim}")

        # Construct a descriptive, unique run name from all hparams
        run_name = generate_run_name(config)
        save_dir = os.path.join(BASE_DIR, run_name)
        
        if os.path.exists(save_dir):
            print(f"Skipping {run_name}, already exists.")
            return None

        os.makedirs(save_dir, exist_ok=True)

        # Save Config before training
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        return config, save_dir, pgm, structure

    if seed_batch:
        groups = list(itertools.product(modes, pgm_seeds))
        if use_random_order:
            random_py.shuffle(groups)
        for mode, pgm_seed in groups:
            jobs = [r for r in (prepare(mode, l, pgm_seed) for l in learner_seeds) if r is not None]
            if jobs:
                train_runs_seed_batched([j[0] for j in jobs], [j[1] for j in jobs], jobs[0][2], jobs[0][3],
                                        steps_per_call=steps_per_call)
    else:
        hparams = list(itertools.product(modes, learner_seeds, pgm_seeds))
        if use_random_order:
            random_py.shuffle(hparams)
        for mode, l_seed, pgm_seed in hparams:
            r = prepare(mode, l_seed, pgm_seed)
            if r is not None:
                train_run(*r, steps_per_call=steps_per_call)

if __name__ == "__main__":
    import fire

    fire.Fire(main)
