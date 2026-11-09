"""Capacity probe for Colored-MNIST: does a bigger MLP budget fix the
under-concentrated conditional p(digit | color)?

Trains a single static inference-aligned flow at the given budget/steps, then
compares the model's digit marginal under conditioning on palette[5] against
the exact posterior (0.79 at digit 5 for rho=0.8 with MNIST class frequencies).

Run from repository root:
    ~/my_jax_env/bin/python scripts/ablations/colored_mnist_capacity_probe.py \
        --budget_params 32000000 --train_steps 150000
"""
from pgm_paper_utils.paths import RESULTS
import os
import sys
import json

# NOTE: do NOT set --xla_gpu_autotune_level=0 here — it slows conv training ~10x.
# It is only needed as a numerics workaround on some eval paths (see
# evaluate_sweep.py / vis_colored_mnist_flows.py); if the post-training eval
# below hits an XLA autotuning error, re-run just the eval with that flag.

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from structflow.pgms.colored_mnist import DIGIT_PALETTE
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.infrastructure import get_hidden_dims
from pgm_paper_utils.utils import make_structure
import sweep  # scripts/sweep.py: train_run

COND_DIGIT = 5


def main(budget_params=32_000_000,
         train_steps=150_000,
         mode='static',
         task='inference',
         depth=8,
         lr=1e-4,
         batch_size=128,
         use_conv=False,
         conv_channels=64,
         out_dir=f'{RESULTS}/colored_mnist_probe'):
    config = {
        'batch_size': batch_size,
        'lr': lr,
        'budget_params': budget_params,
        'train_steps': train_steps,
        'depth': depth,
        'pgm_type': 'colored_mnist',
        'sparsity': 0.3,
        'node_dim': 1,
        'task': task,
        'mechanism_type': 'universal',
        'mode': mode,
        'learner_seed': 0,
        'pgm_seed': 0,
    }

    pgm = create_pgm('colored_mnist', config, random.PRNGKey(0))
    structure = make_structure(pgm, config)
    if use_conv:
        config['block_shapes'] = {k: list(v) for k, v in pgm.variable_shapes.items()}
        config['conv_channels'] = conv_channels
    config['structure'] = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims,
    }
    config['hidden_dim'] = get_hidden_dims(config, budget_params=budget_params,
                                           structure=structure)
    print(f"Probe: mode={mode}, task={task}, budget={budget_params:,}, "
          f"steps={train_steps}, hidden_dim={config['hidden_dim']}")

    save_dir = os.path.join(
        out_dir,
        f"run_probe_{mode}_{task}_b{budget_params}_s{train_steps}"
        + (f"_conv{conv_channels}" if use_conv else ""))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    sweep.train_run(config, save_dir, pgm, structure)

    # --- evaluate the conditional digit marginal --------------------------
    model_skeleton, normaliser_skeleton, _, _ = create_learner(config, structure=structure)
    import equinox as eqx
    ckpts = [f for f in os.listdir(save_dir) if f.endswith('.eqx')]
    steps = [int(f.split('_')[-1].split('.')[0]) for f in ckpts]
    ckpt = ckpts[int(np.argmax(steps))]
    model, normaliser = eqx.tree_deserialise_leaves(
        os.path.join(save_dir, ckpt), (model_skeleton, normaliser_skeleton))
    config.setdefault('n_steps', 50)
    dist = create_distribution(config, model, normaliser)

    c = jnp.array(DIGIT_PALETTE[COND_DIGIT])
    n = 512
    s = dist.b_sample(random.PRNGKey(1), n, conditions={'color': c},
                      targets=frozenset({'digit', 'image'}))
    hist = np.bincount(np.argmax(np.array(s['digit']), axis=-1), minlength=10) / n

    true_post = np.array(jax.nn.softmax(
        pgm._digit_posterior_logits(c, training=False)))
    print(f"model digit marginal | color=palette[{COND_DIGIT}]: "
          f"{np.array2string(hist, precision=2)}")
    print(f"exact posterior:                     "
          f"{np.array2string(true_post, precision=2)}")
    print(f"PROBE RESULT: model peak {hist[COND_DIGIT]:.3f} vs exact "
          f"{true_post[COND_DIGIT]:.3f} at digit {COND_DIGIT}")


if __name__ == "__main__":
    fire.Fire(main)
