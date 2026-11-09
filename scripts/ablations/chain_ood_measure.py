"""OOD + stiffness measurements for dynamic conditioning on chains.

For each dynamic run in $CHAIN_BASE/n{N}:
  (i) z-score of the backward-integrated condition path against the
      training-interpolant marginal x_t = t*x1 + (1-t)*e, on a t grid;
  (ii) per-block max |v| along conditional integration vs joint integration
      (post-fix dist.sample path) — late-t velocity growth with block depth
      is the stiffness signature.

Run: CHAIN_BASE=.../chain_stiffness_strong ~/my_jax_env/bin/python \
     scripts/ablations/chain_ood_measure.py
"""
from pgm_paper_utils.paths import RESULTS
import os, sys, json, glob


import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from jax import random
import matplotlib.pyplot as plt

from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.utils import make_structure

BASE = os.environ.get('CHAIN_BASE', f'{RESULTS}/chain_stiffness_strong')
NS = [2, 4, 8, 16, 32]
TGRID = [0.25, 0.5, 0.75, 0.9]


def main():
    rows = []
    np.random.seed(0)
    for n in NS:
        run_dirs = (sorted(glob.glob(f'{BASE}/n{n}/run_*modedynamic*'))
                    + sorted(glob.glob(f'{BASE}/n{n}/run_*modestatic*')))
        for rd in run_dirs:
            cfg = json.load(open(f'{rd}/config.json'))
            pgm = create_pgm('chain', cfg, random.PRNGKey(cfg['pgm_seed']))
            st = make_structure(pgm, cfg)
            sk, nsk, _, _ = create_learner(cfg, structure=st)
            m, nm = eqx.tree_deserialise_leaves(f'{rd}/model_30000.eqx', (sk, nsk))
            dist = create_distribution(cfg, m, nm)
            inner = getattr(dist, 'dist', dist)
            flow = inner.flow
            net = flow.velocity_fn
            is_static = cfg['mode'] == 'static'
            order = net.structure.block_order
            offs = None if is_static else net.block_offsets
            cname = order[0]                     # compiled order: condition first
            tname = order[-1]

            J = pgm.b_sample(random.PRNGKey(0), 20000)
            cobs = jnp.asarray(np.percentile(np.asarray(J[cname]).ravel(), 90))[None]
            c_norm = nm.normalise({cname: cobs})[cname]

            if is_static:
                # Static null: per-block max|v| via the dict-interface net,
                # joint (ancestral, no conditions) vs conditional (condition
                # block FIXED at its resolved value -- static semantics).
                T = 100
                def static_maxv(key, conditional):
                    x = {b: random.normal(random.fold_in(key, i), (net.structure.dims[b],))
                         for i, b in enumerate(order)}
                    if conditional:
                        x[cname] = c_norm
                    vmax = {b: jnp.zeros(net.structure.dims[b]) for b in order}
                    def body(i, carry):
                        x, vmax = carry
                        v = net(x, i / T)
                        x2 = {b: x[b] + v[b] / T for b in order}
                        if conditional:
                            x2[cname] = c_norm
                        vmax2 = {b: jnp.maximum(vmax[b], jnp.abs(v[b])) for b in order}
                        return (x2, vmax2)
                    _, vmax = jax.lax.fori_loop(0, T, body, (x, vmax))
                    return vmax
                for conditional in (False, True):
                    V = jax.vmap(lambda k: static_maxv(k, conditional))(
                        random.split(random.PRNGKey(5), 200))
                    per_block = {b: float(np.asarray(V[b]).mean(0).max()) for b in order}
                    rows.append(dict(n=n, mode='static', seed=cfg['learner_seed'],
                                     kind='conditional' if conditional else 'joint',
                                     z_scores=None, max_v=per_block))
                    print(f"n={n} static {'cond' if conditional else 'joint'}: "
                          f"max|v| first/last block = "
                          f"{per_block[order[1] if len(order)>1 else order[0]]:.2f}"
                          f"/{per_block[tname]:.2f}", flush=True)
                continue

            # (i) condition-path z-scores vs training interpolant
            ctraj = np.asarray(flow._get_condition_trajectories({cname: c_norm})[cname]).ravel()
            xn = np.asarray(nm.normalise({cname: jnp.asarray(J[cname])})[cname]).ravel()[:5000]
            e = np.random.randn(5000)
            zs = {}
            for tq in TGRID:
                Xt = tq * xn + (1 - tq) * e
                i = int(tq * (len(ctraj) - 1))
                zs[tq] = float((ctraj[i] - Xt.mean()) / Xt.std())

            # (ii) per-block max |v| along conditional vs joint integration
            T = 100
            ctraj_j = jnp.asarray(ctraj)

            def traj_maxv(key, conditional):
                x = {b: random.normal(random.fold_in(key, i), (net.structure.dims[b],))
                     for i, b in enumerate(order)}
                if conditional:
                    x[cname] = ctraj_j[0][None]
                flat = jnp.concatenate([x[b] for b in order])
                def body(i, carry):
                    flat, vmax = carry
                    if_cond = ctraj_j[i][None]
                    s, ee = offs[cname]
                    flat = jax.lax.cond(jnp.array(conditional),
                                        lambda f: f.at[s:ee].set(if_cond),
                                        lambda f: f, flat)
                    v = net(flat, i / T)
                    vmax = jnp.maximum(vmax, jnp.abs(v))
                    return (flat + v / T, vmax)
                (_, vmax) = jax.lax.fori_loop(0, T, body, (flat, jnp.zeros_like(flat)))
                return vmax

            for conditional in (False, True):
                V = jax.vmap(lambda k: traj_maxv(k, conditional))(
                    random.split(random.PRNGKey(5), 200))
                vm = np.asarray(V).mean(0)
                per_block = {b: float(vm[offs[b][0]:offs[b][1]].max()) for b in order}
                rows.append(dict(n=n, mode=cfg['mode'], seed=cfg['learner_seed'],
                                 kind='conditional' if conditional else 'joint',
                                 z_scores=zs, max_v=per_block))
                print(f"n={n} {cfg['mode']} {'cond' if conditional else 'joint'}: "
                      f"max|v| first/last block = {per_block[order[1] if len(order)>1 else order[0]]:.2f}"
                      f"/{per_block[tname]:.2f}  z={ {k: round(v,2) for k,v in zs.items()} }",
                      flush=True)

    out = f'{BASE}/ood_measures_allseeds.json'
    json.dump(rows, open(out, 'w'), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
