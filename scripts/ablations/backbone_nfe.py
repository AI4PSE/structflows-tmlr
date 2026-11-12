"""Backbone ablation, step 4: accuracy-vs-NFE sweep (static CNF vs per-node DDIM).

For each trained nonlinear4 causal run (modes 'static' and 'dcm',
pgm_seed 0..4, learner_seed 0) and each network-call budget
NFE in {4, 8, 16, 32, 64, 128}, computes:

  (a) interventional SWD  — mean SWD over up to 3 conditional causal queries
      via pgm_paper_utils.eval.evaluate_distribution (task='causal'),
  (b) counterfactual SWD  — Counterfactual_SWD from the same calls
      (mean over finite values),
  (c) abduction reconstruction error — mean_batch ||invert(forward(u)) - u||_2
      with u ~ N(0, I) per block, computed on the *inner* (normalised-space)
      distribution so both backbones are compared in the same space and the
      normaliser roundtrip cannot confound the measurement.

CRITICAL NFE accounting (fairness control): the x-axis is measured NETWORK CALLS per sample, not solver steps.
The static backbone uses fixed-step RK4 (4 evals/step) -> n_steps = NFE // 4;
the DCM backbone uses DDIM (1 eval/step) -> n_steps = NFE. The recorded 'nfe'
field is the ACTUAL call count: 4 * n_steps (static) or n_steps (dcm).
Comparing raw step counts would hand the CNF a 4x hidden budget.

The DCM inversion is the published DDIM inversion (eps re-used at the current
point, first-order bias) — that bias is the object of study; do not "fix" it.

Output: results/nonlinear4_kr/backbone_nfe.json (static from nonlinear4_kr, uncoupled; DCM from nonlinear4)
Plot:   scripts/plotters/plot_backbone_nfe.py

Run:  python scripts/ablations/backbone_nfe.py
"""
from pgm_paper_utils.paths import RESULTS
import os
import sys
import json
import glob


import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random

from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.utils import make_structure, get_test_queries
from pgm_paper_utils.eval import evaluate_distribution

BASE = os.environ.get('NFE_BASE', f'{RESULTS}/nonlinear4')
# static runs trained with the paper's coupling for static (none); DCM has no coupling option
STATIC_BASE = os.environ.get('NFE_STATIC_BASE', f'{RESULTS}/nonlinear4_kr')
OUT = os.environ.get('NFE_OUT', f'{STATIC_BASE}/backbone_nfe.json')
CHECKPOINT = 'model_100000.eqx'
NFES = [4, 8, 16, 32, 64, 128]
PGM_SEEDS = set(range(5))
LEARNER_SEED = 0
NUM_SAMPLES = 500
MAX_QUERIES = 3          # conditional causal queries per run (cost control)
ABDUCTION_BATCH = 256
EVAL_SEED = 0


def find_runs():
    """(run_dir, config) for causal static/dcm runs in the sweep subset."""
    runs = []
    for d in sorted(glob.glob(f'{BASE}/run_*')) + sorted(glob.glob(f'{STATIC_BASE}/run_*')):
        cfg_path = os.path.join(d, 'config.json')
        if not os.path.exists(cfg_path) or not os.path.exists(os.path.join(d, CHECKPOINT)):
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get('mode') not in ('static', 'dcm'):
            continue
        base = os.path.dirname(d)
        if cfg['mode'] == 'dcm' and base != BASE.rstrip('/'):
            continue
        if cfg['mode'] == 'static' and (base != STATIC_BASE.rstrip('/') or cfg.get('use_coupling') not in (False, 'false')):
            continue
        if cfg.get('task') != 'causal':
            continue
        if cfg.get('learner_seed') != LEARNER_SEED or cfg.get('pgm_seed') not in PGM_SEEDS:
            continue
        runs.append((d, cfg))
    return runs


def solver_settings(mode, nfe):
    """Map an NFE budget to solver config; return (config-overrides, actual NFE).

    static: fixed-step RK4 = 4 network evals per step -> n_steps = NFE // 4
            (n_steps = 1 at NFE = 4; actual = 4 * n_steps).
    dcm:    DDIM = 1 network eval per step -> n_steps = NFE (actual = n_steps).
    """
    if mode == 'static':
        n_steps = max(1, nfe // 4)
        return {'integrator_name': 'rk4', 'n_steps': n_steps}, 4 * n_steps
    elif mode == 'dcm':
        return {'n_steps': nfe}, nfe
    raise ValueError(f"Unknown mode: {mode}")


def abduction_error(dist, key, batch=ABDUCTION_BATCH):
    """mean_batch || invert(forward(u)) - u ||_2 in NORMALISED space.

    Unwraps NormalisedDistribution and works on the inner distribution: both
    backbones expose the CounterfactualMixin hooks `_predict` (noise -> data,
    full topological simulation) and `transform_to_noise` (data -> noise,
    the abduction map). Base noise is exactly the inner base distribution
    (standard normal per block), so the roundtrip measures pure inversion
    error at fixed weights, with no normaliser effects.
    """
    inner = getattr(dist, 'dist', dist)
    structure = inner._cf_structure
    order = list(structure.block_order)

    keys = random.split(key, len(order))
    u = {b: random.normal(k, (batch, structure.dims[b]))
         for b, k in zip(order, keys)}

    @eqx.filter_jit
    def _roundtrip(mod, u_batch):
        def one(uu):
            fwd, _ = mod._predict(uu, {}, order)   # noise -> data (all blocks)
            return mod.transform_to_noise(fwd)      # data -> noise (abduction)
        return jax.vmap(one)(u_batch)

    u_rec = _roundtrip(inner, u)
    diff = jnp.concatenate(
        [(u_rec[b] - u[b]).reshape(batch, -1) for b in order], axis=1)
    return float(jnp.mean(jnp.linalg.norm(diff, axis=1)))


def main():
    runs = find_runs()
    by_mode = {}
    for _, cfg in runs:
        by_mode.setdefault(cfg['mode'], []).append(cfg['pgm_seed'])
    print(f"Found {len(runs)} runs: "
          + ", ".join(f"{m}: seeds {sorted(s)}" for m, s in sorted(by_mode.items())))

    rows = []
    for run_dir, cfg in runs:
        mode, pgm_seed = cfg['mode'], cfg['pgm_seed']
        print(f"\n=== {mode} pgm_seed={pgm_seed} ({os.path.basename(run_dir)[-8:]}) ===",
              flush=True)

        pgm = create_pgm(cfg['pgm_type'], cfg, random.PRNGKey(pgm_seed))
        structure = make_structure(pgm, cfg)
        skel, nskel, _, _ = create_learner(cfg, structure=structure)
        model, normaliser = eqx.tree_deserialise_leaves(
            os.path.join(run_dir, CHECKPOINT), (skel, nskel))

        # Conditional causal queries only ('|' present), capped for cost.
        # get_test_queries returns a deterministic (sorted) list, so the same
        # queries are used at every NFE and for both backbones on this graph.
        queries = [q for q in get_test_queries(pgm, structure) if '|' in q[2]]
        queries = queries[:MAX_QUERIES]
        print(f"  queries: {[q[2] for q in queries]}", flush=True)

        for nfe in NFES:
            overrides, actual_nfe = solver_settings(mode, nfe)
            c2 = dict(cfg)
            c2.update(overrides)
            dist = create_distribution(c2, model, normaliser)

            swds, cf_swds = [], []
            for targets, conditions, q_str in queries:
                m = evaluate_distribution(
                    pgm, dist, targets, conditions,
                    seed=EVAL_SEED, num_samples=NUM_SAMPLES, task='causal')
                swds.append(m['SWD'])
                if np.isfinite(m['Counterfactual_SWD']):
                    cf_swds.append(m['Counterfactual_SWD'])

            ab_err = abduction_error(dist, random.PRNGKey(1234 + pgm_seed))

            row = dict(
                backbone=mode,
                pgm_seed=pgm_seed,
                learner_seed=LEARNER_SEED,
                nfe=actual_nfe,
                swd_interventional=float(np.mean(swds)) if swds else float('nan'),
                swd_counterfactual=float(np.mean(cf_swds)) if cf_swds else float('nan'),
                abduction_err=ab_err,
            )
            rows.append(row)
            print(f"  [{mode} s{pgm_seed} NFE={actual_nfe:>3}] "
                  f"SWD_int={row['swd_interventional']:.4f} "
                  f"SWD_cf={row['swd_counterfactual']:.4f} "
                  f"abduction={row['abduction_err']:.4f}", flush=True)

            # Incremental save: the sweep is long, keep partial results.
            json.dump(rows, open(OUT, 'w'), indent=1)

    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"\nsaved {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
