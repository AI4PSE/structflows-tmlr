"""Validation of the NUTS ground-truth references (reviewer-facing check).

The paper's reference posterior ensemble for non-ancestral queries is N
independent chains, each contributing ONE post-warmup draw (window adaptation
w/ 250 steps, then a single NUTS step), vmapped over keys. Independent draws
mean no autocorrelation issues, but full reliance on warmup convergence from a
prior init. Three checks:

  1. ANCHOR: on linear-Gaussian PGMs the exact posterior is available; report
     Bures-W2 and per-marginal moment errors of the reference protocol.
  2. WARMUP SENSITIVITY: the same ensemble at 250 / 1000 / 4000 warmup steps;
     if the ensembles agree (SWD at the scale of MC noise), warmup converged.
  3. SBC (Talts et al. 2018): draw (theta~, y) from the joint, build the
     reference ensemble for p(targets | y) with the paper's protocol, rank
     theta~ among L draws. Calibrated sampler <=> ranks uniform on {0..L}.
     Because draws are independent chains, no thinning is needed.

Conditions enter the jitted runner as traced arguments (one trace per
(pgm, warmup) pair -- not per replication).

Run: CUDA_VISIBLE_DEVICES=0 ~/my_jax_env/bin/python scripts/ablations/nuts_validation.py
"""
from pgm_paper_utils.paths import RESULTS
import os, sys, json
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

import numpy as np
import jax, jax.numpy as jnp
from jax import random
import blackjax
import scipy.linalg
from scipy.stats import chisquare
from pgm_paper_utils.pgm_factory import create_pgm

OUT = f'{RESULTS}/nuts_validation.json'
N_REF = 2000          # ensemble size for anchor/warmup checks
N_COND = 3            # condition draws per PGM
SBC_M = 300           # SBC replications
SBC_L = 99            # reference draws per replication (ranks in 0..99)
WARMUPS = (250, 1000, 4000)


def make_runner(pgm, targets, conditions, n_warmup):
    """One jitted function: (condition value dict, keys) -> ensemble of single
    post-warmup NUTS draws. Conditions are traced args (no per-value retrace)."""
    unknowns = sorted(set(pgm.node_names) - set(conditions))
    tgt = sorted(targets)

    def joint_lp(unk, cond):
        full = {**cond, **unk}
        lp = 0.0
        for node in pgm.topo_order:
            parents_flat = pgm._get_parents_flat(node, full)
            x = jnp.atleast_1d(full[node])
            base = pgm.base_distributions.get(node) if pgm.base_distributions else None
            lp += pgm.mechanisms[node].log_prob(parents_flat, x, base_dist=base)
        return lp

    def one_chain(cond, key):
        k_init, k_warm, k_step = random.split(key, 3)
        init_full = pgm._sample_ancestral(k_init, {}, set(unknowns))
        init = {k: init_full[k] for k in unknowns}
        lp = lambda u: joint_lp(u, cond)
        window = blackjax.window_adaptation(blackjax.nuts, lp)
        (state, params), _ = window.run(k_warm, init, num_steps=n_warmup)
        state, _ = blackjax.nuts(lp, **params).step(k_step, state)
        return jnp.concatenate([jnp.atleast_1d(state.position[k]) for k in tgt])

    return jax.jit(jax.vmap(one_chain, in_axes=(None, 0))), tgt


def bures_w2(m1, C1, m2, C2):
    m1, m2 = np.atleast_1d(m1), np.atleast_1d(m2)
    C1, C2 = np.atleast_2d(C1), np.atleast_2d(C2)
    sC2 = scipy.linalg.sqrtm(C2).real
    cross = scipy.linalg.sqrtm(sC2 @ C1 @ sC2).real
    return float(np.sqrt(max(0.0, np.sum((m1 - m2) ** 2)
                             + np.trace(C1 + C2 - 2 * cross))))


def analytic_posterior(pgm, cond, targets, n=200_000):
    """Gaussian conditional from a large joint sample (exact for linear PGMs)."""
    J = pgm.b_sample(random.PRNGKey(0), n)
    names = sorted(pgm.node_names)
    X = np.concatenate([np.asarray(J[k]).reshape(n, -1) for k in names], axis=1)
    mu, Sig = X.mean(0), np.cov(X, rowvar=False)
    dims = np.cumsum([0] + [np.asarray(J[k]).reshape(n, -1).shape[1] for k in names])
    idx = {k: list(range(dims[i], dims[i + 1])) for i, k in enumerate(names)}
    ti = sum((idx[k] for k in sorted(targets)), [])
    oi = sum((idx[k] for k in sorted(cond)), [])
    c = np.concatenate([np.atleast_1d(np.asarray(cond[k])) for k in sorted(cond)])
    S_tt = Sig[np.ix_(ti, ti)]; S_to = Sig[np.ix_(ti, oi)]; S_oo = Sig[np.ix_(oi, oi)]
    K = S_to @ np.linalg.inv(S_oo)
    return mu[ti] + K @ (c - mu[oi]), S_tt - K @ S_to.T


def swd(a, b, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    P = rng.normal(size=(d, n_proj)); P /= np.linalg.norm(P, axis=0)
    pa, pb = a @ P, b @ P
    m = min(len(pa), len(pb))
    qa = np.sort(pa[:m], axis=0); qb = np.sort(pb[:m], axis=0)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def main():
    results = []
    if os.path.exists(OUT):
        results = json.load(open(OUT))
        print(f'resuming: {len(results)} rows in {OUT}', flush=True)
    done = {r['name'] for r in results}
    cases = ([('linear', s, 4, 1) for s in range(2)]
             + [('universal', s, 4, 1) for s in range(3)]
             + [('linear', 0, 8, 1), ('universal', 0, 8, 1)]
             + [('linear', 0, 8, 8), ('universal', 0, 8, 8),
                ('linear', 0, 16, 8), ('universal', 0, 16, 8)])
    for mech, seed, n, nd in cases:
        name = (f'{mech}-n{n}-s{seed}' if nd == 1
                else f'{mech}-n{n}-d{nd}-s{seed}')
        if name in done:
            continue
        cfg = dict(pgm_type='random', mechanism_type=mech, n_nodes=n,
                   node_dim=nd, sparsity=0.3)
        pgm = create_pgm('random', cfg, random.PRNGKey(seed))
        targets, conditions = pgm.get_inference_targets_and_conditions()
        if not conditions or not targets:
            continue
        J = pgm.b_sample(random.PRNGKey(100 + seed), N_COND)
        cond_draws = [{k: jnp.atleast_1d(J[k][i]) for k in conditions}
                      for i in range(N_COND)]
        row = dict(name=name, targets=sorted(targets),
                   conditions=sorted(conditions))

        # --- warmup sensitivity (and anchor for linear) --------------------
        # high-dimensional cases: lighter grid (4x warmup check, smaller ensemble)
        warmups = WARMUPS if n * nd <= 8 else (250, 1000)
        n_ref = N_REF if n * nd <= 64 else 1000
        ensembles = {}
        for w in warmups:
            run, tgt = make_runner(pgm, targets, conditions, w)
            per_cond = []
            for cv in cond_draws:
                keys = random.split(random.PRNGKey(7), n_ref)
                per_cond.append(np.asarray(run(cv, keys)))
            ensembles[w] = per_cond
        for a, b in zip(warmups[:-1], warmups[1:]):
            row[f'swd_{a}_vs_{b}'] = float(np.mean(
                [swd(x, y) for x, y in zip(ensembles[a], ensembles[b])]))
        # MC-noise floor: two half-splits of the same ensemble
        row['swd_mc_floor'] = float(np.mean(
            [swd(e[:n_ref // 2], e[n_ref // 2:]) for e in ensembles[250]]))

        if mech == 'linear':
            w2s, merr, verr = [], [], []
            for cv, e in zip(cond_draws, ensembles[250]):
                m_p, C_p = analytic_posterior(pgm, cv, targets)
                w2s.append(bures_w2(e.mean(0), np.cov(e, rowvar=False),
                                    m_p, C_p))
                merr.append(np.max(np.abs(e.mean(0) - m_p)))
                verr.append(np.max(np.abs(e.var(0) - np.diag(np.atleast_2d(C_p)))))
            row['anchor_w2'] = float(np.mean(w2s))
            row['anchor_w2_rel'] = float(np.mean(w2s) / np.sqrt(
                np.trace(C_p)))
            row['anchor_max_mean_err'] = float(np.mean(merr))
            row['anchor_max_var_err'] = float(np.mean(verr))

        # --- SBC ------------------------------------------------------------
        run, tgt = make_runner(pgm, targets, conditions, 250)
        tdim = sum(np.atleast_1d(J[k][0]).shape[0] for k in sorted(targets))
        ranks = np.zeros((SBC_M, tdim), dtype=int)
        Jsbc = pgm.b_sample(random.PRNGKey(999), SBC_M)
        for m in range(SBC_M):
            cv = {k: jnp.atleast_1d(Jsbc[k][m]) for k in conditions}
            theta = np.concatenate([np.atleast_1d(np.asarray(Jsbc[k][m]))
                                    for k in sorted(targets)])
            keys = random.split(random.fold_in(random.PRNGKey(11), m), SBC_L)
            draws = np.asarray(run(cv, keys))
            ranks[m] = (draws < theta[None, :]).sum(0)
        # chi-square uniformity per dim, 10 bins
        pvals = []
        for d in range(tdim):
            hist, _ = np.histogram(ranks[:, d], bins=10, range=(0, SBC_L + 1))
            pvals.append(float(chisquare(hist).pvalue))
        row['sbc_pvals'] = pvals
        row['sbc_min_p'] = float(min(pvals))
        results.append(row)
        json.dump(results, open(OUT, 'w'), indent=1)   # incremental save
        print(json.dumps(row), flush=True)
        jax.clear_caches()

    print('saved', OUT)


if __name__ == '__main__':
    main()
