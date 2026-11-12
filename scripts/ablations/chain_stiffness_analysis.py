"""Chain-depth -> stiffness -> accuracy analysis for static vs dynamic conditioning.

For each trained run in results/chain_stiffness/n{N} (linear-Gaussian chains,
inference task p(x0..x_{N-2} | x_{N-1})), computes the 2-Wasserstein (Bures)
distance between the flow posterior and the exact Gaussian posterior, under:
  - the adaptive solver (diffrax, tol 1e-5): end-to-end accuracy, and
  - fixed-step RK4 at n_steps in {4,8,16,32,64,128}: accuracy-vs-step-budget.
Stiffness hypothesis: dynamic mode needs more steps (and degrades with depth N)
because the conditioning signal resolves late in t; static stays flat.

Outputs results JSON + summary plots.

Run:  python scripts/ablations/chain_stiffness_analysis.py
"""
from pgm_paper_utils.paths import RESULTS
import os, sys, json, glob


import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from jax import random
import scipy.linalg
import matplotlib.pyplot as plt

from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.utils import make_structure

import os as _os
BASE = _os.environ.get('CHAIN_BASE', f'{RESULTS}/chain_stiffness')
NS = [int(x) for x in _os.environ.get('CHAIN_NS', '2,4,8,16,32').split(',')]
SOLVERS = ([('diffrax', 50)] + [('rk4', k) for k in (4, 8, 16, 32, 64, 128)]) if not _os.environ.get('CHAIN_ADAPTIVE_ONLY') else [('diffrax', 50)]
N_COND, N_POST = int(_os.environ.get('CHAIN_NCOND', 5)), 2000


def bures_w2(m1, C1, m2, C2):
    m1, m2 = np.atleast_1d(m1), np.atleast_1d(m2)
    C1, C2 = np.atleast_2d(C1), np.atleast_2d(C2)
    sC2 = scipy.linalg.sqrtm(C2).real
    cross = scipy.linalg.sqrtm(sC2 @ C1 @ sC2).real
    return float(np.sqrt(max(0.0, np.sum((m1 - m2) ** 2)
                             + np.trace(C1 + C2 - 2 * cross))))


def analytic_posterior(mu, Sigma, obs_idx, tgt_idx, c):
    S_tt = Sigma[np.ix_(tgt_idx, tgt_idx)]
    S_to = Sigma[np.ix_(tgt_idx, obs_idx)]
    S_oo = Sigma[np.ix_(obs_idx, obs_idx)]
    K = S_to @ np.linalg.inv(S_oo)
    return mu[tgt_idx] + K @ (c - mu[obs_idx]), S_tt - K @ S_to.T


def main():
    rows = []
    if _os.environ.get('CHAIN_MERGE'):
        for f in sorted(glob.glob(f'{BASE}/stiffness_results*.json')):
            rows.extend(json.load(open(f)))
        return finish(rows)
    # CHAIN_TAG decouples the output name from the default: the 5-draw and
    # 25-draw experiments share BASE, and identical filenames already caused
    # one overwrite and one bogus resume-skip.
    tag = _os.environ.get('CHAIN_TAG') or '-'.join(map(str, NS))
    out = f'{BASE}/stiffness_results_{tag}.json'
    done = set()
    if os.path.exists(out):
        rows = json.load(open(out))
        done = {(r['n'], r['mode'], r['seed'], r['integrator'], r['n_steps'])
                for r in rows}
        print(f"resuming: {len(rows)} rows already in {out}", flush=True)
    for n in NS:
        for run_dir in sorted(glob.glob(f'{BASE}/n{n}/run_*')):
            cfg = json.load(open(f'{run_dir}/config.json'))
            if not os.path.exists(f'{run_dir}/model_30000.eqx'):
                continue
            todo = [(integ, k) for integ, k in SOLVERS
                    if (n, cfg['mode'], cfg['learner_seed'], integ, k) not in done]
            if not todo:
                continue
            pgm = create_pgm('chain', cfg, random.PRNGKey(cfg['pgm_seed']))
            structure = make_structure(pgm, cfg)
            skel, nskel, _, _ = create_learner(cfg, structure=structure)
            model, normaliser = eqx.tree_deserialise_leaves(
                f'{run_dir}/model_30000.eqx', (skel, nskel))

            order = pgm.structure.block_order          # causal order x0..x_{n-1}
            targets, conditions = pgm.get_inference_targets_and_conditions()
            tgt_list = [b for b in order if b in targets]
            obs_list = [b for b in order if b in conditions]
            tgt_idx = [order.index(b) for b in tgt_list]
            obs_idx = [order.index(b) for b in obs_list]

            # joint gaussian of the true PGM (once per pgm; cheap)
            J = pgm.b_sample(random.PRNGKey(0), 100_000)
            X = np.concatenate([np.atleast_2d(np.asarray(J[b]).T).T.reshape(100_000, -1)
                                for b in order], axis=1)
            mu, Sigma = X.mean(0), np.cov(X, rowvar=False)

            cond_draws = [{b: J[b][i] for b in obs_list} for i in range(N_COND)]

            # Flat-index range of the root block x0 (order is the causal
            # order, so the root is order[0]) within the target vector, for
            # the root-only W2.
            tdims = [np.asarray(J[b]).reshape(100_000, -1).shape[1] for b in tgt_list]
            toff = np.concatenate([[0], np.cumsum(tdims)])
            ri = tgt_list.index(order[0])
            rsl = slice(int(toff[ri]), int(toff[ri + 1]))

            # One compiled function per (run, solver): conditions enter as
            # traced ARGUMENTS. The old per-draw closure retraced/recompiled
            # the full sampling graph 25x per run, which both crawled and
            # grew the never-evicted executable cache until the host OOMed.
            tgt_fs = frozenset(tgt_list)
            b_sample_cond = eqx.filter_jit(
                lambda d, cv, ks: jax.vmap(lambda kk: d.sample(
                    kk, conditions=cv, targets=tgt_fs))(ks))

            for integ, k in todo:
                c2 = dict(cfg); c2['integrator_name'] = integ; c2['n_steps'] = k
                dist = create_distribution(c2, model, normaliser)
                w2s, w2s_root, w2s_base, w2s_base_root = [], [], [], []
                for cv in cond_draws:
                    ks = random.split(random.PRNGKey(7), N_POST)
                    S = b_sample_cond(dist, cv, ks)
                    Sf = np.concatenate(
                        [np.asarray(S[b]).reshape(N_POST, -1) for b in tgt_list], axis=1)
                    c_vec = np.concatenate([np.atleast_1d(np.asarray(cv[b])) for b in obs_list])
                    m_p, C_p = analytic_posterior(mu, Sigma, obs_idx, tgt_idx, c_vec)
                    Sm, Sc = Sf.mean(0), np.cov(Sf, rowvar=False)
                    Sc = np.atleast_2d(Sc)
                    w2s.append(bures_w2(Sm, Sc, m_p, C_p))
                    w2s_root.append(bures_w2(Sm[rsl], Sc[rsl, rsl], m_p[rsl], C_p[rsl, rsl]))
                    # prior baseline: the score of ignoring the evidence
                    m_0, C_0 = mu[tgt_idx], Sigma[np.ix_(tgt_idx, tgt_idx)]
                    w2s_base.append(bures_w2(m_0, C_0, m_p, C_p))
                    w2s_base_root.append(bures_w2(m_0[rsl], C_0[rsl, rsl], m_p[rsl], C_p[rsl, rsl]))
                rows.append(dict(n=n, mode=cfg['mode'], seed=cfg['learner_seed'],
                                 integrator=integ, n_steps=k, w2=float(np.mean(w2s)),
                                 w2_root=float(np.mean(w2s_root)),
                                 w2_prior_baseline=float(np.mean(w2s_base)),
                                 w2_root_prior_baseline=float(np.mean(w2s_base_root))))
                print(f"n={n} {cfg['mode']} s{cfg['learner_seed']} {integ}-{k}: "
                      f"W2={rows[-1]['w2']:.4f} root={rows[-1]['w2_root']:.4f}", flush=True)
                json.dump(rows, open(out, 'w'), indent=1)   # incremental save
            jax.clear_caches()

    print(f"saved {out}")
    if len(NS) < 5:
        return
    finish(rows)


def finish(rows):

    # --- plots --------------------------------------------------------------
    import collections
    def agg(pred):
        d = collections.defaultdict(list)
        for r in rows:
            if pred(r):
                d[r['n']].append(r['w2'])
        ns = sorted(d)
        return ns, [np.mean(d[k]) for k in ns]

    def agg_std(pred):
        d = collections.defaultdict(list)
        for r in rows:
            if pred(r):
                d[r['n']].append(r['w2'])
        ns = sorted(d)
        return ns, [np.mean(d[k]) for k in ns], [np.std(d[k]) for k in ns]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mode, col in [('static', 'tab:blue'), ('dynamic', 'tab:red')]:
        ns, w, sd = agg_std(lambda r: r['mode'] == mode and r['integrator'] == 'diffrax')
        axes[0].errorbar(ns, w, yerr=sd, fmt='o-', color=col, label=mode, capsize=3)
    axes[0].set(xlabel='chain length $n$', ylabel='$W_2$ to exact posterior',
                title='Adaptive solver (tol $10^{-5}$)', xscale='log')
    axes[0].legend()

    for ax, mode in zip(axes[1:], ['static', 'dynamic']):
        for n in NS:
            ks = sorted({r['n_steps'] for r in rows if r['integrator'] == 'rk4'})
            vals = [[r['w2'] for r in rows
                     if r['mode'] == mode and r['integrator'] == 'rk4'
                     and r['n'] == n and r['n_steps'] == k] for k in ks]
            w = [np.mean(v) for v in vals]
            sd = [np.std(v) for v in vals]
            line, = ax.plot(ks, w, 'o-', label=f'n={n}')
            ax.fill_between(ks, [m - s2 for m, s2 in zip(w, sd)],
                            [m + s2 for m, s2 in zip(w, sd)],
                            color=line.get_color(), alpha=0.18, linewidth=0)
        ax.set(xlabel='RK4 steps', ylabel='$W_2$', title=f'{mode} conditioning',
               xscale='log', yscale='log')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{BASE}/stiffness_curves.png', dpi=200)
    print(f"saved {BASE}/stiffness_curves.png")


if __name__ == "__main__":
    main()
