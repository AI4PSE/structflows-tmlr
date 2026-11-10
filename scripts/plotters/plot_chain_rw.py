"""Figure: chain-depth study (Section "Price of Parallelism"), from the full25 solver grids.

P1: W2 vs depth at matched NFE (RK4, 32 steps) for static (no coupling), dynamic with the
    per-block C2OT coupling and dynamic with independent pairing, plus the prior baseline and
    the population limit of dynamic under the batch-128 coupling (chain_population_check.py).
P2/P3: W2 vs RK4 steps per depth for static and for dynamic (per-block C2OT).

Each variant is one results tree written by `make chain-rw` (or chain_rerun.sh):
  CHAIN_RW_BASE/<variant>/stiffness_results_full25-<n>.json
Run: python scripts/plotters/plot_chain_rw.py
"""
from pgm_paper_utils.paths import RESULTS, IMAGES
import collections
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get('CHAIN_RW_BASE', f'{RESULTS}/chain_rw_kr')
OUT = os.environ.get('CHAIN_FIG_OUT',
                     f'{IMAGES}/ablations/chain_stiffness_rw.png')
NFE_K = 32
VARIANTS = [  # (results dir, label, colour, line style, marker)
    ('static_none', 'static', 'tab:blue', '-', 'o'),
    ('dynamic_c2ot', 'dynamic, per-block coupling', 'tab:red', '-', 's'),
    ('dynamic_none', 'dynamic, independent pairing', 'tab:red', ':', 'v'),
]


def load(variant):
    rows = []
    for f in sorted(glob.glob(f'{BASE}/{variant}/stiffness_results_full25-*.json')):
        rows.extend(json.load(open(f)))
    d = collections.defaultdict(list)
    for r in rows:
        d[(r['n'], r['integrator'], r['n_steps'])].append(r['w2_root'])
    prior = collections.defaultdict(list)
    for r in rows:
        prior[r['n']].append(r['w2_root_prior_baseline'])
    return d, prior


data = {v[0]: load(v[0]) for v in VARIANTS}
NS = sorted({k[0] for k in data['static_none'][0]})
ks = sorted({k[2] for k in data['static_none'][0] if k[1] == 'rk4'})

fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
axes[2].sharey(axes[1])

# --- P1: matched-NFE W2 vs depth -----------------------------------------
ax = axes[0]
for v, lab, col, ls, mk in VARIANTS:
    d = data[v][0]
    m = [np.mean(d[(n, 'rk4', NFE_K)]) for n in NS]
    s = [np.std(d[(n, 'rk4', NFE_K)]) for n in NS]
    ax.errorbar(NS, m, yerr=s, fmt=mk + ls, color=col, label=lab, capsize=3, ms=4.5)
pop_f = f'{BASE}/population.json'
if os.path.exists(pop_f):
    pop = json.load(open(pop_f))
    ax.plot(NS, [pop[str(n)]['c2ot']['w2_root_given_leaf'] for n in NS], '--', color='0.45',
            lw=1.2, label='dynamic, per-block coupling:\npopulation limit (linear)')
ax.plot(NS, [np.mean(data['static_none'][1][n]) for n in NS], 'k--', lw=1.2,
        label='prior (ignore evidence)')
ax.set(xlabel='chain length $n$', ylabel='$W_2$ to exact posterior',
       title=f'Matched budget (RK4, {NFE_K} steps)', xscale='log', yscale='log')
ax.set_xticks(NS); ax.set_xticklabels(NS); ax.minorticks_off()
ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# --- P2/P3: step sweeps ---------------------------------------------------
for ax, (v, title) in zip(axes[1:], [('static_none', 'static'),
                                     ('dynamic_c2ot', 'dynamic (per-block coupling)')]):
    d = data[v][0]
    for n in NS:
        m = [np.mean(d[(n, 'rk4', k)]) for k in ks]
        s = [np.std(d[(n, 'rk4', k)]) for k in ks]
        line, = ax.plot(ks, m, 'o-', ms=3.5, label=f'$n$={n}')
        ax.fill_between(ks, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                        color=line.get_color(), alpha=0.18, linewidth=0)
    ax.set(xlabel='RK4 steps', ylabel='$W_2$', xscale='log', yscale='log',
           title=f'{title}: error vs solver effort')
    ax.set_xticks(ks); ax.set_xticklabels(ks); ax.minorticks_off()
    ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print('saved', OUT)
