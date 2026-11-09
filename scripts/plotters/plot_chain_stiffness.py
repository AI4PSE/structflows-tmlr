"""Figure: chain-depth study, 4 panels from the full25 solver grid + all-seed OOD.

P1: W2 vs depth at matched NFE (rk4-32; static is solver-converged there,
    identical to adaptive to 3 decimals) + prior-baseline reference.
P2/P3: W2 vs RK4 steps per depth (static converges / dynamic anti-converges).
P4: conditional/joint max-|v| amplification vs depth: dynamic clamp block
    grows, integrated blocks (both modes) stay flat. 8-seed mean +/- std.

Run: ~/my_jax_env/bin/python scripts/plotters/plot_chain_stiffness.py
"""
from pgm_paper_utils.paths import RESULTS, IMAGES
import json, glob, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
BASE = os.environ.get('CHAIN_BASE',
                      f'{RESULTS}/chain_stiffness_strong')
OUT = os.environ.get('CHAIN_FIG_OUT',
                     f'{IMAGES}/ablations/chain_stiffness.png')
NFE_K = 32
COLS = {'static': 'tab:blue', 'dynamic': 'tab:red'}

rows = []
for f in sorted(glob.glob(f'{BASE}/stiffness_results_full25-*.json')):
    rows.extend(json.load(open(f)))
NS = sorted({r['n'] for r in rows})

d = collections.defaultdict(list)
for r in rows:
    d[(r['n'], r['mode'], r['integrator'], r['n_steps'])].append(r['w2_root'])
base = collections.defaultdict(list)
for r in rows:
    base[r['n']].append(r['w2_root_prior_baseline'])

fig, axes = plt.subplots(1, 4, figsize=(19, 3.9))

# --- P1: matched-NFE W2 vs depth -----------------------------------------
ax = axes[0]
for mode in ('static', 'dynamic'):
    m = [np.mean(d[(n, mode, 'rk4', NFE_K)]) for n in NS]
    s = [np.std(d[(n, mode, 'rk4', NFE_K)]) for n in NS]
    ax.errorbar(NS, m, yerr=s, fmt='o-', color=COLS[mode], label=mode, capsize=3)
ax.plot(NS, [np.mean(base[n]) for n in NS], 'k--', lw=1.2,
        label='prior (ignore evidence)')
ax.set(xlabel='chain length $n$', ylabel='$W_2$ to exact posterior',
       title=f'Matched budget (RK4, {NFE_K} steps)', xscale='log')
ax.set_xticks(NS); ax.set_xticklabels(NS); ax.minorticks_off()
ax.legend(fontsize=8)

# --- P2/P3: step sweeps ---------------------------------------------------
ks = sorted({r['n_steps'] for r in rows if r['integrator'] == 'rk4'})
for ax, mode in zip(axes[1:3], ('static', 'dynamic')):
    for n in NS:
        m = [np.mean(d[(n, mode, 'rk4', k)]) for k in ks]
        s = [np.std(d[(n, mode, 'rk4', k)]) for k in ks]
        line, = ax.plot(ks, m, 'o-', ms=3.5, label=f'$n$={n}')
        ax.fill_between(ks, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)],
                        color=line.get_color(), alpha=0.18, linewidth=0)
    ax.set(xlabel='RK4 steps', ylabel='$W_2$', xscale='log', yscale='log',
           title=f'{mode}: error vs solver effort')
    ax.set_xticks(ks); ax.set_xticklabels(ks); ax.minorticks_off()
    ax.legend(fontsize=7, ncol=2)

# --- P4: velocity amplification vs depth ---------------------------------
ood = json.load(open(f'{BASE}/ood_measures_allseeds.json'))
by = collections.defaultdict(dict)
for r in ood:
    by[(r['n'], r['mode'], r['seed'])][r['kind']] = r
series = {'dyn_clamp': collections.defaultdict(list),
          'dyn_int': collections.defaultdict(list),
          'stat_int': collections.defaultdict(list)}
for (n, mode, seed), kk in by.items():
    if 'conditional' not in kk or 'joint' not in kk:
        continue
    c, j = kk['conditional']['max_v'], kk['joint']['max_v']
    blocks = list(c.keys())              # compiled order: condition first
    ratios = {b: c[b] / max(j[b], 1e-9) for b in blocks}
    intmax = max(ratios[b] for b in blocks[1:])
    if mode == 'dynamic':
        series['dyn_clamp'][n].append(ratios[blocks[0]])
        series['dyn_int'][n].append(intmax)
    else:
        series['stat_int'][n].append(intmax)

ax = axes[3]
for key, lab, col, ls in [
        ('dyn_clamp', 'dynamic: clamped cond. block', 'tab:red', '-'),
        ('dyn_int', 'dynamic: integrated blocks (max)', 'tab:red', '--'),
        ('stat_int', 'static: integrated blocks (max)', 'tab:blue', '--')]:
    m = [np.mean(series[key][n]) for n in NS]
    s = [np.std(series[key][n]) for n in NS]
    ax.errorbar(NS, m, yerr=s, fmt='o' + ls, color=col, label=lab,
                capsize=3, ms=3.5)
ax.axhline(1.0, color='0.6', lw=0.8, zorder=0)
ax.set(xlabel='chain length $n$',
       ylabel=r'max $|v|$ ratio, conditional / joint',
       title='Where the velocity grows', xscale='log', yscale='log')
ax.set_xticks(NS); ax.set_xticklabels(NS); ax.minorticks_off()
ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print('saved', OUT)
