"""Plot the backbone NFE sweep (static CNF vs per-node DDIM).

Reads results/nonlinear4_kr/backbone_nfe.json (written by
scripts/ablations/backbone_nfe.py) and renders three panels against the
measured NFE (network calls per sample, log2 axis):

  (1) interventional SWD          — predicted flat / agreeing at high NFE,
  (2) counterfactual SWD          — predicted to rise more steeply for DCM
                                    as NFE falls (first-order DDIM inversion),
  (3) abduction reconstruction err — the direct measurement of
                                    Prop. \\ref{prop:abduction-error} (log y).

One line per backbone: mean over pgm_seeds, with a min-max band across seeds.

Run:  python scripts/plotters/plot_backbone_nfe.py
"""
from pgm_paper_utils.paths import RESULTS, IMAGES
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# Sized to the printed width (\textwidth = 6.5in): font sizes are the sizes on the page.
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5})

RESULTS = os.environ.get(
    'NFE_RESULTS', f'{RESULTS}/nonlinear4_kr/backbone_nfe.json')
OUT = f'{IMAGES}/ablations/backbone_nfe.png'

# Okabe-Ito blue / vermillion: CVD-safe pair (validated), fixed assignment.
STYLES = {
    'static': dict(color='#0072B2', marker='o', label='static (CNF, RK4)'),
    'dcm':    dict(color='#D55E00', marker='s', label='dcm (per-node DDIM)'),
}

PANELS = [
    ('swd_interventional', 'Interventional SWD', 'linear'),
    ('swd_counterfactual', 'Counterfactual SWD', 'log'),
    ('abduction_err', r'Abduction error  $\|\hat{u} - u\|_2$', 'log'),
]


def main():
    rows = json.load(open(RESULTS))

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
    for ax, (field, ylabel, yscale) in zip(axes, PANELS):
        for backbone, style in STYLES.items():
            nfes = sorted({r['nfe'] for r in rows if r['backbone'] == backbone})
            mean, lo, hi = [], [], []
            for nfe in nfes:
                vals = np.array([r[field] for r in rows
                                 if r['backbone'] == backbone and r['nfe'] == nfe])
                vals = vals[np.isfinite(vals)]
                mean.append(vals.mean())
                lo.append(vals.min())
                hi.append(vals.max())
            ax.plot(nfes, mean, '-', marker=style['marker'], markersize=3,
                    lw=1.2, color=style['color'], label=style['label'])
            ax.fill_between(nfes, lo, hi, color=style['color'], alpha=0.15, lw=0)
        ax.set_xscale('log', base=2)
        ax.set_yscale(yscale)
        ax.set_xticks(sorted({r['nfe'] for r in rows}))
        ax.set_xticklabels([str(n) for n in sorted({r['nfe'] for r in rows})])
        ax.set_xlabel('NFE (network calls / sample)')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    axes[0].set_title('Interventional (no inversion)')
    axes[1].set_title('Counterfactual (needs abduction)')
    axes[2].set_title('Noise-reconstruction roundtrip')

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
