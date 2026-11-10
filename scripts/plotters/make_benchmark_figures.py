"""Regenerate the 4-node (and 8-node) benchmark figures of the paper from the evaluation CSVs.

Each results/<bench>_kr/evaluation_results.csv holds every evaluated run of a benchmark: the
re-run models trained with the corrected couplings and the published grid re-evaluated with the
same code. The paper reports one variant per mode, the coupling under which that mode is
consistent (Appendix "Noise--Data Coupling"):

    static: no coupling    dynamic: per-block C2OT    guided: Hungarian    carefl, universal: as-is

paper_csv() keeps exactly those rows and writes results/<bench>_kr/paper_results.csv; the existing
plotters then draw the figures under the file names the paper includes.

Run: python scripts/plotters/make_benchmark_figures.py [--results DIR] [--images DIR] [--n8]
"""
from pgm_paper_utils.paths import RESULTS, IMAGES
import os
import sys

import fire
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_final_box import plot_final_box                      # noqa: E402
from plot_joint_task_comparison import plot_joint_task_comparison  # noqa: E402
from plot_pgm_type_comparison import plot_pgm_type_comparison  # noqa: E402

PAPER_COUPLING = {'static': 'none', 'dynamic': 'c2ot', 'guided': 'hungarian'}


def coupling_of(df):
    """'none' / 'joint' (the pre-fix default: use_coupling unset or True) / 'c2ot' / 'hungarian'."""
    uc = df['use_coupling'] if 'use_coupling' in df else pd.Series(np.nan, index=df.index)
    uc = uc.astype(object).where(uc.notna(), 'joint')
    return uc.map(lambda v: {'False': 'none', False: 'none', 'True': 'joint', True: 'joint'}.get(v, v))


def paper_csv(raw, out):
    df = pd.read_csv(raw)
    cpl = coupling_of(df)
    keep = df['mode'].isin(['carefl', 'universal'])
    for mode, c in PAPER_COUPLING.items():
        keep |= (df['mode'] == mode) & (cpl == c)
    df = df[keep]
    df.to_csv(out, index=False)
    runs = df.groupby(['mode', 'task'])['name of folder'].nunique()
    print(f'{out}: runs per (mode, task)\n{runs.to_string()}')
    return out


def without_guided(csv):
    out = csv.replace('.csv', '_noguided.csv')
    df = pd.read_csv(csv)
    df[df['mode'] != 'guided'].to_csv(out, index=False)
    return out


def main(results=f'{RESULTS}',
         images=f'{IMAGES}/experimental_figures', n8=False):
    lin = paper_csv(f'{results}/linear4_kr/evaluation_results.csv', f'{results}/linear4_kr/paper_results.csv')
    nl = paper_csv(f'{results}/nonlinear4_kr/evaluation_results.csv', f'{results}/nonlinear4_kr/paper_results.csv')
    # guided answers counterfactuals for only a subset of the causal queries, so the
    # counterfactual panels leave it out (every box then covers the same queries)
    lin_cf, nl_cf = without_guided(lin), without_guided(nl)
    out = f'{images}/nonlinear'
    # main text, Figure "pgm-results" (a-c)
    plot_final_box(nl, out, metric='SWD', task='causal', avg_queries=True, conditional_only=True)
    plot_final_box(nl_cf, out, metric='Counterfactual_SWD', task='causal', avg_queries=True, conditional_only=True)
    plot_final_box(nl, out, metric='SWD', task='inference', avg_queries=True, conditional_only=True)
    # appendix: per-query breakdowns and the joint
    plot_final_box(nl, out, metric='SWD', task='causal')
    plot_final_box(nl, out, metric='SWD', task='inference')
    plot_joint_task_comparison(nl, out)
    # appendix: linear vs nonlinear
    plot_pgm_type_comparison(lin, nl, images, metric='SWD', task='causal')
    plot_pgm_type_comparison(lin_cf, nl_cf, images, metric='Counterfactual_SWD', task='causal')
    # appendix: 8-node inference (--n8 once its evaluation is complete)
    raw8 = f'{results}/nonlinear8_kr/evaluation_results.csv'
    if n8:
        n8 = paper_csv(raw8, f'{results}/nonlinear8_kr/paper_results.csv')
        plot_final_box(n8, f'{images}/nonlinear8', metric='SWD', task='inference',
                       avg_queries=True, conditional_only=True)


if __name__ == '__main__':
    fire.Fire(main)
