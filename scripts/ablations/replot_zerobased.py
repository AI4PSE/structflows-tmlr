"""Plot the main SWD box plots with zero-based y-axes.

Uses the same aggregation as the paper figures (average over conditional queries per
learner seed) from an evaluation CSV, with the y-axis anchored at zero so differences
are shown on an absolute scale.

Reads:  $STRUCTFLOW_RESULTS/<folder>/evaluation_results.csv
Writes: <IMAGES>/experimental_figures/nonlinear/<name>_zerobased.png
"""

from pgm_paper_utils.paths import IMAGES
import os
import argparse
import numpy as np
import pandas as pd

METHODS = ["carefl", "dynamic", "guided", "static"]
PRETTY = {"carefl": "CAREFL", "dynamic": "Dynamic", "guided": "Guided", "static": "Static"}
STEP_COL = "t=100000"


def is_conditional(q):
    return isinstance(q, str) and "|" in q


def aggregate(csv, metric, task):
    df = pd.read_csv(csv)
    df = df[df["metric name"] == metric]
    df = df[df["task"] == task]
    df = df[df["query"].apply(is_conditional)]
    df = df[df["mode"].isin(METHODS)]
    df["val"] = pd.to_numeric(df[STEP_COL], errors="coerce")
    df = df.dropna(subset=["val"])
    # one value per (mode, learner_seed): average over conditional queries & pgm seeds
    agg = df.groupby(["mode", "learner_seed"])["val"].mean().reset_index()
    return {m: agg[agg["mode"] == m]["val"].values for m in METHODS}


def plot(data, metric, task, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    series = [data[m] for m in METHODS]
    ax.boxplot(series, tick_labels=[PRETTY[m] for m in METHODS], showfliers=False)
    ax.set_ylim(bottom=0)                      # honest, zero-based axis
    ax.set_ylabel(f"{metric} (lower is better)")
    ax.set_title(f"{task.capitalize()} queries")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    meds = {PRETTY[m]: round(float(np.median(v)), 4) for m, v in data.items()}
    print(f"  medians: {meds}")
    print(f"  saved {os.path.abspath(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", default=f"{RESULTS}")
    ap.add_argument("--folder", default="nonlinear4")
    ap.add_argument("--metric", default="SWD")
    ap.add_argument("--outdir", default=os.path.join(IMAGES, "experimental_figures", "nonlinear"))
    args = ap.parse_args()

    csv = os.path.join(args.results_base, args.folder, "evaluation_results.csv")
    for task in ["inference", "causal"]:
        print(f"[{task}]")
        data = aggregate(csv, args.metric, task)
        out = os.path.join(
            args.outdir,
            f"final_box_{args.metric.lower()}_pgms_all_task_{task}_queries_avg_conditional_zerobased.png")
        plot(data, args.metric, task, out)


if __name__ == "__main__":
    main()
