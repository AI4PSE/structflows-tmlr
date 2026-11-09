from pgm_paper_utils.paths import RESULTS
import pandas as pd
import numpy as np
import os
import fire


def parse_query_str(query_str: str):
    if not query_str.startswith('p(') or not query_str.endswith(')'):
        return frozenset(), frozenset()
    content = query_str[2:-1]
    if '|' in content:
        t_part, c_part = content.split('|')
        targets = frozenset(s.strip() for s in t_part.split(',') if s.strip())
        conditions = frozenset(s.strip() for s in c_part.split(',') if s.strip())
        return targets, conditions
    else:
        targets = frozenset(s.strip() for s in content.split(',') if s.strip())
        return targets, frozenset()


def is_conditional_query(query_str: str) -> bool:
    _, conditions = parse_query_str(query_str)
    return len(conditions) > 0


def get_final_col(df):
    time_cols = [col for col in df.columns if col.startswith('t=')]
    if not time_cols:
        return None
    def step_val(col):
        try:
            return int(col.split('=')[1])
        except:
            return -1
    return sorted(time_cols, key=step_val)[-1]


def compute_avg_conditional(df, metric, task, modes, max_value=1000.0):
    """Compute mean and std of the average conditional metric per mode.

    For each mode, the process mirrors plot_final_box with avg_queries=True
    and conditional_only=True:
      1. Filter to the metric and task.
      2. Keep only conditional queries.
      3. Group by (mode, learner_seed) and average over queries/pgm_seeds.
      4. Report mean ± std across learner seeds.
    """
    sub = df[(df['metric name'] == metric) & (df['task'] == task)].copy()
    if sub.empty:
        return {}

    final_col = get_final_col(sub)
    if final_col is None:
        return {}

    # Filter conditional queries
    sub = sub[sub['query'].apply(is_conditional_query)]

    # Remove invalid / extreme values
    sub = sub[pd.notna(sub[final_col]) & np.isfinite(sub[final_col])]
    sub = sub[sub[final_col] <= max_value]

    if sub.empty:
        return {}

    # Average over queries and pgm_seed per (mode, learner_seed)
    grouped = sub.groupby(['mode', 'learner_seed'])[final_col].mean().reset_index()

    results = {}
    for mode in modes:
        vals = grouped[grouped['mode'] == mode][final_col].values
        if len(vals) > 0:
            results[mode] = (np.mean(vals), np.std(vals))
        else:
            results[mode] = (np.nan, np.nan)
    return results


def plot_summary_table(
    results_base: str = f"{RESULTS}",
    experiments: str = "linear4,linear8,nonlinear4,nonlinear8",
    output_path: str = "summary_table.csv",
    output_latex: str = "summary_table.tex",
    max_value: float = 1000.0,
    modes: str = "dynamic,static,carefl,guided", # universal
):
    """Generate a summary table across experiments.

    Collects the average-conditional metrics for:
      - Causal KL
      - Causal Counterfactual KL
      - Inference SWD
    across all experiments and modes, and writes CSV and LaTeX tables.
    """
    experiments = [e.strip() for e in experiments.split(',')]
    modes = [m.strip() for m in modes.split(',')]

    # (metric_name, task, display_name, which modes to include)
    metric_specs = [
        ("KL", "causal", "Causal KL", [m for m in modes if m not in ("guided",)]),
        ("Counterfactual_KL", "causal", "CF KL", [m for m in modes if m not in ("guided",)]),
        ("SWD", "inference", "Inference SWD", modes),
    ]

    rows = []
    for exp in experiments:
        csv_path = os.path.join(results_base, exp, "evaluation_results.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {exp}")
            continue

        df = pd.read_csv(csv_path)
        for metric, task, display, spec_modes in metric_specs:
            stats = compute_avg_conditional(df, metric, task, spec_modes, max_value)
            for mode in spec_modes:
                mean, std = stats.get(mode, (np.nan, np.nan))
                rows.append({
                    "Experiment": exp,
                    "Metric": display,
                    "Mode": mode,
                    "Mean": mean,
                    "Std": std,
                })

    table = pd.DataFrame(rows)
    if table.empty:
        print("No data collected.")
        return

    # Format mean±std string with compact notation
    def fmt(row):
        if np.isnan(row['Mean']):
            return "-"
        mean, std = row['Mean'], row['Std']
        return f"{mean:.3f} ± {std:.0e}"

    def fmt_latex(row):
        if np.isnan(row['Mean']):
            return "-"
        mean, std = row['Mean'], row['Std']
        if std == 0:
            std_str = "0"
        else:
            std_str = f"{std:.0e}"
        return f"${mean:.3f}" + r"_{\pm " + std_str + r"}$"

    table['Value'] = table.apply(fmt, axis=1)
    table['LatexValue'] = table.apply(fmt_latex, axis=1)

    # Pivot: rows = (Experiment, Metric), columns = Mode
    pivot = table.pivot_table(
        index=['Experiment', 'Metric'],
        columns='Mode',
        values='Value',
        aggfunc='first',
    )
    pivot_latex = table.pivot_table(
        index=['Experiment', 'Metric'],
        columns='Mode',
        values='LatexValue',
        aggfunc='first',
    )

    # Reorder columns to match modes order (only those present)
    col_order = [m for m in modes if m in pivot.columns]
    pivot = pivot[col_order]
    pivot_latex = pivot_latex[col_order]

    # Save CSV
    pivot.to_csv(output_path)
    print(f"Saved CSV table to {output_path}")
    print()
    print(pivot.to_string())

    # Save LaTeX — bold the best (min) mean in each row
    # Extract raw means for comparison
    raw_means = table.pivot_table(
        index=['Experiment', 'Metric'],
        columns='Mode',
        values='Mean',
        aggfunc='first',
    )[col_order]

    def bold_min_row(idx):
        means = raw_means.loc[idx]
        min_val = means.min()
        latex_row = pivot_latex.loc[idx]
        result = {}
        for col in col_order:
            cell = latex_row.get(col, "-")
            if pd.isna(cell):
                result[col] = "-"
            elif pd.notna(means.get(col)) and means[col] == min_val:
                # \textbf doesn't work inside $...$, use \mathbf inside math mode
                result[col] = "$\\mathbf{" + cell[1:-1] + "}$"
            else:
                result[col] = cell
        return pd.Series(result)

    pivot_bold = pd.DataFrame(
        [bold_min_row(idx) for idx in pivot_latex.index],
        index=pivot_latex.index,
    )

    # Build LaTeX manually to avoid jinja2 dependency
    lines = []
    lines.append(r"\begin{tabular}{ll" + "r" * len(col_order) + "}")
    lines.append(r"\toprule")
    lines.append("Experiment & Metric & " + " & ".join(col_order) + r" \\")
    lines.append(r"\midrule")
    prev_exp = None
    for (exp, metric), row in pivot_bold.iterrows():
        exp_label = exp if exp != prev_exp else ""
        cells = [str(row.get(c, "-")) if pd.notna(row.get(c)) else "-" for c in col_order]
        lines.append(f"{exp_label} & {metric} & " + " & ".join(cells) + r" \\")
        prev_exp = exp
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    latex = "\n".join(lines)

    with open(output_latex, 'w') as f:
        f.write(latex)
    print(f"\nSaved LaTeX table to {output_latex}")


if __name__ == "__main__":
    fire.Fire(plot_summary_table)
