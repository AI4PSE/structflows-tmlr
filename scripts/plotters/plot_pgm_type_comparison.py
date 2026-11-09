import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import fire
from collections import defaultdict


def parse_query_str(query_str: str):
    """Parse query string like 'p(x0,x1,x2)' or 'p(x1|x0)' into targets and conditions"""
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
    """Check if query has conditions (is conditional)"""
    targets, conditions = parse_query_str(query_str)
    return len(conditions) > 0


def plot_pgm_type_comparison(
    linear_results_path: str = "evaluation_results.csv",
    nonlinear_results_path: str = "evaluation_results.csv",
    output_dir: str = "plots",
    metric: str = "Counterfactual_SWD",
    max_value: float = 1000.0,
    task: str = 'causal',
    conditional_only: bool = True,
    y_max: float = None,
    log_scale: bool = False,
    hide_outliers: bool = False,
):
    """
    Compare linear and nonlinear PGM types across 4 modes (static, dynamic, guided, carefl).
    
    Creates a box-and-whisker plot with:
    - 2 PGM types (linear, nonlinear) shown side-by-side
    - 4 modes (static, dynamic, guided, carefl) across the plot
    - 8 total boxes
    - Data aggregated across pgm_seeds, learner_seeds, and queries
    
    Args:
        linear_results_path: Path to linear PGM evaluation results CSV
        nonlinear_results_path: Path to nonlinear PGM evaluation results CSV
        output_dir: Output directory for plots
        metric: Metric name to plot (e.g., "Counterfactual_SWD")
        max_value: Maximum value to include in plot
        task: Optional task filter (e.g., "causal" or "inference")
        conditional_only: If True, filter to only conditional queries (queries with conditions)
        y_max: If set, limit y-axis maximum to this value (helps see details when outliers dominate)
        log_scale: If True, use logarithmic scale on y-axis to compress range
        hide_outliers: If True, don't plot outlier points (showfliers=False in boxplot)
    """
    
    # Load both result files
    if not os.path.exists(linear_results_path):
        print(f"Linear results file not found: {linear_results_path}")
        return
    
    if not os.path.exists(nonlinear_results_path):
        print(f"Nonlinear results file not found: {nonlinear_results_path}")
        return
    
    # Read CSV files
    df_linear = pd.read_csv(linear_results_path)
    df_nonlinear = pd.read_csv(nonlinear_results_path)
    
    # Add pgm_type column to distinguish them
    df_linear['pgm_type'] = 'linear'
    df_nonlinear['pgm_type'] = 'nonlinear'
    
    # Combine dataframes
    df = pd.concat([df_linear, df_nonlinear], ignore_index=True)
    
    # Filter by the desired metric
    df = df[df['metric name'] == metric]
    
    if df.empty:
        print(f"No data found for metric '{metric}'")
        return
    
    # Optional filtering by task
    if task is not None and 'task' in df.columns:
        df = df[df['task'] == task]
        if df.empty:
            print(f"No data found for metric '{metric}' and task '{task}'")
            return
    
    # Optional filtering for conditional queries only
    if conditional_only and 'query' in df.columns:
        cond_mask = df['query'].apply(is_conditional_query)
        df = df[cond_mask]
        if df.empty:
            print(f"No conditional queries found for metric '{metric}'")
            return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all time step columns (t=0, t=1000, etc.)
    time_cols = [col for col in df.columns if col.startswith('t=')]
    if not time_cols:
        print("No time step columns (starting with 't=') found.")
        return
    
    # Identify the final time step
    def get_step_val(col):
        try:
            return int(col.split('=')[1])
        except:
            return -1
    
    final_col = sorted(time_cols, key=get_step_val)[-1]
    print(f"Using final time step column: {final_col}")
    
    # For each pgm_type, use the highest available time step if final_col is missing
    # This handles cases where different datasets have different time step columns
    df_list = []
    for pgm_type in df['pgm_type'].unique():
        df_pgm = df[df['pgm_type'] == pgm_type].copy()
        
        # Check if final_col has any data for this pgm_type
        if df_pgm[final_col].isna().all():
            # Find the highest time column available for this pgm_type
            available_cols = [col for col in time_cols if df_pgm[col].notna().any()]
            if available_cols:
                available_cols_sorted = sorted(available_cols, key=get_step_val)
                fallback_col = available_cols_sorted[-1]
                print(f"  {pgm_type}: using {fallback_col} instead of {final_col} (missing for this pgm_type)")
                # Copy the fallback column to final_col for this pgm_type
                df_pgm[final_col] = df_pgm[fallback_col]
        
        df_list.append(df_pgm)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Filter out invalid values before grouping
    df = df[pd.notna(df[final_col]) & np.isfinite(df[final_col])]
    try:
        max_val_float = float(max_value)
        df = df[df[final_col] <= max_val_float]
    except (ValueError, TypeError):
        print(f"Warning: max_value '{max_value}' is not a valid float. Skipping max_value filtering.")
    
    # Ensure required columns exist
    if 'learner_seed' not in df.columns:
        print("Warning: 'learner_seed' column not found. Using dummy learner_seed=0.")
        df['learner_seed'] = 0
    
    if 'mode' not in df.columns:
        print("Error: 'mode' column not found.")
        return
    
    if 'pgm_type' not in df.columns:
        print("Error: 'pgm_type' column not found.")
        return
    
    print(f"Loaded {len(df)} total records")
    print(f"Metric: {metric}")
    
    # Aggregate data: group by (pgm_type, mode), averaging over pgm_seed, learner_seed, and query
    # This means we're averaging all values for each (pgm_type, mode) combination
    df_avg = df.groupby(['pgm_type', 'mode'])[final_col].apply(list).reset_index()
    
    if df_avg.empty:
        print(f"No valid data points found after aggregation.")
        return
    
    # Organize data by (pgm_type, mode) for box plots
    data = {}
    for _, row in df_avg.iterrows():
        pgm_type = row['pgm_type']
        mode = row['mode']
        values = row[final_col]
        data[(pgm_type, mode)] = values
    
    pgm_types = sorted(set(pt for pt, m in data.keys()))
    modes = sorted(set(m for pt, m in data.keys()))
    
    print(f"PGM types: {pgm_types}")
    print(f"Modes: {modes}")
    
    # Create the plot with box-and-whisker plots
    # Layout: modes across x-axis, pgm_types side-by-side within each mode
    fig, ax = plt.subplots(figsize=(max(10, 3 * len(modes)), 4))
    
    # Prepare data for boxplot
    plot_data = []
    labels = []
    colors_list = []
    
    x_positions = []
    x_pos = 1
    colors = {'linear': '#2ca02c', 'nonlinear': '#d62728'}  # Green for linear, red for nonlinear
    
    for mode in modes:
        for pgm_type in pgm_types:
            key = (pgm_type, mode)
            if key in data and data[key]:
                plot_data.append(data[key])
                labels.append(f"{pgm_type}\n{mode}")
                colors_list.append(colors.get(pgm_type, 'C0'))
                x_positions.append(x_pos)
            x_pos += 1
        x_pos += 0.5  # Add spacing between mode groups
    
    # Create box plot
    bp = ax.boxplot(
        plot_data,
        positions=x_positions,
        widths=0.6,
        patch_artist=True,
        labels=labels,
        showfliers=not hide_outliers,
    )
    
    # Color the boxes by pgm_type
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Mode')
    ax.set_ylabel(f'{metric} (Final Step)')
    title = f'PGM Type Comparison: {metric} (Linear vs Nonlinear)'
    if task is not None:
        title += f' - {task.capitalize()} Task'
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Apply y-axis scaling options
    if log_scale:
        ax.set_yscale('log')
    
    if y_max is not None:
        ax.set_ylim(top=y_max)

    ax.set_ylim(bottom=0)
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['linear'], alpha=0.7, label='Linear'),
        Patch(facecolor=colors['nonlinear'], alpha=0.7, label='Nonlinear'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    task_str = f"_task_{task}" if task is not None else ""
    cond_str = "_conditional" if conditional_only else ""
    log_str = "_log" if log_scale else ""
    ymax_str = f"_ymax{int(y_max)}" if y_max is not None else ""
    outliers_str = "_nooutliers" if hide_outliers else ""
    output_filename = f"pgm_type_comparison_{metric.lower()}{task_str}{cond_str}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved PGM type comparison plot to {output_path}")


if __name__ == "__main__":
    fire.Fire(plot_pgm_type_comparison)

