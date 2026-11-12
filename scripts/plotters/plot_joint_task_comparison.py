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

def is_joint_query(query_str: str, n_vars: int) -> bool:
    """Check if query is a joint distribution over all variables (no conditions, all targets)"""
    targets, conditions = parse_query_str(query_str)
    return len(targets) == n_vars and len(conditions) == 0

def plot_joint_task_comparison(results_path: str = "evaluation_results.csv", output_dir: str = "plots", metric: str = "SWD", max_value: float = 1000.0):
    """
    Compare causal and inference tasks for joint queries across different modes.
    For each mode, the avg metric for causal and inference tasks will appear side by side.
    Data is averaged over PGM seeds and learner seeds.
    Joint queries are identified as: len(targets) == n_vars and no conditions.
    
    Args:
        results_path: Path to evaluation results CSV file
        output_dir: Output directory for plots
        metric: Metric name to plot (e.g., "SWD")
        max_value: Maximum value to include in plot
    """
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    # Read CSV file
    df = pd.read_csv(results_path)
    
    # Filter by the desired metric
    df = df[df['metric name'] == metric]
    
    if df.empty:
        print(f"No data found for metric '{metric}'")
        return
    
    # Filter to only joint queries
    joint_mask = df.apply(lambda row: is_joint_query(row['query'], row['n_nodes']), axis=1)
    df = df[joint_mask]
    
    if df.empty:
        print(f"No joint queries found in the data")
        return
    
    print(f"Found {len(df)} records with joint queries")
    
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
    
    if 'task' not in df.columns:
        print("Error: 'task' column not found. Cannot compare causal vs inference.")
        return
    
    if 'mode' not in df.columns:
        print("Error: 'mode' column not found.")
        return

    # Average over pgm_seed for each (mode, task, learner_seed) combination
    df_avg = df.groupby(['mode', 'task', 'learner_seed'])[final_col].mean().reset_index()
    
    if df_avg.empty:
        print(f"No valid data points found after aggregation.")
        return
    
    # Organize data by (mode, task) for box plots
    data = defaultdict(list)
    for _, row in df_avg.iterrows():
        mode = row['mode']
        task = row['task']
        value = row[final_col]
        data[(mode, task)].append(value)
    
    modes = sorted(set(m for m, t in data.keys()))
    tasks = sorted(set(t for m, t in data.keys()))
    
    print(f"Modes: {modes}")
    print(f"Tasks: {tasks}")
    
    # Check that we have both causal and inference tasks
    if len(tasks) < 2:
        print(f"Warning: Only found {len(tasks)} task type(s). Expected at least 2 (causal and inference).")
    
    # Create the plot with box-and-whisker plots
    n_modes = len(modes)
    n_tasks = len(tasks)
    
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7, 'legend.fontsize': 7.5})
    fig, ax = plt.subplots(figsize=(5.2, 2.4))  # printed at 0.8\textwidth
    
    # Prepare data for boxplot
    plot_data = []
    labels = []
    colors_list = []
    
    x_positions = []
    x_pos = 1
    colors = {'causal': '#1f77b4', 'inference': '#ff7f0e'}
    
    for mode in modes:
        for task in tasks:
            key = (mode, task)
            if key in data and data[key]:
                plot_data.append(data[key])
                labels.append('')
                colors_list.append(colors.get(task, f'C0'))
                x_positions.append(x_pos)
            x_pos += 1
        x_pos += 0.5  # Add spacing between mode groups
    
    # Create box plot
    bp = ax.boxplot(plot_data, positions=x_positions, widths=0.6, patch_artist=True,
                    labels=labels, showfliers=False)
    keys = [(m, t) for m in modes for t in tasks if (m, t) in data and data[(m, t)]]
    ax.set_xticks([np.mean([x for x, k in zip(x_positions, keys) if k[0] == m]) for m in modes])
    ax.set_xticklabels(modes)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=colors[t], alpha=0.7, label=t) for t in tasks if t in colors],
              loc='upper right')
    
    # Color the boxes by task
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('joint SWD')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_filename = f"task_comparison_{metric.lower()}_joint.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Saved task comparison plot to {output_path}")

if __name__ == "__main__":
    fire.Fire(plot_joint_task_comparison)
