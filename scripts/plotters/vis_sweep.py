import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import fire
from collections import defaultdict

def plot_learning_dynamics(results_path: str = "evaluation_results.csv", output_dir: str = "plots", metric: str = "SWD", max_value: float = 1000.0, pgm_seed: int = None, task: str = None):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    # Read CSV file
    df = pd.read_csv(results_path)
    
    # Filter by the desired metric
    df = df[df['metric name'] == metric]
    # Optional filtering by pgm_seed and task
    if pgm_seed is not None and 'pgm_seed' in df.columns:
        df = df[df['pgm_seed'] == pgm_seed]
    if task is not None and 'task' in df.columns:
        df = df[df['task'] == task]
    
    if df.empty:
        print(f"No data found for metric '{metric}'")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # Data structure: query -> mode -> step -> list of metric values (from seeds)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Get time step columns (t=0, t=1000, etc.)
    time_cols = [col for col in df.columns if col.startswith('t=')]
    
    # Extract step numbers from column names
    steps = []
    for col in time_cols:
        step_str = col.split('=')[1]
        steps.append(int(step_str))
    
    # Refactored data accumulation to handle averaging over pgm_seed/task
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Filter out invalid values early
    # (Existing loop modified to be more robust)
    
    # We want to group by query, mode, step, learner_seed and average over others
    # Then append that average to data[query][mode][step]
    
    groupby_cols = ['query', 'mode', 'learner_seed']
    # Melt the dataframe to have steps as a column
    df_melted = df.melt(id_vars=groupby_cols, value_vars=time_cols, var_name='time_col', value_name='val')
    df_melted['step'] = df_melted['time_col'].apply(lambda x: int(x.split('=')[1]))
    
    # Filter
    df_melted = df_melted[pd.notna(df_melted['val']) & np.isfinite(df_melted['val']) & (df_melted['val'] <= max_value)]
    
    # Average over pgm_seed/task (which are now implicitly averaged since they aren't in groupby_cols)
    df_avg = df_melted.groupby(['query', 'mode', 'step', 'learner_seed'])['val'].mean().reset_index()
    
    for _, row in df_avg.iterrows():
        data[row['query']][row['mode']][row['step']].append(row['val'])
    
    target_queries = sorted(data.keys())
    if not target_queries:
        print(f"No queries with metric '{metric}' found.")
        return

    n_queries = len(target_queries)
    n_cols = min(3, n_queries)
    n_rows = (n_queries + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    
    all_modes = sorted(list(set(m for q in data for m in data[q])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_modes)))
    mode_colors = {mode: colors[i] for i, mode in enumerate(all_modes)}

    for i, query in enumerate(target_queries):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        for mode in all_modes:
            if mode not in data[query]:
                continue
                
            steps_data = data[query][mode]
            
            # Get all available steps for this query/mode combo
            available_steps = sorted(steps_data.keys())
            
            if not available_steps:
                continue
            
            means = []
            stds = []
            for s in available_steps:
                vals = steps_data[s]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                
            means = np.array(means)
            stds = np.array(stds)
            
            ax.plot(available_steps, means, label=mode, color=mode_colors[mode], marker='o', markersize=4)
            ax.fill_between(available_steps, means - stds, means + stds, color=mode_colors[mode], alpha=0.15)
            
        ax.set_title(query, fontsize=12)
        ax.set_xlabel("Training Step")
        if col == 0:
            ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        # Put legend in the last subplot or first one
        if i == n_queries - 1:
            ax.legend()
            
    # Hide empty subplots
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    plt.tight_layout()
    pgm_str = f"pgms_{pgm_seed}" if pgm_seed is not None else "pgms_all"
    task_str = f"task_{task}" if task is not None else "task_all"
    output_filename = f"sweep_dynamics_{metric.lower()}_{pgm_str}_{task_str}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved learning dynamics plot to {output_path}")

if __name__ == "__main__":
    fire.Fire(plot_learning_dynamics)
