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


def plot_final_box(results_path: str = "evaluation_results.csv", output_dir: str = "plots", metric: str = "SWD", max_value: float = 1000.0, pgm_seed: int = None, task: str = None, avg_queries: bool = False, conditional_only: bool = False, showfliers: bool = True, modes=None):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    # Read CSV file (assume CSV even if extension is different as per user feedback)
    df = pd.read_csv(results_path)
    
    # Filter by the desired metric
    df = df[df['metric name'] == metric]
    
    # Optional filtering by pgm_seed and task
    if pgm_seed is not None and 'pgm_seed' in df.columns:
        df = df[df['pgm_seed'] == pgm_seed]
    if task is not None and 'task' in df.columns:
        df = df[df['task'] == task]
    
    # Optional filtering for conditional queries only
    if conditional_only and 'query' in df.columns:
        cond_mask = df['query'].apply(is_conditional_query)
        df = df[cond_mask]
        if df.empty:
            print(f"No conditional queries found for metric '{metric}'")
            return
    
    if df.empty:
        print(f"No data found for metric '{metric}', pgm_seed '{pgm_seed}', task '{task}'")
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
    
    # Filter out invalid values before grouping
    df = df[pd.notna(df[final_col]) & np.isfinite(df[final_col])]
    try:
        max_val_float = float(max_value)
        df = df[df[final_col] <= max_val_float]
    except (ValueError, TypeError):
        print(f"Warning: max_value '{max_value}' is not a valid float. Skipping max_value filtering.")

    # Group data and average over pgm_seed and task if they are not filtered
    # We want one value per (query, mode, learner_seed) or (mode, learner_seed) if avg_queries
    
    if 'learner_seed' not in df.columns:
        # Fallback if learner_seed is missing (though it should be there)
        print("Warning: 'learner_seed' column not found. Using dummy learner_seed=0.")
        df['learner_seed'] = 0

    # Ensure avg_queries is bool if fire passed it as string
    if isinstance(avg_queries, str):
        avg_queries = avg_queries.lower() == 'true'

    data = defaultdict(lambda: defaultdict(list))
    if avg_queries:
        # Average over queries, pgm_seed, and task if multiple exist for the same learner_seed
        df_avg = df.groupby(['mode', 'learner_seed'])[final_col].mean().reset_index()
        for _, row in df_avg.iterrows():
            data[f"Avg{' conditional' if conditional_only else ''}"][row['mode']].append(row[final_col])
    else:
        # Average over pgm_seed and task if multiple exist for the same learner_seed
        df_avg = df.groupby(['query', 'mode', 'learner_seed'])[final_col].mean().reset_index()
        for _, row in df_avg.iterrows():
            data[row['query']][row['mode']].append(row[final_col])
            
    target_queries = sorted(data.keys())
    if not target_queries:
        print(f"No valid data points found for metric '{metric}' at final step.")
        return

    n_queries = len(target_queries)
    n_cols = min(3, n_queries)
    n_rows = (n_queries + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    
    if modes is None:
        all_modes = sorted(list(set(m for q in data for m in data[q])))
    else:
        all_modes = sorted(modes)

    for i, query in enumerate(target_queries):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]
        
        plot_data = []
        labels = []
        for mode in all_modes:
            if mode in data[query] and data[query][mode]:
                plot_data.append(data[query][mode])
                labels.append(mode)
        
        if plot_data:
            ax.boxplot(plot_data, tick_labels=labels, showfliers=showfliers)
            ax.set_title(f"Query: {query} Task: {task}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{metric} (Final Step)")
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(query)

    # Hide empty subplots
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    plt.tight_layout()
    
    pgm_str = f"pgms_{pgm_seed}" if pgm_seed is not None else "pgms_all"
    task_str = f"task_{task}" if task is not None else "task_all"
    query_str = "queries_avg" if avg_queries else "queries_all"
    cond_str = "_conditional" if conditional_only else ""
    output_filename = f"final_box_{metric.lower()}_{pgm_str}_{task_str}_{query_str}{cond_str}.png"
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved final metric box plot to {output_path}")

if __name__ == "__main__":
    fire.Fire(plot_final_box)
