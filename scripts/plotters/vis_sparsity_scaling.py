import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import fire

def plot_sparsity_scaling(results_path: str = "evaluation_results.csv", output_dir: str = "../../images", metric: str = "KL"):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Filter for the desired metric
    df_metric = df[df['metric name'] == metric].copy()
    
    # 2. Select joint distribution results (those without '|' in the query)
    df_joint = df_metric[~df_metric['query'].str.contains('\|', regex=True)].copy()
    
    if df_joint.empty:
        print(f"No joint distribution data found for metric {metric}")
        return

    # 3. Identify the final value.
    # Columns are 't=0', 't=1', ...
    t_cols = [c for c in df.columns if c.startswith('t=')]
    if not t_cols:
        print("No time-step columns (t=0, t=1, ...) found.")
        return
    
    # Take the last non-NaN value across time steps for each row
    df_joint['final_val'] = df_joint[t_cols].ffill(axis=1).iloc[:, -1]
    
    # Drop rows where final_val is NaN
    df_joint = df_joint.dropna(subset=['final_val'])

    # 4. Average over seeds (learner_seed, pgm_seed, etc. are columns in the CSV)
    # Group by mode and sparsity
    grouped = df_joint.groupby(['mode', 'sparsity'])['final_val'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(10, 6))
    colors = {'joint': 'blue', 'static': 'green', 'dynamic': 'red', 'triangular': 'orange'}
    
    for mode in grouped['mode'].unique():
        mode_df = grouped[grouped['mode'] == mode].sort_values('sparsity')
        
        sparsities = mode_df['sparsity']
        means = mode_df['mean']
        stds = mode_df['std'].fillna(0) # handle case with 1 seed
        
        color = colors.get(mode)
        plt.plot(sparsities, means, label=mode, marker='o', color=color)
        plt.fill_between(sparsities, means - stds, means + stds, alpha=0.15, color=color)

    plt.xlabel("Sparsity Level (0.0 = Fully connected, 1.0 = Fully disconnected)")
    plt.ylabel(f"Final {metric}")
    plt.title(f"Sparsity vs. {metric} (Averaged over seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, f"sparsity_vs_{metric.lower()}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved sparsity plot to {output_path}")

if __name__ == "__main__":
    fire.Fire(plot_sparsity_scaling)
