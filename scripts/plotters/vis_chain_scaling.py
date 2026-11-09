import json
import os
import matplotlib.pyplot as plt
import numpy as np
import fire
from collections import defaultdict
import re

def plot_chain_scaling(results_path: str = "../../results/scaling/evaluation_results.json", output_dir: str = "../../images", metric: str = "KL"):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Data structure: mode -> (n_vars) -> list of [KL for each i]
    # We want to plot x-axis = i, y-axis = KL(p(x_i|x_i-1))
    
    # We aggregate over seeds
    # mode_key example: "dynamic_chain_n32_b100000_s0.3_pseed0"
    mode_pattern = re.compile(r"([a-z]+)_chain_n([0-9]+).+")
    
    # query_key example: "p(x5 | x4)_step_FINAL"
    query_pattern = re.compile(r"p\(x([0-9]+)\s*\|\s*x([0-9]+)\)_step_FINAL")

    # aggregated_data[mode][n_vars][i] = [vals from seeds]
    aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for seed, modes_data in results.items():
        for mode_key, queries in modes_data.items():
            m_match = mode_pattern.match(mode_key)
            if not m_match: continue
            
            mode = m_match.group(1)
            n_vars = int(m_match.group(2))
            
            for q_key, metrics in queries.items():
                q_match = query_pattern.match(q_key)
                if not q_match: continue
                
                target_idx = int(q_match.group(1))
                cond_idx = int(q_match.group(2))
                
                if target_idx == cond_idx + 1:
                    if metric in metrics and np.isfinite(metrics[metric]):
                        aggregated_data[mode][n_vars][target_idx].append(metrics[metric])

    if not aggregated_data:
        print(f"No valid chain scaling data found for metric {metric}")
        return

    for n_vars in sorted([nv for m in aggregated_data for nv in aggregated_data[m]]):
        plt.figure(figsize=(12, 6))
        
        modes = sorted(aggregated_data.keys())
        colors = {'joint': 'blue', 'static': 'green', 'triangular': 'red', 'dynamic': 'red'}
        
        for mode in modes:
            if n_vars not in aggregated_data[mode]: continue
            
            indices = sorted(aggregated_data[mode][n_vars].keys())
            if not indices: continue
            
            means = []
            stds = []
            for i in indices:
                vals = aggregated_data[mode][n_vars][i]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            
            means = np.array(means)
            stds = np.array(stds)
            
            color = colors.get(mode.lower(), None)
            plt.plot(indices, means, label=mode, marker='.', markersize=4, color=color)
            plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color=color)

        plt.xlabel("Chain Position ($i$ for $p(x_i | x_{i-1})$)")
        plt.ylabel(f"{metric}")
        plt.title(f"Chain Scaling: {metric} vs. Position ($N={n_vars}$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, f"chain_scaling_n{n_vars}_{metric.lower()}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved chain scaling plot for N={n_vars} to {output_path}")

if __name__ == "__main__":
    fire.Fire(plot_chain_scaling)
