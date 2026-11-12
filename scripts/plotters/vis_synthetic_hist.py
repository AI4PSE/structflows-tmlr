from pgm_paper_utils.paths import IMAGES
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import json
import equinox as eqx
import fire
from jax import random
import numpy as np

# Ensure CPU usage
jax.config.update('jax_platform_name', 'cpu')

import sys
sys.path.append(os.path.abspath(".."))

from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.dist_factory import create_distribution

def load_model_from_run(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    from structflow.pgms.structure import BlockTriangularStructure
    s_dict = config['structure']
    structure = BlockTriangularStructure(
        dependencies=s_dict['dependencies'],
        block_order=s_dict['block_order'],
        dims=s_dict['dims']
    )
    
    model_skeleton, normaliser_skeleton, _, _ = create_learner(config, structure=structure)
    ckpt_path = os.path.join(run_dir, "model.eqx")
    if not os.path.exists(ckpt_path):
        return None, None, config
    model, normaliser = eqx.tree_deserialise_leaves(ckpt_path, (model_skeleton, normaliser_skeleton))
    dist = create_distribution(config, model, normaliser)
    return dist, structure, config

def plot_synthetic_hists(results_dir: str, pgm_type: str = "random", n_vars: int = 16, output_path: str = None):
    """
    Finds one run for each mode (joint, static, dynamic) and plots histograms vs PGM.
    """
    modes = ['joint', 'static', 'dynamic']
    best_runs = {mode: None for mode in modes}
    
    # Simple heuristic: pick seed 0 for each mode that matches pgm_type and n_vars
    for item in os.listdir(results_dir):
        if not item.startswith("run_"): continue
        path = os.path.join(results_dir, item)
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path): continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if config.get('pgm_type') == pgm_type and config.get('n_vars') == n_vars:
            mode = config.get('mode')
            if mode in modes and best_runs[mode] is None:
                best_runs[mode] = path

    # 1. Sample from PGM (Ground Truth)
    # Use config from one of the runs to recreate PGM
    first_run = next(r for r in best_runs.values() if r is not None)
    with open(os.path.join(first_run, "config.json"), 'r') as f:
        gt_config = json.load(f)
    
    pgm = create_pgm(pgm_type, gt_config, random.PRNGKey(0))
    n_samples = 2000
    gt_samples = jax.vmap(pgm.sample)(random.split(random.PRNGKey(777), n_samples))
    
    # 2. Sample from each mode
    mode_samples = {}
    for mode, run_path in best_runs.items():
        if run_path is None: continue
        print(f"Loading {mode} from {run_path}")
        dist, structure, _ = load_model_from_run(run_path)
        if dist is None: continue
        
        # Sample joint
        # For dynamic/static, joint sampling is just sampling all blocks in order.
        # NormalisedDistribution.sample handles this.
        keys = random.split(random.PRNGKey(888), n_samples)
        samples = jax.vmap(lambda k: dist.sample(k))(keys)
        mode_samples[mode] = samples

    # 3. Plot Histograms
    var_names = sorted(list(gt_samples.keys()))
    n_vars = len(var_names)
    cols = 4
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
    axes = axes.flatten()
    
    colors = {'joint': 'blue', 'static': 'green', 'dynamic': 'red'}
    
    for i, var in enumerate(var_names):
        ax = axes[i]
        # PGM GT
        ax.hist(np.array(gt_samples[var]).flatten(), bins=30, density=True, alpha=0.3, color='gray', label='PGM (GT)')
        
        for mode, samples in mode_samples.items():
            ax.hist(np.array(samples[var]).flatten(), bins=30, density=True, histtype='step', color=colors[mode], label=mode)
        
        ax.set_title(var)
        if i == 0:
            ax.legend()
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(IMAGES, f"synthetic_{pgm_type}_hists.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved synthetic histograms to {output_path}")

if __name__ == "__main__":
    fire.Fire(plot_synthetic_hists)
