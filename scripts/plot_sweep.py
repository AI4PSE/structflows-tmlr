import os
# JAX_PLATFORM_NAME=cpu
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import json
import fire
import subprocess
from pathlib import Path
from tqdm import tqdm
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_losses(run_dir):
    """Plots training losses from losses.npy in the run directory."""
    losses_path = os.path.join(run_dir, "losses.npy")
    if not os.path.exists(losses_path):
        return
    try:
        losses = jnp.load(losses_path)
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.yscale('log')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss: {os.path.basename(run_dir)}")
        plt.grid(True, alpha=0.3)
        
        out_path = os.path.join(run_dir, "losses.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved loss plot to {out_path}")
    except Exception as e:
        print(f"  Error plotting losses for {run_dir}: {e}")

def main(results_dir):
    """
    Iterates through each run directory in results_dir, plotting losses and 
    generating visualizations using existing scripts.
    """
    results_path = Path(results_dir).resolve()
    if not results_path.exists():
        print(f"Directory {results_dir} does not exist.")
        return
    
    # Get all subdirectories that contain a config.json
    run_dirs = []
    for d in results_path.iterdir():
        if d.is_dir() and (d / "config.json").exists():
            run_dirs.append(d)
    
    run_dirs.sort()
    print(f"Found {len(run_dirs)} run directories in {results_dir}")
    
    # Path to experiment scripts
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        config_path = run_dir / "config.json"
        
        with open(config_path, 'r') as f:
            try:
                config = json.load(f)
            except Exception as e:
                print(f"  Error loading config for {run_dir.name}: {e}")
                continue
            
        print(f"\nProcessing {run_dir.name}...")
        
        # 1. Plot training losses
        plot_losses(str(run_dir))
        
        # 2. Visual of the distribution
        pgm_type = config.get('pgm_type')
        
        if pgm_type == 'mixture' or pgm_type == 'nonlinear':
            # Use vis_pgm_density.py for mixture models
            print(f"  Generating density plots for mixture model...")
            my_env = os.environ.copy()
            subprocess.run([
                "python", "experiments/vis_pgm_density.py",
                "--run_dir", str(run_dir.resolve()), 
            ], env=my_env)
            
        
        else:
            print(f"  No specialized visualization script for pgm_type: {pgm_type}")

if __name__ == "__main__":
    fire.Fire(main)
