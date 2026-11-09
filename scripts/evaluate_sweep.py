import os

# Disable XLA GPU autotuning to avoid precision issues
# This prevents the autotuner from generating incorrect kernels that cause numerical precision loss
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0 --xla_force_host_platform_device_count=1')

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import json
import re
import pandas as pd
import fire
from pathlib import Path
from jax import random
from typing import Dict, List, Set, Optional, Union, Tuple, Callable
import time
import traceback
import random as random_py

from structflow.bayes.metrics import compare_distributions, compute_counterfactual
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.eval import evaluate_distribution
from structflow.pgms.utilities import format_dist_str
from pgm_paper_utils.utils import get_test_queries, parse_dist_str, generate_run_name, make_structure
from structflow.pgms.pgm_compiler import PGMCompiler

# --- Parsing Utilities ---

def is_run_dir(dirname: str) -> bool:
    """Checks if directory starts with 'run_'"""
    return dirname.startswith("run_")

def parse_checkpoint_step(filename: str) -> Optional[int]:
    """Extracts step number from 'checkpoint_123.eqx' or 'model_123.eqx'"""
    if filename.endswith(".eqx"):
        if filename.startswith("checkpoint_") or filename.startswith("model_"):
            return int(filename.split('_')[1].split('.')[0])
    return None

def load_config(folder: str):
    config_path = os.path.join(folder, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def flatten_config(config: Dict, prefix: str = "") -> Dict:
    """Flattens a nested dictionary."""
    flat = {}
    for k, v in config.items():
        if isinstance(v, dict) and k != "structure":
            flat.update(flatten_config(v, prefix + k + "_"))
        else:
            flat[prefix + k] = v
    return flat

def should_skip_query(output_path: str, folder_name: str, mode: str, query_str: str, step: int) -> bool:
    """Check if results for a query at a specific step already exist in the output CSV."""
    if not os.path.exists(output_path):
        return False
    
    try:
        df = _read_results_csv(output_path)
        col_name = f"t={step}"
        
        # Check if we have any results for this folder/mode/query combination at this step
        mask = (df['name of folder'] == folder_name) & \
               (df['mode'] == mode) & \
               (df['query'] == query_str)
        
        if not mask.any():
            return False
        
        # Check if the step column exists and has non-NaN values for this query
        if col_name not in df.columns:
            return False
        
        # If all metric values for this query at this step are computed (not NaN), skip it
        relevant_rows = df[mask]
        if (relevant_rows[col_name].notna()).all():
            return True
        
        return False
    except Exception as e:
        print(f"    Warning: Could not check if query was already computed: {e}")
        return False

def _read_results_csv(path):
    """Read a results CSV tolerating malformed lines (e.g. from historical
    concurrent writers); malformed rows are dropped with a warning."""
    return pd.read_csv(path, engine='python', on_bad_lines='warn')


import fcntl
from contextlib import contextmanager

@contextmanager
def _results_lock(output_path):
    """Advisory inter-process lock so multiple evaluate_sweep workers (or
    machines sharing an NFS home) can safely append to one results CSV.
    NOTE: fcntl locks over NFS depend on the lock daemon; prefer one writer
    per CSV (one machine per results dir) when possible."""
    lock_path = output_path + '.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def update_results(output_path: str, folder_name: str, config: Dict, query_str: str, metrics: Dict, step: int):
    with _results_lock(output_path):
        return _update_results_locked(output_path, folder_name, config, query_str, metrics, step)


def _update_results_locked(output_path: str, folder_name: str, config: Dict, query_str: str, metrics: Dict, step: int):
    flat_conf = flatten_config(config)
    
    if os.path.exists(output_path):
        df = _read_results_csv(output_path)
    else:
        df = pd.DataFrame()

    base_info = {
        'name of folder': folder_name,
        'mode': config['mode'],
        'sparsity': config.get('sparsity', 0.0),
        'pgm_seed': config.get('pgm_seed', 0),
        'query': query_str,
    }
    # Add other config variables, avoiding duplicates and complex structures
    for k, v in flat_conf.items():
        if k not in base_info and k != "structure":
            base_info[k] = v
            
    for m_name, m_val in metrics.items():
        row_info = base_info.copy()
        row_info['metric name'] = m_name
        col_name = f"t={step}"
        
        if df.empty:
            new_row = row_info.copy()
            new_row[col_name] = m_val
            df = pd.DataFrame([new_row])
        else:
            # Match based on folder, mode, query and metric name
            mask = (df['name of folder'] == row_info['name of folder']) & \
                   (df['mode'] == row_info['mode']) & \
                   (df['query'] == row_info['query']) & \
                   (df['metric name'] == row_info['metric name'])
            
            if mask.any():
                if col_name not in df.columns:
                    df[col_name] = np.nan
                df.loc[mask, col_name] = m_val
            else:
                new_row = row_info.copy()
                new_row[col_name] = m_val
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
    tmp_path = output_path + f'.tmp.{os.getpid()}'
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_path)

def evaluate_sweep(
    basedir: str, 
    output_path: str = "evaluation_results.csv", 
    num_samples: int = 5000, 
    n_steps: int = 100,
    only_last=True,
    use_random_order=True,
    ):
    
    # 1. Scan for run directories
    # Only look for directories that match our pattern
    all_experiments = []
    if not os.path.exists(basedir):
        print(f"Directory {basedir} does not exist.")
        return

    for item in os.listdir(basedir):
        full_path = os.path.join(basedir, item)
        if os.path.isdir(full_path) and is_run_dir(item):
            config_path = os.path.join(full_path, "config.json")
            if not os.path.exists(config_path):
                print(f"  Skipping {item}: No config.json found.")
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                info = {
                    'path': full_path,
                    'folder_name': item,
                    'learner_seed': config['learner_seed'],
                    'mode': config['mode'],
                    'n_nodes': config['n_nodes'],
                    'config': config
                }
                all_experiments.append(info)
            except Exception as e:
                print(f"  Error reading config in {item}: {e}")
    
    print(f"Found {len(all_experiments)} experiments.")

    # hack to allow parallelisation
    if use_random_order:
        random_py.shuffle(all_experiments)
        
    # 2. Iterate and Evaluate
    for exp_idx, exp in enumerate(all_experiments):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_idx+1}/{len(all_experiments)}")
        print(f"{'='*80}")
        
        folder = exp['path']
        config = exp['config']
        
        run_name = generate_run_name(config, include_hash=False)
        
        print(f"Evaluating {run_name} in {folder}")
        
        # Compile Structure (for creating distributions and getting queries)
        # We need a structure object to get tractable queries.

        print(f"  Creating structure...")
        from structflow.pgms.structure import BlockTriangularStructure
        
        structure_dict = config['structure']
        loaded_struct = BlockTriangularStructure(
            dependencies=structure_dict['dependencies'],
            block_order=structure_dict['block_order'],
            dims=structure_dict['dims']
        )

        pgm = create_pgm(config['pgm_type'], config, random.PRNGKey(config['pgm_seed']))
        constructed_struct = make_structure(pgm, config)
        assert loaded_struct.dependencies == constructed_struct.dependencies
        assert loaded_struct.block_order == constructed_struct.block_order
        assert loaded_struct.dims == constructed_struct.dims
        struct_obj = constructed_struct

        print(f"  ✓ Structure created.")
        print(struct_obj.dependencies)

        print(f"  Getting test queries...")
        queries = get_test_queries(pgm, struct_obj)
        print(f"  ✓ Found {len(queries)} queries to evaluate: {[q[2] for q in queries]}")

        from pgm_paper_utils.learner_factory import create_learner
        model_skeleton, normaliser_skeleton, _, _ = create_learner(config, structure=struct_obj)

        def is_checkpoint(path):
            if path.endswith(".eqx"):
                if path.startswith("checkpoint_") or path.startswith("model_"):
                    return True
            return False
        
        # Find Checkpoints + Final Model
        print(f"  Finding checkpoints...")
        checkpoints = [f for f in os.listdir(folder) if is_checkpoint(f)]
        steps = [parse_checkpoint_step(f) for f in checkpoints]
        sorted_checkpoints = sorted(zip(steps, checkpoints), key=lambda x: x[0])

        if only_last:
            sorted_checkpoints = sorted_checkpoints[-1:]

        print(f'Found checkpoints: {sorted_checkpoints}')

        for ckpt_idx, (step, ckpt_name) in enumerate(sorted_checkpoints):
            print(f"\n  Checkpoint {ckpt_idx+1}/{len(sorted_checkpoints)}: step={step}")

            try:
                # Load weights (model and normaliser saved as a tuple)
                ckpt_path = os.path.join(folder, ckpt_name)
                print(f"    Loading checkpoint from {ckpt_path}...")
                load_start = time.time()
                model, normaliser = eqx.tree_deserialise_leaves(ckpt_path, (model_skeleton, normaliser_skeleton))
                print(f"    ✓ Checkpoint loaded ({time.time() - load_start:.2f}s)")
                
                # Wrap in Distribution
                print(f"    Creating distribution...")
                dist_start = time.time()
                config['n_steps'] = n_steps
                dist = create_distribution(config, model, normaliser)
                print(f"    ✓ Distribution created ({time.time() - dist_start:.2f}s)")
                
                # Metric Loop
                for query_idx, (targets, conditions, q_str) in enumerate(queries):
                    print(f"    Query {query_idx+1}/{len(queries)}: {q_str}")
                    
                    # Skip if we've already evaluated this query at this step
                    mode = config['mode']
                    if should_skip_query(output_path, exp['folder_name'], mode, q_str, step):
                        print(f"      Skipping (already computed)")
                        continue
                    
                    query_start = time.time()
                    try:
                        print(f"      Evaluating...")
                        
                        eval_seed = 200  # Fixed seed for evaluation
                        task = config.get('task', 'inference')
                        
                        metrics = evaluate_distribution(pgm, dist, targets, conditions, seed=eval_seed, num_samples=num_samples, task=task)
                        
                        print(f"      ✓ Metrics: {list(metrics.keys())}")
                        
                        # Log it
                        print(f"      Writing to CSV...")
                        update_results(output_path, exp['folder_name'], config, q_str, metrics, step)
                        print(f"      ✓ Done ({time.time() - query_start:.2f}s)")
                        
                    except Exception as e:
                        elapsed = time.time() - query_start
                        print(f"      ✗ ERROR: {type(e).__name__}: {str(e)}")
                        print(f"      Elapsed time: {elapsed:.2f}s")
                        traceback.print_exc()
                        print(f"      Skipping to next query...")
                        # Write error indicator to CSV for debugging
                        try:
                            error_metrics = {m: np.nan for m in ['SWD', 'NLL', 'KL', 'Counterfactual_SWD', 'Counterfactual_NLL', 'Counterfactual_KL']}
                            update_results(output_path, exp['folder_name'], config, q_str, error_metrics, step)
                        except:
                            pass
                        continue
                        
            except Exception as e:
                print(f"    ✗ ERROR loading/processing checkpoint: {type(e).__name__}: {str(e)}")
                print(f"    Elapsed time: {elapsed:.2f}s")
                traceback.print_exc()
                print(f"    Skipping to next checkpoint...")
                continue

def main():
    """Main entry point with fire"""
    import sys
    try:
        fire.Fire(evaluate_sweep)
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("INTERRUPTED by user")
        print("="*80)
        sys.exit(1)
    except Exception as e:
        print("\n\n" + "="*80)
        print(f"FATAL ERROR: {type(e).__name__}: {str(e)}")
        print("="*80)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()