import os
import json
import hashlib
from typing import Dict, Any, List, Tuple, Set, Optional, Union
import equinox as eqx
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.pgm_factory import create_pgm
from structflow.pgms.utilities import format_dist_str
from structflow.pgms.pgm_compiler import PGMCompiler
from jax import random

def make_structure(pgm, config: Dict[str, Any]):
    task = config['task']
    dag = pgm.get_pgm_structure()

    n_nodes = len(dag.keys())
    config['n_nodes'] = n_nodes
    
    # 2. Get Targets and Conditions
    compiler = PGMCompiler(dag, dims=pgm.variable_dims)
    if task == 'causal':
        targets, conditions = ({}, {})
        structure = compiler.compile_causal()
    elif task == 'inference':
        targets, conditions = pgm.get_inference_targets_and_conditions()
        print(f"Targets: {targets}, Conditions: {conditions}")
        structure = compiler.compile_inference(targets=frozenset(targets), conditions=frozenset(conditions))
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return structure

def parse_dist_str(dist_str: str) -> Tuple[frozenset[str], frozenset[str]]:
    """Parses 'p(x1 | x0)' -> (frozenset({'x1'}), frozenset({'x0'}))"""
    if not dist_str.startswith('p(') or not dist_str.endswith(')'):
        return frozenset(), frozenset()
    content = dist_str[2:-1]
    if '|' in content:
        t_part, c_part = content.split('|')
        targets = [s.strip() for s in t_part.split(',') if s.strip()]
        conditions = [s.strip() for s in c_part.split(',') if s.strip()]
        return frozenset(targets), frozenset(conditions)
    else:
        targets = [s.strip() for s in content.split(',') if s.strip()]
        return frozenset(targets), frozenset()

def get_test_queries(pgm, structure) -> List[Tuple[frozenset, frozenset, str]]:
    """
    Returns list of tractable queries from the intersection of PGM supported
    and flow tractable distributions.
    
    Filters out:
    - Unconditional distributions with multiple variables that are not the full joint
      (e.g., p(x0, x1, x2) is filtered out, but p(x0) and p(x0, x1, x2, x3) are kept)
    - Conditional distributions with multiple targets
      (e.g., p(x0, x1 | x2) is filtered out, but p(x0 | x1, x2) is kept)
    
    Returns (targets, conditions, query_string)
    """
    # 1. Get supported by PGM
    supported_pgm = pgm.get_supported_distributions()
    
    # 2. Get tractable by Flow
    tractable_flow = structure.get_tractable_distributions()

    print('PGM supports \n', [format_dist_str(tc['targets'], tc['conditions']) for tc in supported_pgm])
    print('Flow supports \n', [format_dist_str(tc['targets'], tc['conditions']) for tc in tractable_flow])
    
    # 3. Intersection
    # We use (targets, conditions) frozensets as keys
    def to_key(d):
         return (d['targets'], d['conditions'])
    
    pgm_keys = {to_key(d) for d in supported_pgm}
    flow_keys = {to_key(d) for d in tractable_flow}
    
    intersection_keys = pgm_keys.intersection(flow_keys)
    
    # Get total number of variables from structure
    n_vars = len(structure.block_order)
    
    # 4. Filter distributions
    # Keep only:
    # - Unconditional: single marginals (len(targets)==1) OR full joint (len(targets)==n_vars)
    # - Conditional: single target (len(targets)==1)
    filtered_keys = set()
    for t, c in intersection_keys:
        if len(c) == 0:  # Unconditional distribution
            # Keep only single marginals or full joint
            if len(t) == 1 or len(t) == n_vars:
                filtered_keys.add((t, c))
        else:  # Conditional distribution
            # Keep only single-target conditionals
            if len(t) == 1:
                filtered_keys.add((t, c))
    
    # 5. Form query triples
    test_queries = []
    for t, c in filtered_keys:
        q_str = format_dist_str(t, c)
        test_queries.append((t, c, q_str))
        
    return sorted(test_queries, key=lambda x: x[2])

def load_config(path: str) -> Dict[str, Any]:
    """Loads a configuration file."""
    with open(path, "r") as f:
        return json.load(f)

def load_model(config, path: str, structure=None):
    """Loads a model from a file."""
    model, _, _, _ = create_learner(config, structure=structure)
    model = eqx.tree_deserialise_leaves(path, model)
    return model

def create_distribution_from_checkpoint(checkpoint_path: str):
    """
    Load a distribution from a saved checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint directory containing:
            - config.json
            - model_*.eqx (or checkpoint_*.eqx)
            
    Returns:
        ConditionalDistribution object
    """
    config = load_config(os.path.join(checkpoint_path, 'config.json'))
    
    # Search for model or checkpoint file
    files = os.listdir(checkpoint_path)
    ckpt_files = [f for f in files if (f.startswith('model_') or f.startswith('checkpoint_')) and f.endswith('.eqx')]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    # Take the latest one
    ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    ckpt_path = os.path.join(checkpoint_path, ckpt_files[-1])
    print(f"Loading model from {ckpt_path}")
    
    # Prepare skeletons
    from structflow.pgms.structure import BlockTriangularStructure
    structure_dict = config['structure']
    struct_obj = BlockTriangularStructure(
        dependencies=structure_dict['dependencies'],
        block_order=structure_dict['block_order'],
        dims=structure_dict['dims']
    )
    
    model_skeleton, normaliser_skeleton, _, _ = create_learner(config, structure=struct_obj)
    
    # Load weights
    model, normalizer = eqx.tree_deserialise_leaves(ckpt_path, (model_skeleton, normaliser_skeleton))

    return create_distribution(config, model, normalizer)

def generate_run_name(config: Dict[str, Any], include_hash: bool = True, exclude_keys: Optional[List[str]] = None) -> str:
    """
    Generates a unified, descriptive run name by encoding all hyperparameters from config.
    
    The name includes all config fields (except excluded ones) encoded as key=value pairs,
    plus an optional hash for absolute uniqueness.
    
    Format: run_key1=val1_key2=val2_..._hash
    
    Args:
        config: Configuration dictionary to encode
        include_hash: If True, appends an MD5 hash of the config for absolute uniqueness
        exclude_keys: List of keys to exclude from the name (defaults to ['structure'])
    
    Returns:
        Descriptive run name suitable for folder naming
    """
    if exclude_keys is None:
        exclude_keys = ['structure']  # structure can be reconstructed from seeds
    
    # Build list of key=value pairs, sorted by key for determinism
    hparams = []
    for key in sorted(config.keys()):
        if key in exclude_keys:
            continue
        value = config[key]

        # Skip complex types that can't be simply converted to strings
        if isinstance(value, (dict, list)):
            continue

        # Unset knobs (None) stay out of the name: they add no information
        # and push long configs past ext4's 255-char filename limit.
        if value is None:
            continue
        
        # Convert floats to p-notation (e.g., 0.3 -> 0p3)
        if isinstance(value, float):
            value_str = str(value).replace('.', 'p')
        else:
            value_str = str(value)
        
        hparams.append(f"{key}{value_str}")
    
    # Join all hparams; cap well under ext4's 255-char filename limit (the
    # hash below keeps truncated names unique).
    run_name = f"run_{'_'.join(hparams)}"[:240]

    # Append hash for uniqueness if requested
    if include_hash:
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        run_name = f"{run_name}_{config_hash}"

    return run_name
