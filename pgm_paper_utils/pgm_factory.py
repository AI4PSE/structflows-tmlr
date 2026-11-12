import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Union, Set, List
import os
from pathlib import Path
import sys

from structflow.pgms.pgm import (
    make_chain_pgm,
    make_random_pgm,
    make_dense_pgm
)
from structflow.pgms.image_benchmarks import DigitsPGM, MNISTPGM
from structflow.pgms.colored_mnist import ColoredMNISTPGM


def create_pgm(pgm_type: str, config: Dict[str, Any], key: jax.Array):
    """
    Factory function to create a PGM instance based on type and configuration.
    
    This function creates PGMs using the refactored pgms codebase.
    Structural PGMs support 4 mechanisms with 3 structures:
    
    Mechanisms: 'linear', 'nonlinear_parent', 'nonlinear_noise', 'universal'
    Structures: 'chain', 'random', 'dense'
    
    Args:
        pgm_type: One of ['chain', 'random', 'dense', 'digits', 'mnist',
                         'eight_schools', 'diamonds']
        config: Dictionary containing PGM-specific parameters.
                Common keys:
                - 'n_nodes': Number of variables (default 16)
                - 'mechanism_type': One of 'linear', 'nonlinear_parent', 
                                   'nonlinear_noise', 'universal' (default 'linear')
                - 'node_dim': Dimension of each node (default 1)
                - 'sparsity': Sparsity for random DAGs (default 0.3)
                - 'hidden_width': Width of hidden layers (for nonlinear mechanisms)
                - 'num_bins': Number of bins for spline mechanisms
        key: JAX PRNGKey for random initializations.
        
    Returns:
        A StructuralPGM instance, or the image PGM class for image benchmarks.
    """
    
    # Determine mechanism type and mechanism kwargs
    mechanism_type = config.get('mechanism_type', 'linear')
    node_dim = config.get('node_dim', 1)
    n_nodes = config.get('n_nodes', 16)
    
    # Prepare mechanism kwargs
    mechanism_kwargs = {}
    mechanism_kwargs['hidden_width'] = config.get('mechanism_hidden_width', 32)
    mechanism_kwargs['num_bins'] = config.get('num_bins', 8)
    # Strong-coupling linear chains (stiffness study): deterministic A/B scales
    if config.get('weight_scale') is not None:
        mechanism_kwargs['weight_scale'] = config['weight_scale']
    if config.get('noise_scale') is not None:
        mechanism_kwargs['noise_scale'] = config['noise_scale']
    if config.get('root_scale') is not None:
        mechanism_kwargs['root_scale'] = config['root_scale']
    
    # Structural PGM creation
    if pgm_type == 'chain':
        return make_chain_pgm(
            n_nodes=n_nodes,
            mechanism_type=mechanism_type,
            key=key,
            node_dim=node_dim,
            **mechanism_kwargs
        )
        
    elif pgm_type == 'random':
        sparsity = config.get('sparsity', 0.3)
        return make_random_pgm(
            n_nodes=n_nodes,
            mechanism_type=mechanism_type,
            key=key,
            sparsity=sparsity,
            node_dim=node_dim,
            **mechanism_kwargs
        )
    
    elif pgm_type == 'dense':
        return make_dense_pgm(
            n_nodes=n_nodes,
            mechanism_type=mechanism_type,
            key=key,
            node_dim=node_dim,
            **mechanism_kwargs
        )
    
    # Image PGMs
    elif pgm_type == 'digits':
        return DigitsPGM()

    elif pgm_type == 'mnist':
        return MNISTPGM()

    elif pgm_type == 'colored_mnist':
        return ColoredMNISTPGM(
            rho=config.get('color_correlation', 0.8),
            color_noise=config.get('color_noise', 0.05),
        )

    else:
        raise ValueError(f"Unknown PGM type: {pgm_type}. "
                        f"Supported: chain, random, dense, digits, mnist, colored_mnist")
