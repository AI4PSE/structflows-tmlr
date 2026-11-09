"""
Thin wrapper for distribution factory that handles config parsing.

This module takes care of extracting parameters from config dicts and
delegating to the core distribution factory in conditional_flows.dist_factory.
"""

from typing import Dict, Any, Optional
from structflow.conditional_flows.dist_factory import (
    create_distribution as core_create_distribution,
    StructuredFieldAdapter,
    BlockFieldAdapter,
    BlockModelAdapter,
    ParallelFieldAdapter,
)
from sinterp.stochasticfield import OneSidedField_eta1

# Re-export adapter classes for backward compatibility
__all__ = [
    'create_distribution',
    'StructuredFieldAdapter',
    'BlockFieldAdapter',
    'BlockModelAdapter',
    'ParallelFieldAdapter',
    'OneSidedField_eta1',
]


def create_distribution(config: Dict[str, Any], model: Any, normaliser: Optional[Any] = None) -> Any:
    """
    Create a ConditionalDistribution from a config dict and trained model.
    
    This is a thin wrapper that:
    1. Extracts parameters from the config dict
    2. Delegates to the core factory function
    3. Returns the final distribution (possibly wrapped with normaliser)
    
    Args:
        config: Configuration dictionary with keys:
            - 'mode': Distribution type ('static', 'joint', 'dynamic', 'carefl', 'dcm')
            - 'structure': Dict with 'dependencies', 'block_order', 'dims'
            - 'n_steps': Number of integration steps (optional, default: 50)
            - 'integrator_name': Integrator name (optional, default: 'diffrax')
            - 'guidance_scale': Guidance scale for 'joint' mode (optional, default: 7.0)
        model: The trained model.
        normaliser: Optional normaliser to wrap the distribution.
        
    Returns:
        A ConditionalDistribution instance.
    """
    mode = config['mode']
    structure_dict = config['structure']
    n_steps = config.get('n_steps', 50)
    integrator_name = config.get('integrator_name', 'diffrax')
    guidance_scale = config.get('guidance_scale', 7.0)
    
    return core_create_distribution(
        mode=mode,
        structure_dict=structure_dict,
        model=model,
        normaliser=normaliser,
        n_steps=n_steps,
        integrator_name=integrator_name,
        guidance_scale=guidance_scale
    )
