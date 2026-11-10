import optax
from jax import random
from typing import Dict, Any, Optional
from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.utils import DictNormaliser
from structflow.conditional_flows.flow_factory import (
    create_joint_flow,
    create_static_flow,
    create_dynamic_flow,
    create_carefl_flow,
    create_universal_flow,
    create_dcm_flow
)



# Couplings that keep each mode consistent (Appendix "Noise--Data Coupling"); used when a
# config does not set use_coupling. Modes not listed take no coupling argument.
DEFAULT_COUPLING = {'static': False, 'dynamic': 'c2ot', 'guided': 'hungarian'}

def create_learner(config: Dict[str, Any], structure: Optional[Any] = None):
    """
    Thin wrapper for creating standardised learners from config.

    Parses config to extract parameters and calls the appropriate factory function.
    Each learner has a standard interface for training:
        step_fn: (model, opt_state, normaliser, x_dict_batch, key) ->
                 (updated_model, updated_opt_state, updated_normaliser, loss)

    Args:
        config: Configuration dictionary with keys:
            - mode: 'joint', 'static', 'dynamic', 'carefl', 'universal', or 'dcm'
            - lr: learning rate
            - train_steps: total training steps
            - hidden_dim: network width
            - depth: network depth (default: 3)
            - use_coupling: True/'joint' (Sinkhorn OT), 'hungarian', 'c2ot' or False;
              default per mode (DEFAULT_COUPLING: static False, dynamic 'c2ot', guided 'hungarian')
              (see conditional_flows.flow_factory.COUPLING_DOC)
            - c2ot_target_ratio: adaptive-weight target for use_coupling='c2ot' (default 0.05)
            - use_sparse: for dynamic mode, whether to use sparse interactions (default: False)
            - learner_seed: random seed (default: 0)
            - structure: optional structure dict with 'dependencies', 'block_order', 'dims'
        structure: Optional BlockTriangularStructure (overrides config['structure'] if provided)

    Returns:
        Tuple of (model, normaliser, step_fn, optimizer)
    """
    mode = config['mode']
    key = random.PRNGKey(config.get('learner_seed', 0))
    
    # Create optimizer from config
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.get('init_lr', 1e-4),
        peak_value=config['lr'],
        warmup_steps=config['train_steps'] // 10,
        decay_steps=config['train_steps'],
        end_value=config['lr'] / 100
    )
    optimizer = optax.adam(lr_schedule)

    # Load structure from config if not provided
    if structure is None and 'structure' in config and config['structure'] is not None:
        s_dict = config['structure']
        structure = BlockTriangularStructure(
            dependencies=s_dict['dependencies'],
            block_order=s_dict['block_order'],
            dims=s_dict['dims']
        )

    # Parse config parameters
    use_coupling = config.get('use_coupling', DEFAULT_COUPLING.get(mode, False))
    if use_coupling == 'c2ot' and config.get('c2ot_target_ratio') is not None:
        from sinterp.couplings import C2OTCoupling
        use_coupling = C2OTCoupling(target_ratio=config['c2ot_target_ratio'])
    model_key, _ = random.split(key)
    depth = config.get('depth', 3)
    width = config['hidden_dim']

    # Initialize normaliser
    if structure is not None:
        normaliser = DictNormaliser({name: (dim,) for name, dim in structure.dims.items()})
    else:
        normaliser = None

    # Dispatch to appropriate factory function
    if mode == 'guided':
        model, step_fn = create_joint_flow(
            structure=structure,
            depth=depth,
            width=width,
            model_key=model_key,
            optimizer=optimizer,
            use_coupling=use_coupling,
            normaliser=normaliser,
        )

    elif mode == 'static':
        if structure is None:
            raise ValueError("mode='static' requires a 'structure'")
        model, step_fn = create_static_flow(
            structure=structure,
            depth=depth,
            width=width,
            model_key=model_key,
            optimizer=optimizer,
            use_coupling=use_coupling,
            normaliser=normaliser,
            block_shapes=config.get('block_shapes'),
            conv_channels=config.get('conv_channels', 64),
        )

    elif mode == 'dynamic':
        if structure is None:
            raise ValueError("mode='dynamic' requires a 'structure'")
        model, step_fn = create_dynamic_flow(
            structure=structure,
            depth=depth,
            width=width,
            model_key=model_key,
            optimizer=optimizer,
            use_coupling=use_coupling,
            use_sparse=config.get('use_sparse', False),
            normaliser=normaliser,
        )

    elif mode == 'carefl':
        if structure is None:
            raise ValueError("mode='carefl' requires a 'structure'")
        model, step_fn = create_carefl_flow(
            structure=structure,
            depth=depth,
            width=width,
            model_key=model_key,
            optimizer=optimizer,
            normaliser=normaliser,
        )

    elif mode == 'dcm':
        if structure is None:
            raise ValueError("mode='dcm' requires a 'structure'")
        model, step_fn = create_dcm_flow(
            structure=structure,
            depth=depth,
            width=width,
            model_key=model_key,
            optimizer=optimizer,
            normaliser=normaliser,
        )

    elif mode == 'universal':
        if structure is None:
            raise ValueError("mode='universal' requires a 'structure'")
        model, step_fn = create_universal_flow(
            structure=structure,
            depth=depth,
            width=width,
            model_key=model_key,
            optimizer=optimizer,
            num_bins=config.get('num_bins', 8),
            normaliser=normaliser,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return model, normaliser, step_fn, optimizer
