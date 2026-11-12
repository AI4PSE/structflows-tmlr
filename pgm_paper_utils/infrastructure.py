import jax
import equinox as eqx
import jax.numpy as jnp
from pgm_paper_utils.learner_factory import create_learner
import copy

def count_total_params(model):
    """Counts actual trainable parameters for a given model."""

    # Custom count to handle masked layers, non-trainable components, and avoid double-counting
    from structflow.conditional_flows.dynamic.layers import DenseMaskedLinear, SparseMaskedLinear
    from flow.net import FourierFeatures

    class SpecialCount:
        def __init__(self, count, obj_id):
            self.count = count
            self.obj_id = obj_id

    def get_params_contribution(node):
        if isinstance(node, DenseMaskedLinear):
            # Masked weights are not trainable, only the active part is what we count
            count = int(jnp.sum(node.mask))
            if node.linear.bias is not None:
                count += node.linear.bias.size
            return SpecialCount(count, id(node))
        
        elif isinstance(node, SparseMaskedLinear):
            # Sparse weights are exactly what we want to count
            count = node.weight_values.size
            if node.bias is not None:
                count += node.bias.size
            return SpecialCount(count, id(node))
            
        elif isinstance(node, FourierFeatures):
            # Only count if trainable
            count = node.w.size if getattr(node, 'trainable', True) else 0
            return SpecialCount(count, id(node))
            
        elif eqx.is_array(node):
            return node
            
        return None

    # Filter the tree to things we care about
    # We stop at special modules OR arrays
    params_tree = jax.tree_util.tree_map(
        get_params_contribution,
        model,
        is_leaf=lambda x: isinstance(x, (DenseMaskedLinear, SparseMaskedLinear, FourierFeatures, SpecialCount)) or eqx.is_array(x)
    )
    
    # We must also use is_leaf here to ensure SpecialCount objects are treated as leaves
    leaves = jax.tree_util.tree_leaves(params_tree, is_leaf=lambda x: isinstance(x, SpecialCount))
    unique_arrays = {}
    unique_special = {} # id -> count
    
    for leaf in leaves:
        if isinstance(leaf, SpecialCount):
            unique_special[leaf.obj_id] = leaf.count
        elif eqx.is_array(leaf):
            unique_arrays[id(leaf)] = leaf.size
            
    return int(sum(unique_arrays.values()) + sum(unique_special.values()))

def get_hidden_dims(config, budget_params, structure, max_search_width=1024):
    """
    Finds the hidden dimension that most closely matches the parameter budget.
    """
    # Simple iterative search for the best width
    # Lower bound: 2, Upper bound: 1024 (reasonable range)
    low = 2
    high = max_search_width
    best_h = low
    min_diff = float('inf')
    
    # Binary search for the best hidden_dim.
    # The param count is monotonic with hidden_dim.

    print(f"Searching for hidden_dim in range [{low}, {high}] with budget {budget_params}")
    
    while low <= high:
        mid = (low + high) // 2
        
        # model = init_model(mode, test_config, structure)
        # dummy model
        dummy_config = copy.deepcopy(config)
        dummy_config['hidden_dim'] = mid
        try:
            model, _, _, _ = create_learner(dummy_config, structure)
            p_count = count_total_params(model)
        except Exception as e:
            # A trial candidate can OOM before it is even counted (e.g. dynamic
            # mode's dense masked layers scale as (width * n_blocks)^2). Treat
            # an unconstructible candidate as over budget and search lower.
            if 'RESOURCE_EXHAUSTED' in str(e) or 'Out of memory' in str(e):
                print(f"  hidden_dim={mid} OOM during trial construction; searching lower")
                high = mid - 1
                continue
            raise
        finally:
            model = None
        
        if p_count <= budget_params:
            # This is a valid candidate
            diff = budget_params - p_count
            if diff < min_diff:
                min_diff = diff
                best_h = mid
            low = mid + 1
        else:
            # Over budget, must reduce hidden_dim
            high = mid - 1
            
    return best_h
