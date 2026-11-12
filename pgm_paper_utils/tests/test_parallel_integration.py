
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.dist_factory import create_distribution
from structflow.conditional_flows.dynamic.dist import DynamicDistribution
from structflow.conditional_flows.dist import NormalisedDistribution

def test_parallel_mode_integration():
    # 1. Setup Config
    structure_dict = {
        'block_order': ['x', 'y'],
        'dependencies': {'x': [], 'y': ['x']},
        'dims': {'x': 1, 'y': 1}
    }
    
    config = {
        'mode': 'dynamic',  # 'parallel' was renamed 'dynamic'
        'structure': structure_dict,
        'hidden_dim': 16,
        'depth': 2,
        'lr': 1e-3,
        'train_steps': 100,
        'use_coupling': False,
        'learner_seed': 42
    }
    
    # 2. Create Learner
    model, normaliser, step_fn, optimizer = create_learner(config)
    
    # 3. Verify Step Function (which uses flattening logic)
    key = jax.random.PRNGKey(0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Dummy data batch (size 10)
    x_batch = {
        'x': jnp.zeros((10, 1)),
        'y': jnp.zeros((10, 1))
    }
    
    # Run one step
    model, opt_state, normaliser, loss = step_fn(model, opt_state, normaliser, x_batch, key)
    
    assert loss.shape == ()
    print(f"Loss: {loss}")
    
    # 4. Create Distribution
    dist = create_distribution(config, model, normaliser)
    
    # 'dynamic' (formerly 'parallel') now builds a DynamicDistribution
    # (dist_factory maps 'dynamic' -> create_dynamic_distribution).
    assert isinstance(dist, NormalisedDistribution)
    assert isinstance(dist.dist, DynamicDistribution)
    
    # 5. Verify Sampling
    sample_key = jax.random.PRNGKey(1)
    samples = dist.sample(sample_key)
    
    assert 'x' in samples
    assert 'y' in samples
    assert samples['x'].shape == (1,)
    assert samples['y'].shape == (1,)
    
    print("Parallel mode integration test passed!")

if __name__ == "__main__":
    test_parallel_mode_integration()
