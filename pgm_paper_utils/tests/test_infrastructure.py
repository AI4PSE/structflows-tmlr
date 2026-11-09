import jax
import jax.random as jr
import equinox as eqx
import pytest
import time
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.infrastructure import count_total_params, get_hidden_dims
from pgm_paper_utils.utils import make_structure

# Updated to the current API (as used by scripts/sweep.py):
#   * PGMs/structures: create_pgm(...) + make_structure(pgm, config)
#     (PGMCompiler.compile was split into compile_inference / compile_causal).
#   * count_total_params(model) counts an instantiated model; the old
#     count_total_params(mode, config, structure, pgm) signature is gone.
#   * get_hidden_dims(config, budget_params, structure) takes the full learner
#     config (incl. 'mode' and 'depth').
#   * The joint/guidance learner mode is now called 'guided' (was 'joint').


def _learner_config(mode, structure, **overrides):
    config = {
        'mode': mode,
        'hidden_dim': 8,
        'depth': 3,
        'lr': 1e-3,
        'train_steps': 100,
        'structure': {
            'dependencies': structure.dependencies,
            'block_order': structure.block_order,
            'dims': structure.dims,
        },
    }
    config.update(overrides)
    return config


def _count(config, structure):
    model, _, _, _ = create_learner(config, structure)
    return count_total_params(model)


@pytest.fixture
def setup_pgm():
    pgm_seed = 0
    config = {
        'pgm_type': 'random',
        'n_nodes': 4,
        'sparsity': 0.3,
        'node_dim': 1,
        'task': 'inference',  # default inference task structure
    }
    pgm = create_pgm('random', config, jr.PRNGKey(pgm_seed))
    structure = make_structure(pgm, config)
    return pgm, structure


@pytest.mark.parametrize("mode,hidden_dim", [("guided", 32), ("static", 8), ("dynamic", 8)])
def test_count_total_params(setup_pgm, mode, hidden_dim):
    pgm, structure = setup_pgm
    config = _learner_config(mode, structure, hidden_dim=hidden_dim)

    p_count = _count(config, structure)
    assert p_count > 0
    assert isinstance(p_count, int)


def test_get_hidden_dims_monotonic(setup_pgm):
    pgm, structure = setup_pgm
    budget = 10000
    config = _learner_config('guided', structure, depth=3)

    h = get_hidden_dims(config, budget, structure)
    p_count = _count({**config, 'hidden_dim': h}, structure)

    assert p_count <= budget

    # Check if h+1 is over budget (or at least p_count increases)
    p_count_plus = _count({**config, 'hidden_dim': h + 1}, structure)
    assert p_count_plus >= p_count


@pytest.mark.parametrize("mode", ["guided", "static", "dynamic"])
@pytest.mark.parametrize("budget", [5000, 100000, 500000])
def test_get_hidden_dims_thorough(setup_pgm, mode, budget):
    pgm, structure = setup_pgm
    depth = 8
    config = _learner_config(mode, structure, depth=depth)

    start_time = time.time()
    h = get_hidden_dims(config, budget, structure)
    duration = time.time() - start_time

    p_count = _count({**config, 'hidden_dim': h}, structure)

    assert h >= 2
    assert p_count <= budget
    print(f"\nMode {mode}, Budget {budget}: hidden_dim={h}, params={p_count}, took {duration:.2f}s")

    # Safety check for performance
    assert duration < 20, f"get_hidden_dims took too long: {duration:.2f}s"


def test_count_total_params_sparse_vs_dense(setup_pgm):
    pgm, structure = setup_pgm
    base_config = _learner_config('dynamic', structure, hidden_dim=32)

    # Dense
    p_count_dense = _count({**base_config, 'use_sparse': False}, structure)

    # Sparse
    p_count_sparse = _count({**base_config, 'use_sparse': True}, structure)

    # They should be exactly the same because our counting logic for Dense
    # only counts the non-zero entries of the mask, which matches Sparse.
    assert p_count_dense == p_count_sparse
    assert p_count_dense > 0


def test_count_total_params_static_deduplication(setup_pgm):
    pgm, structure = setup_pgm
    # Use a config that will likely have shared block signatures
    config = _learner_config('static', structure, hidden_dim=8)

    model, _, _, _ = create_learner(config)
    p_count = count_total_params(model)

    # Our count_total_params logic should match a simple deduplicated array count
    # for static mode (since it doesn't use masked layers by default there)
    unique_arrays = {}
    for leaf in jax.tree_util.tree_leaves(model):
        if eqx.is_array(leaf):
            unique_arrays[id(leaf)] = leaf.size

    manual_count = sum(unique_arrays.values())
    assert p_count == manual_count
