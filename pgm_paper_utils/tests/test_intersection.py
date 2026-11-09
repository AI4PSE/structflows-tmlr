import pytest
import jax
import jax.numpy as jnp
from jax import random
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.utils import get_test_queries
from structflow.pgms.pgm_compiler import PGMCompiler
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.learner_factory import create_learner

def test_query_intersection_is_exhaustive():
    """
    Test that the intersection between PGM supported and Flow tractable 
    distributions includes all distributions the flow can compute.
    """
    key = random.PRNGKey(42)
    config = {
        'n_nodes': 6,  # was 'n_vars', which create_pgm now ignores (-> 16 nodes)
        'mode': 'static',
        'n_steps': 5,
        'hidden_dim': 16,
        'lr': 1e-3,
        'train_steps': 100
    }
    
    # 1. Create PGM
    # Use a random one to be sure it works in general cases
    pgm = create_pgm('random', config, key)
    
    # 2. Compile Structure (Inference mode)
    targets, conditions = pgm.get_inference_targets_and_conditions()
    dag = pgm.get_pgm_structure()
    compiler = PGMCompiler(dag, dims=pgm.variable_dims)
    structure = compiler.compile_inference(targets=frozenset(targets), conditions=frozenset(conditions))
    
    # 3. Get Test Queries (the intersection)
    queries = get_test_queries(pgm, structure)
    
    # 4. Verification: All tractable flow distributions MUST be in the intersection
    # because the PGM now supports everything the flow does (via MCMC if needed)
    flow_tractable = structure.get_tractable_distributions()
    
    # get_test_queries deliberately keeps only single-target queries and the
    # full joint (partial joints / multi-target conditionals are dropped; see
    # its docstring). That filter postdates this test, so exhaustiveness is
    # checked for the queries the filter keeps, and the dropped ones must be
    # exactly those the filter excludes.
    n_vars = len(structure.block_order)
    def passes_query_filter(t, c):
        return len(t) == 1 or (len(c) == 0 and len(t) == n_vars)

    query_keys = {(q[0], q[1]) for q in queries}
    flow_keys = {(d['targets'], d['conditions']) for d in flow_tractable}
    for dist in flow_tractable:
        t, c = dist['targets'], dist['conditions']
        if passes_query_filter(t, c):
            assert (t, c) in query_keys, f"Missing tractable flow distribution in intersection: p({t} | {c})"
        else:
            assert (t, c) not in query_keys, f"Filtered query leaked into intersection: p({t} | {c})"
    # Soundness: nothing outside the flow-tractable set.
    assert query_keys <= flow_keys

    # 5. Runtime check: can we actually compute log_prob/sample for these?
    # Create a mock model using learner_factory
    model_config = config.copy()
    model_config['structure'] = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    model, normaliser, _, _ = create_learner(model_config, structure=structure)
    flow_dist = create_distribution(model_config, model, normaliser)
    
    # Test a few from the intersection
    for targets, conditions, q_str in queries[:5]:
        # PGM sample (some PGMs don't support num_samples yet)
        pgm_samples = pgm.sample(key, conditions={c: jnp.zeros((1,)) for c in conditions}, targets=targets)
        for t in targets:
            assert t in pgm_samples, f"Target {t} missing from PGM sample for query {q_str}"
        
        # Flow log_prob
        data = {n: jnp.zeros((1,)) for n in pgm.node_names}
        lp = flow_dist.log_prob(data, targets=list(targets), conditions=list(conditions))
        assert not jnp.isnan(lp)
        
        # Flow sample
        flow_samples = flow_dist.sample(key, conditions={c: jnp.zeros((1,)) for c in conditions}, targets=list(targets))
        for t in targets:
            assert t in flow_samples, f"Target {t} missing from Flow sample for query {q_str}"

def test_query_intersection_causal():
    """Verify intersection for causal task."""
    key = random.PRNGKey(0)
    # Old 'nonlinear' type = dense DAG with nonlinear mechanisms.
    pgm = create_pgm('dense', {'n_nodes': 3, 'mechanism_type': 'nonlinear_parent'}, key)
    
    dag = pgm.get_pgm_structure()
    compiler = PGMCompiler(dag, dims=pgm.variable_dims)
    structure = compiler.compile_causal()
    
    queries = get_test_queries(pgm, structure)
    query_strs = [q[2] for q in queries]
    
    # In causal mode, standard causal queries should be there
    assert "p(x0, x1, x2)" in query_strs
    assert "p(x1 | x0)" in query_strs
    assert "p(x2 | x0, x1)" in query_strs
