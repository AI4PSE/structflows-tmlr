import pytest
import jax
from jax import random
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.utils import get_test_queries
from structflow.pgms.pgm_compiler import PGMCompiler

# The old 'nonlinear' PGM type (pgm_benchmarks.make_dense_nonlinear_pgm: a
# dense DAG with nonlinear mechanisms) no longer exists in create_pgm. Its
# current equivalent is the 'dense' structure with a nonlinear mechanism, and
# the node-count key is now 'n_nodes' (was 'n_vars', which is silently ignored).
def make_dense_nonlinear_pgm(n_nodes, key):
    config = {'n_nodes': n_nodes, 'mechanism_type': 'nonlinear_parent'}
    return create_pgm('dense', config, key)


def passes_query_filter(targets, conditions, n_vars):
    """The documented filter in get_test_queries (added after these tests were
    written): keep single-target queries and the full joint; drop partial
    joints p(x0, x1) and multi-target conditionals p(x0, x1 | x2)."""
    return len(targets) == 1 or (len(conditions) == 0 and len(targets) == n_vars)

def test_get_test_queries_nonlinear_causal():
    # Setup 3-node non-linear PGM
    key = random.PRNGKey(0)
    pgm = make_dense_nonlinear_pgm(3, key)
    
    # Compile Causal Structure
    dag = pgm.get_pgm_structure()
    compiler = PGMCompiler(dag, dims=pgm.variable_dims)
    structure = compiler.compile_causal()
    
    # Get Queries
    queries = get_test_queries(pgm, structure)
    
    # Expected queries for 3-node dense DAG in causal mode:
    # 1. p(x0, x1, x2) [Joint]
    # 2. p(x1 | x0) [Local Causal]
    # 3. p(x2 | x0, x1) [Local Causal]
    
    query_strs = [q[2] for q in queries]
    
    assert "p(x0, x1, x2)" in query_strs
    assert "p(x1 | x0)" in query_strs
    assert "p(x2 | x0, x1)" in query_strs
    
    # Verify we don't have inference queries (that would require inversion)
    # e.g. p(x0 | x1) or p(x0, x1 | x2)
    assert "p(x0 | x1)" not in query_strs
    assert "p(x0, x1 | x2)" not in query_strs

def test_get_test_queries_nonlinear_inference():
    # Setup 3-node non-linear PGM
    key = random.PRNGKey(0)
    pgm = make_dense_nonlinear_pgm(3, key)
    
    # Compile Inference Structure (optimized for roots | leaves)
    # roots = {x0}, leaves = {x2}
    targets, conditions = pgm.get_inference_targets_and_conditions()
    dag = pgm.get_pgm_structure()
    compiler = PGMCompiler(dag, dims=pgm.variable_dims)
    structure = compiler.compile_inference(targets=frozenset(targets), conditions=frozenset(conditions))
    
    # Get Queries
    queries = get_test_queries(pgm, structure)
    
    query_strs = [q[2] for q in queries]
    
    # The PGM supports:
    # - Joint: p(x0, x1, x2)
    # - Inference: p(x0 | x2) [Global Inference]
    # - Inference: p(x0, x1 | x2) [Local Inference p(parents | child) for child x2]
    # - Inference: p(x0 | x1) [Local Inference p(parents | child) for child x1]
    
    # The Inference Structure (order [x2, x1, x0]) supports:
    # - p(x2, x1, x0) [Joint]
    # - p(x1, x0 | x2) [Prefix]
    # - p(x0 | x2, x1) [Prefix]
    
    assert "p(x0, x1, x2)" in query_strs
    # p(x0, x1 | x2) is tractable for this structure and supported by the PGM,
    # but get_test_queries deliberately drops multi-target conditionals; it is
    # evaluated through its single-target chain-rule factors in the inference
    # order [x2, x1, x0] instead.
    assert "p(x0, x1 | x2)" not in query_strs
    assert "p(x1 | x2)" in query_strs
    assert "p(x0 | x1, x2)" in query_strs
    
    # Note: p(x0 | x2) might be missing if it's not directly tractable (Parent Rule violation)
    # because x0 depends on x1 in the inference structure.
    # This is an important check to see what's actually being evaluated.
    
    # Let's check if the inference query we actually want is there!
    # If it's not, we might want to know.
    # For UniversalNonLinearPGM, the supported distributions are specific.
    
    # Check that we have the joint
    assert any("x0" in q[0] and "x1" in q[0] and "x2" in q[0] and not q[1] for q in queries)

def test_get_test_queries_no_missing_pgm_supported():
    """
    Verify that we are not missing any PGM supported distributions 
    that ARE tractable by the flow.
    """
    key = random.PRNGKey(42)
    pgm = make_dense_nonlinear_pgm(4, key)
    
    dag = pgm.get_pgm_structure()
    compiler = PGMCompiler(dag, dims=pgm.variable_dims)
    
    for task in ['causal', 'inference']:
        if task == 'causal':
            structure = compiler.compile_causal()
        else:
            targets, conditions = pgm.get_inference_targets_and_conditions()
            structure = compiler.compile_inference(targets=frozenset(targets), conditions=frozenset(conditions))
            
        queries = get_test_queries(pgm, structure)
        query_keys = {(q[0], q[1]) for q in queries}
        
        supported_pgm = pgm.get_supported_distributions()
        for dist in supported_pgm:
            t, c = dist['targets'], dist['conditions']
            if structure.is_tractable_query(set(t), set(c)):
                if passes_query_filter(t, c, len(structure.block_order)):
                    assert (t, c) in query_keys, f"Missing tractable PGM distribution: p({t} | {c}) for task {task}"
                else:
                    assert (t, c) not in query_keys, f"Filtered query leaked into test queries: p({t} | {c}) for task {task}"

        # Soundness: every returned query is PGM-supported, flow-tractable and
        # passes the filter.
        supported_keys = {(d['targets'], d['conditions']) for d in supported_pgm}
        flow_keys = {(d['targets'], d['conditions']) for d in structure.get_tractable_distributions()}
        for t, c in query_keys:
            assert (t, c) in supported_keys and (t, c) in flow_keys
            assert passes_query_filter(t, c, len(structure.block_order))
