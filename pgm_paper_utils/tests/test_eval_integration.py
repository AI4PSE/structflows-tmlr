
import pytest
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import sys
import os
import numpy as np

from structflow.pgms.structure import BlockTriangularStructure
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.eval import evaluate_distribution
from structflow.conditional_flows.tests.numerical_utils import (
    create_exact_velocity_functions,
    ExactGaussianStaticVelocity,
    create_guided_field_from_velocities,
    analytical_gaussian_log_prob
)

# ============================================================================
# Mock PGM for Ground Truth
# ============================================================================

class GaussianPGM:
    """Independent Gaussians x_k ~ N(mean_k, var_k) (no edges).

    evaluate_distribution now decides ground-truth log_prob tractability from
    ``pgm.structure.is_tractable_query`` (no ``structure`` => NLL/KL are
    reported as inf), and probes ``pgm.counterfactual`` for one example before
    computing counterfactual metrics, so the mock exposes both.
    """
    def __init__(self, target_params, structure):
        self.target_params = target_params
        self.structure = structure
        self.dims = {k: v['mean'].shape[0] for k, v in target_params.items()}

    def sample(self, key, conditions=None, targets=None, training=False):
        keys = random.split(key, len(self.target_params))
        samples = {}
        for i, (k, params) in enumerate(self.target_params.items()):
            mu = params['mean']
            var = params['var']
            std = jnp.sqrt(var)
            s = mu + random.normal(keys[i], mu.shape) * std
            samples[k] = s
            
        if conditions:
            for k, v in conditions.items():
                samples[k] = v
                
        if targets:
            return {k: samples[k] for k in targets}
        return samples

    def log_prob(self, x, conditions=None, targets=None):
        lp = 0.0
        relevant_vars = targets if targets else x.keys()
        
        for k in relevant_vars:
            if conditions and k in conditions:
                continue
                
            val = x[k]
            mu = self.target_params[k]['mean']
            var = self.target_params[k]['var']
            lp += analytical_gaussian_log_prob(val, mu, var)
            
        return lp
        
    def b_sample(self, key, n_samples, conditions=None, targets=None):
        keys = random.split(key, n_samples)
        def single_sample(k):
            return self.sample(k, conditions, targets)
        return jax.vmap(single_sample)(keys)

    def counterfactual(self, factual_sample, interventions, targets=None):
        """Single-example counterfactual. With independent mechanisms the
        abducted noise reproduces the factual value of every non-intervened
        variable; intervened variables take their do() value. The log prob is
        that of the non-intervened targets (as in StructuralPGM)."""
        cf_samples = {k: v for k, v in factual_sample.items()}
        if interventions:
            for k, v in interventions.items():
                cf_samples[k] = v

        relevant_vars = targets if targets else list(cf_samples.keys())
        log_prob = 0.0
        for k in relevant_vars:
            if interventions and k in interventions:
                continue
            mu = self.target_params[k]['mean']
            var = self.target_params[k]['var']
            log_prob = log_prob + analytical_gaussian_log_prob(cf_samples[k], mu, var)

        if targets:
            cf_samples = {k: cf_samples[k] for k in targets}
        return cf_samples, jnp.asarray(log_prob)

    def b_counterfactual(self, factual_sample, interventions, targets=None):
        return jax.vmap(lambda f, i: self.counterfactual(f, i, targets=targets))(
            factual_sample, interventions)


# ============================================================================
# Exact velocity network for dynamic mode
# ============================================================================
#
# 'dynamic' now builds a DynamicV2Distribution via
# ParallelFieldAdapter(model): it needs model.net to look like a
# ParallelStructuredFn (structure, block_offsets, output_dim, time_embed,
# hidden_block_dims, layers), because log_prob / counterfactual call
# clip_tail(), which slices the network's layers per block. The old
# ExactGaussianParallelVelocity adapter has none of that (no .net, no layers).
#
# ExactParallelNet is a ParallelStructuredFn-shaped network that computes the
# exact Gaussian velocity and survives clip_tail: its first layer is a real
# DenseMaskedLinear that copies, for each block b, [x_b, t] into b's hidden
# units and writes b's index into a third hidden unit via the bias (so the
# block identity survives row/column clipping); the second "layer" is a plain
# callable (clip_tail keeps those as-is) that evaluates the exact velocity per
# hidden chunk. It is wrapped in OneSidedField_eta1 with LinearDeterministic,
# for which gamma == 0 and so get_velocity_b(x, t) == net(x, t) exactly (the
# same device create_guided_field_from_velocities uses for guided mode).
# Requires all blocks to have the same dimension.

class _TimeEmbed(eqx.Module):
    output_features: int = eqx.field(static=True)

    def __call__(self, t_arr):
        # (2,) so that .squeeze() in the network keeps a 1-D vector.
        return jnp.concatenate([t_arr, jnp.zeros_like(t_arr)])


class _ExactHead(eqx.Module):
    v_fns: list
    block_dim: int = eqx.field(static=True)

    def __call__(self, h):
        d = self.block_dim
        chunks = h.reshape(-1, d + 2)

        def one(chunk):
            x, t, idx = chunk[:d], chunk[d], chunk[d + 1]
            branches = [lambda x, t, f=f: f(x, t) for f in self.v_fns]
            return jax.lax.switch(jnp.round(idx).astype(jnp.int32), branches, x, t)

        return jax.vmap(one)(chunks).reshape(-1)


class ExactParallelNet(eqx.Module):
    time_embed: _TimeEmbed
    layers: list
    structure: BlockTriangularStructure
    block_dims: dict
    hidden_block_dims: dict
    output_dim: int
    block_offsets: dict = eqx.field(static=True)

    def __init__(self, structure, v_fns):
        from structflow.conditional_flows.parallel.layers import DenseMaskedLinear
        order = structure.block_order
        dims = {b: structure.dims[b] for b in order}
        d = dims[order[0]]
        assert all(v == d for v in dims.values()), "ExactParallelNet needs equal block dims"
        n = len(order)
        state_dim, t_dim = n * d, 2
        hidden = n * (d + 2)

        W = np.zeros((hidden, state_dim + t_dim), dtype=np.float32)
        bias = np.zeros((hidden,), dtype=np.float32)
        for i in range(n):
            r = i * (d + 2)
            W[r:r + d, i * d:(i + 1) * d] = np.eye(d)   # copy x_b
            W[r + d, state_dim] = 1.0                    # copy t
            bias[r + d + 1] = float(i)                   # block id
        mask = (W != 0).astype(np.float32)
        mask[np.arange(n) * (d + 2) + d + 1, :] = 0.0
        first = DenseMaskedLinear(state_dim + t_dim, hidden, jnp.asarray(mask), key=jax.random.PRNGKey(0))
        first = eqx.tree_at(lambda l: l.linear.weight, first, jnp.asarray(W))
        first = eqx.tree_at(lambda l: l.linear.bias, first, jnp.asarray(bias))

        self.time_embed = _TimeEmbed(output_features=t_dim)
        self.layers = [first, _ExactHead([v_fns[b] for b in order], d)]
        self.structure = structure
        self.block_dims = dims
        self.hidden_block_dims = {b: d + 2 for b in order}
        self.output_dim = state_dim
        self.block_offsets = {b: (i * d, (i + 1) * d) for i, b in enumerate(order)}

    def __call__(self, x_state, t):
        # Mirrors ParallelStructuredFn / ClippedParallelFn.__call__
        t_embed = self.time_embed(jnp.array([t])).squeeze()
        h = jnp.concatenate([x_state, t_embed])
        for layer in self.layers:
            h = layer(h)
        return jnp.clip(h, -10.0, 10.0)


def create_exact_dynamic_model(structure, v_fns):
    from sinterp.stochasticfield import OneSidedField_eta1
    from sinterp.interpolants import LinearDeterministic
    return OneSidedField_eta1(LinearDeterministic(), ExactParallelNet(structure, v_fns))

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def independent_pgm_setup():
    TARGET_PARAMS = {
        'x0': {'mean': jnp.array([1.5]), 'var': jnp.array([3.0])},
        'x1': {'mean': jnp.array([-0.5]), 'var': jnp.array([2.5])},
    }

    INDEPENDENT_STRUCTURE = BlockTriangularStructure(
        block_order=['x0', 'x1'],
        dependencies={'x0': [], 'x1': []},
        dims={'x0': 1, 'x1': 1}
    )
    
    pgm = GaussianPGM(TARGET_PARAMS, INDEPENDENT_STRUCTURE)
    exact_velocity_fns = create_exact_velocity_functions(INDEPENDENT_STRUCTURE, TARGET_PARAMS)
    
    return pgm, INDEPENDENT_STRUCTURE, exact_velocity_fns

@pytest.fixture(scope="module")
def models(independent_pgm_setup):
    pgm, structure, exact_velocity_fns = independent_pgm_setup
    
    # Static
    model_static = ExactGaussianStaticVelocity(structure, exact_velocity_fns)
    
    # Dynamic (DynamicV2 needs a ParallelStructuredFn-shaped net; see above)
    model_dynamic = create_exact_dynamic_model(structure, exact_velocity_fns)
    
    # Guided
    dims = {name: structure.dims[name] for name in structure.block_order}
    model_guided = create_guided_field_from_velocities(
        exact_velocity_fns,
        structure.block_order,
        dims=dims
    )
    
    return {
        'static': model_static,
        'dynamic': model_dynamic,
        'guided': model_guided
    }

# ============================================================================
# Tests
# ============================================================================

KL_TOL_PER_DIM = 0.05

# Regression guard (fixed 2026-10-26): guided conditional cases.
GUIDED_CONDITIONAL_LOG_PROB_BUG = (
    "Fixed library bug (conditional_flows/guidance/dist.py GuidanceBasedDistribution.log_prob): "
    "for p(targets | conditions) it adds the base log-density of ALL dims (incl. the "
    "condition dims) but the divergence of the target dims only. On this exact "
    "independent model guided p(x0|x1) has NLL 3.40 vs 2.06 for static (KL ~1.36-1.49 "
    "~= E[-log N(z_cond)] = 1.42), although its samples match static exactly."
)

@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
@pytest.mark.parametrize("task", ["inference", "causal"])
@pytest.mark.parametrize("query_type,targets,conditions", [
    ("marginal", frozenset(['x0']), frozenset()),
    ("conditional", frozenset(['x0']), frozenset(['x1'])),
    ("joint", frozenset(['x0', 'x1']), frozenset()),
    ("conditional_reverse", frozenset(['x1']), frozenset(['x0'])),
])
def test_eval_metrics(request, independent_pgm_setup, models, mode, task, query_type, targets, conditions):
    pgm, structure, _ = independent_pgm_setup
    model = models[mode]
    
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    config = {
        'mode': mode,
        'structure': structure_dict,
        'n_steps': 50
    }
    
    dist = create_distribution(config, model)
    
    # Run evaluation
    metrics = evaluate_distribution(
        pgm=pgm,
        dist_obj=dist,
        targets=targets,
        conditions=conditions,
        seed=0,
        num_samples=100,
        n_condition_samples=5,
        task=task
    )
    
    # Check SWD (Should always be finite and reasonable for exact models)
    assert np.isfinite(metrics['SWD'])
    # Exact models should have low SWD
    assert metrics['SWD'] < 0.5, f"SWD too high: {metrics['SWD']}"
    
    # Check NLL and KL
    if mode == 'guided' and len(targets | conditions) < len(structure.block_order):
        # Guided mode doesn't support marginals (requires all vars)
        # Note: In our setup, 'joint' and 'conditional_reverse' might cover all vars depending on set logic
        # 'x0', 'x1' are all vars.
        all_vars = {'x0', 'x1'}
        query_vars = targets | conditions
        
        if query_vars == all_vars:
            # Should work
            assert np.isfinite(metrics['NLL']), f"NLL should be finite for full query in guided mode"
            assert np.isfinite(metrics['KL']), f"KL should be finite for full query in guided mode"
        else:
            # Should be inf
            assert not np.isfinite(metrics['NLL']), f"NLL should be inf for marginal query in guided mode"
            assert not np.isfinite(metrics['KL']), f"KL should be inf for marginal query in guided mode"
    else:
        # Static/Dynamic support all queries
        assert np.isfinite(metrics['NLL'])
        assert np.isfinite(metrics['KL'])

    # The models are exact, so KL(p || q) must vanish up to ODE discretisation.
    # (Measured: ~0.02 nats per target dim at n_steps=50 for static/dynamic/
    # guided-joint -- the first-order divergence quadrature in the flows'
    # log_prob, bias ~ dt/2 * [div v(1) - div v(0)].)
    if np.isfinite(metrics['KL']):
        assert abs(metrics['KL']) < KL_TOL_PER_DIM * len(targets), f"KL of an exact model: {metrics['KL']}"
        
    # Check Counterfactuals
    if task == 'inference':
        # Should be inf (skipped)
        assert not np.isfinite(metrics['Counterfactual_SWD'])
        assert not np.isfinite(metrics['Counterfactual_NLL'])
        assert not np.isfinite(metrics['Counterfactual_KL'])
    elif task == 'causal':
        if len(conditions) == 0:
            # Cannot do counterfactuals without conditions
            assert not np.isfinite(metrics['Counterfactual_SWD'])
            assert not np.isfinite(metrics['Counterfactual_NLL'])
            assert not np.isfinite(metrics['Counterfactual_KL'])
        else:
            # Should be finite
            assert np.isfinite(metrics['Counterfactual_SWD'])
            assert metrics['Counterfactual_SWD'] < 0.5
            
            # Check Counterfactual NLL/KL consistency
            # If NLL/KL were finite, CF metrics likely finite too (unless Guided marginal issue applies)
            
            if mode == 'guided' and len(targets | conditions) < len(structure.block_order):
                assert not np.isfinite(metrics['Counterfactual_NLL'])
                assert not np.isfinite(metrics['Counterfactual_KL'])
            else:
                 assert np.isfinite(metrics['Counterfactual_NLL'])
                 assert np.isfinite(metrics['Counterfactual_KL'])
                 assert abs(metrics['Counterfactual_KL']) < KL_TOL_PER_DIM * len(targets), \
                     f"Counterfactual_KL of an exact model: {metrics['Counterfactual_KL']}"



# ============================================================================
# Dynamic mode with the production field type
# ============================================================================
#
# The 'dynamic' fixture above wraps an exact *velocity* net in a
# LinearDeterministic field, where get_velocity_b(x, t) == net(x, t). In
# production (flow_factory.create_dynamic_flow) the net is an eta1 denoiser
# inside OneSidedField_eta1(OneSidedLinear), where the two differ. Here the
# net computes the exact denoiser E[x1 | x_t] = m + t v / s_t^2 (x_t - t m),
# s_t^2 = (1-t)^2 + t^2 v, so the distribution is exact in production form.

# Regression guard (fixed 2026-10-26): test_eval_dynamic_production_field_exact_model.
DYNAMIC_LOG_PROB_BUG = (
    "Fixed library bug (conditional_flows/dynamicv2/dist.py DynamicV2Distribution.log_prob, "
    "also .counterfactual): the divergence is taken of clip_tail(velocity_fn) = the RAW "
    "ParallelStructuredFn (the eta1 denoiser), not of the field velocity "
    "get_velocity_b that the backward trajectory integrates. Exact model: joint "
    "KL 1.15 nats (n_steps=50) / 1.14 (n_steps=200); with the sampler velocity's "
    "divergence 0.012 / 0.006. Sampling is unaffected."
)


def create_exact_dynamic_model_eta1(target_params, structure):
    from sinterp.stochasticfield import OneSidedField_eta1
    from sinterp.interpolants import OneSidedLinear

    def eta1_fn(m, v):
        return lambda x, t: m + t * v / ((1 - t) ** 2 + t ** 2 * v) * (x - t * m)

    eta1_fns = {k: eta1_fn(p['mean'], p['var']) for k, p in target_params.items()}
    return OneSidedField_eta1(OneSidedLinear(), ExactParallelNet(structure, eta1_fns))


@pytest.mark.parametrize("query_type,targets,conditions", [
    ("joint", frozenset(['x0', 'x1']), frozenset()),
    ("marginal", frozenset(['x0']), frozenset()),
])
def test_eval_dynamic_production_field_exact_model(independent_pgm_setup, query_type, targets, conditions):
    pgm, structure, _ = independent_pgm_setup
    model = create_exact_dynamic_model_eta1(pgm.target_params, structure)
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    dist = create_distribution({'mode': 'dynamic', 'structure': structure_dict, 'n_steps': 50}, model)

    metrics = evaluate_distribution(
        pgm=pgm, dist_obj=dist, targets=targets, conditions=conditions,
        seed=0, num_samples=100, n_condition_samples=5, task='inference')

    assert np.isfinite(metrics['SWD']) and metrics['SWD'] < 0.5
    assert np.isfinite(metrics['NLL'])
    assert abs(metrics['KL']) < KL_TOL_PER_DIM * len(targets), f"KL of an exact model: {metrics['KL']}"
