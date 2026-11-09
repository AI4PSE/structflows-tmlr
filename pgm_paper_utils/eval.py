"""Evaluation utilities for PGM flow models.

This module contains the core evaluation logic for comparing learned distributions
against ground truth PGMs using various metrics including SWD, NLL, and KL divergence.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jax import random
from typing import Dict, Tuple, FrozenSet, Optional


# --- Module-level jitted helpers -------------------------------------------
# These take the PGM / flow modules as ARGUMENTS (filter_jit traces their
# arrays as inputs, statics by structure), so tracing/lowering happens once
# per shape and is reused across conditions, checkpoints, and queries within
# a process. Previously the per-condition work lived in a jax.lax.map closure
# capturing the modules (incl. a 219MB dataset) as embedded constants, which
# re-traced a giant graph on every evaluate_distribution call (~10 min/query).

@eqx.filter_jit
def _b_sample_mod(mod, keys, condition_values, targets):
    return jax.vmap(lambda k: mod.sample(k, conditions=condition_values, targets=targets))(keys)


@eqx.filter_jit
def _b_sample_mod_test(mod, keys, condition_values, targets):
    return jax.vmap(lambda k: mod.sample(k, conditions=condition_values, targets=targets,
                                         training=False))(keys)


@eqx.filter_jit
def _b_log_prob_mod(mod, x, targets, conditions):
    return jax.vmap(lambda xx: mod.log_prob(xx, targets=targets, conditions=conditions))(x)


def evaluate_distribution_placeholder(*args, **kwargs):
    """Placeholder evaluation function for testing.
    
    Returns dummy metrics sampled from a normal distribution.
    """
    return {
        "SWD": np.random.normal(),
        "Test_NLL": np.random.normal(),
        "KL": np.random.normal(),
        "k_hat": np.random.normal()
    }


def evaluate_distribution(
    pgm,
    dist_obj,
    targets: FrozenSet[str],
    conditions: FrozenSet[str],
    seed: int,
    num_samples: int = 1000,
    n_condition_samples: int = 10,
    task: str = 'inference'
) -> Dict[str, float]:
    """Simplified distribution evaluation using vmap over pgm.sample and dist_obj.sample.
    
    Algorithm:
    1. Sample a batch of conditions from the PGM
    2. vmap(lambda k, c: pgm.sample(k, c, targets))(keys, conditions)
    3. vmap(lambda k, c: dist_obj.sample(k, c, targets))(keys, conditions)
    4. Compare samples using SWD
    
    Args:
        pgm: Ground truth PGM
        dist_obj: Learned distribution
        targets: Set of target variable names
        conditions: Set of condition variable names
        seed: Random seed
        num_samples: Number of samples per condition
        n_condition_samples: Number of different condition values to sample
        task: 'inference' or 'causal' (causal adds counterfactual SWD)
    
    Returns:
        Dict of averaged metrics:
        - SWD: Sliced Wasserstein Distance
        - NLL: Negative Log Likelihood
        - KL: KL divergence
        - Counterfactual_SWD: SWD on counterfactual samples
        - Counterfactual_NLL: NLL on counterfactual samples
        - Counterfactual_KL: KL on counterfactual samples
    """
    num_samples = num_samples // n_condition_samples
    from structflow.bayes.metrics import compute_swd_jax, _flatten_samples, compute_sinkhorn_jax
    
    start_time = time.time()
    print(f"      [evaluate_distribution_v3 START] targets={targets}, conditions={conditions}")
    
    # Ensure targets is a frozenset
    if not isinstance(targets, frozenset):
        targets = frozenset(targets)
    
    # Import utilities
    from structflow.pgms.image_benchmarks import MNISTPGM
    from structflow.pgms.colored_mnist import ColoredMNISTPGM
    
    print(f"      [Step 1] Setting up sampling functions...")
    
    # Generate keys
    main_key = random.PRNGKey(seed + 777)
    cond_key, pgm_key, dist_key, cf_key, cf_factual_key, swd_key = random.split(main_key, 6)
    cond_keys = random.split(cond_key, n_condition_samples)
    pgm_keys = random.split(pgm_key, n_condition_samples)
    dist_keys = random.split(dist_key, n_condition_samples * num_samples).reshape((n_condition_samples, num_samples, 2))
    cf_keys = random.split(cf_key, n_condition_samples)
    cf_factual_keys = random.split(cf_factual_key, n_condition_samples * num_samples).reshape((n_condition_samples, num_samples, 2))
    swd_keys = random.split(swd_key, n_condition_samples)
    
    def sample_pgm_joint(key):
        """Sample from PGM joint (all variables)."""
        if isinstance(pgm, (MNISTPGM, ColoredMNISTPGM)):
            return pgm.sample(key, training=False)
        else:
            return pgm.sample(key)
    
    def sample_pgm_conditional(key, condition_values):
        """Sample targets from PGM given fixed condition values."""
        if isinstance(pgm, (MNISTPGM, ColoredMNISTPGM)):
            return pgm.sample(key, conditions=condition_values, targets=targets, training=False)
        else:
            return pgm.sample(key, conditions=condition_values, targets=targets)
    
    def sample_dist_conditional(key, condition_values):
        """Sample targets from learned distribution given fixed condition values."""
        return dist_obj.sample(key, conditions=condition_values, targets=targets)

    def log_prob_pgm(x):
        """Log probability of PGM given condition values."""
        return pgm.log_prob(x, conditions=conditions, targets=targets)

    def log_prob_dist(x):
        """Log probability of learned distribution given condition values."""
        return dist_obj.log_prob(x, conditions=conditions, targets=targets)
    
    # Pre-check tractability *outside* jax.lax.map as Python bools.
    #
    # CRITICAL: try/except inside jax.lax.map (or any JAX-traced function) does NOT
    # work correctly. JAX traces the function once to build an XLA program; Python
    # exceptions raised during tracing are caught by the except clause but the XLA
    # program is left in a corrupted partial state. This produces downstream
    # "Too small divisible part of the contracting dimension" XlaRuntimeErrors because
    # the compiled kernel has a wrong shape from the aborted trace.
    #
    # The fix: resolve ALL conditional branching on tractability as Python bools HERE,
    # before jax.lax.map. JAX will see only one branch when tracing compare_single().

    # (1) PGM log_prob tractability — raises NotImplementedError for inference queries
    pgm_log_prob_tractable = False
    try:
        from structflow.pgms.image_benchmarks import MNISTPGM
        if not isinstance(pgm, (MNISTPGM, ColoredMNISTPGM)) and hasattr(pgm, 'structure'):
            pgm_log_prob_tractable = bool(pgm.structure.is_tractable_query(
                set(targets), set(conditions)
            ))
    except Exception:
        pgm_log_prob_tractable = False

    # (2) Flow (dist_obj) log_prob tractability.
    # GuidanceBasedDistribution (guided/joint mode) can SAMPLE p(targets|conditions)
    # via guidance but CANNOT evaluate log_prob unless all variables are provided —
    # the backward ODE needs the full joint state. This is separate from the structure's
    # get_tractable_distributions(), which only reflects sampling tractability.
    dist_log_prob_tractable = True
    try:
        from structflow.conditional_flows.guidance.dist import GuidanceBasedDistribution
        # Unwrap NormalisedDistribution to inspect the inner dist
        inner_dist = getattr(dist_obj, 'dist', dist_obj)
        if isinstance(inner_dist, GuidanceBasedDistribution) and inner_dist.key_order is not None:
            # Joint/guided mode: log_prob requires ALL variables in targets ∪ conditions
            all_vars = set(inner_dist.key_order)
            provided = set(targets) | set(conditions)
            dist_log_prob_tractable = all_vars.issubset(provided)
    except Exception:
        dist_log_prob_tractable = True  # assume tractable if check fails
    try:
        from structflow.conditional_flows.dcm.dist import DCMDistribution
        inner_dist = getattr(dist_obj, 'dist', dist_obj)
        if isinstance(inner_dist, DCMDistribution):
            # DCM (per-node diffusion) backbone: no conditional density at
            # all — DCMDistribution.log_prob raises NotImplementedError
            # eagerly, so it must never be called (deciding this statically
            # here keeps the NotImplementedError out of traced code paths).
            dist_log_prob_tractable = False
    except Exception:
        pass

    # (3) Counterfactual computation — only meaningful for causal task with conditions,
    # and only when the PGM provides ground-truth counterfactuals for this intervention
    # set (dataset-backed PGMs support a subset of interventions, e.g. do(color) but not
    # do(digit) on Colored-MNIST). Probe eagerly with a single example so the query's
    # other metrics (SWD/NLL/KL) survive when counterfactual ground truth is unavailable.
    compute_counterfactuals = (task == 'causal') and (len(conditions) > 0)
    if compute_counterfactuals:
        try:
            probe_factual = sample_pgm_joint(cond_keys[0])
            probe_interventions = {k: probe_factual[k] for k in conditions}
            pgm.counterfactual(probe_factual, probe_interventions, targets=targets)
        except NotImplementedError:
            compute_counterfactuals = False
    if compute_counterfactuals:
        # Flow-side capability must be decided STATICALLY (like the
        # dist_log_prob_tractable check above): calling the jitted
        # counterfactual as a probe raises NotImplementedError mid-trace,
        # which aborts the trace and poisons XLA state — later legitimate
        # calls then fail with "Too small divisible part of the contracting
        # dimension". GuidanceBasedDistribution (joint mode) only supports
        # counterfactual densities over exactly the non-intervened set.
        try:
            from structflow.conditional_flows.guidance.dist import GuidanceBasedDistribution
            inner_dist = getattr(dist_obj, 'dist', dist_obj)
            if isinstance(inner_dist, GuidanceBasedDistribution) and inner_dist.key_order is not None:
                non_intervened = set(inner_dist.key_order) - set(conditions)
                if set(targets) != non_intervened:
                    compute_counterfactuals = False
        except Exception:
            pass

    print(f"      [Step 2] Sampling conditions and conditional samples...")
    print(f"      [Tractability] pgm_log_prob_tractable={pgm_log_prob_tractable}, dist_log_prob_tractable={dist_log_prob_tractable}, compute_counterfactuals={compute_counterfactuals}")
    
    # Per-condition evaluation: plain Python loop over condition values, with
    # all heavy work routed through the module-level jitted helpers above.
    from structflow.pgms.image_benchmarks import MNISTPGM as _M
    from structflow.pgms.colored_mnist import ColoredMNISTPGM as _C
    _pgm_sampler = _b_sample_mod_test if isinstance(pgm, (_M, _C)) else _b_sample_mod
    swds, nlls, kls, cf_swds, cf_nlls, cf_kls = [], [], [], [], [], []
    for i in range(n_condition_samples):
        _it_t0 = time.time()
        cond_sample = sample_pgm_joint(cond_keys[i])
        condition_values = {k: cond_sample[k] for k in conditions}

        pgm_targets_i = _pgm_sampler(pgm, random.split(pgm_keys[i], num_samples),
                                     condition_values, targets)
        # CRITICAL: dataset-backed PGMs commit their arrays (and thus every
        # eager sample derived from them) to the CPU device; feeding those
        # committed values into the flow's jitted sampler silently pins the
        # whole ODE computation to CPU (~80s vs ~0.2s per batch). Re-home
        # everything destined for the flow onto the default device.
        _dev = jax.devices()[0]
        dist_targets_i = _b_sample_mod(dist_obj, dist_keys[i],
                                       jax.device_put(condition_values, _dev), targets)
        X = jax.device_put(_flatten_samples(pgm_targets_i), _dev)
        Y = _flatten_samples(dist_targets_i)
        swds.append(compute_swd_jax(swd_keys[i], X, Y, num_projections=100))

        full_pgm_samples = {**pgm_targets_i}
        for k2 in conditions:
            full_pgm_samples[k2] = jnp.broadcast_to(
                condition_values[k2], (num_samples,) + condition_values[k2].shape)

        if dist_log_prob_tractable and pgm_log_prob_tractable:
            # See the tractability pre-checks above: both required, and the
            # flow NLL is skipped entirely when ground truth is unavailable
            # (the exact Jacobian log-det is O(D^2) memory).
            nll_all = -_b_log_prob_mod(dist_obj, jax.device_put(full_pgm_samples, jax.devices()[0]),
                                       targets, conditions)
            nlls.append(jnp.mean(nll_all))
            kls.append(jnp.mean(_b_log_prob_mod(pgm, full_pgm_samples, targets, conditions) + nll_all))
        else:
            nlls.append(jnp.inf)
            kls.append(jnp.inf)

        if compute_counterfactuals:
            factual_pgm = jax.vmap(sample_pgm_joint)(cf_factual_keys[i])
            interventions = {k2: jnp.broadcast_to(
                condition_values[k2], (num_samples,) + condition_values[k2].shape)
                for k2 in conditions}
            pgm_cf, pgm_cf_log_probs = pgm.b_counterfactual(factual_pgm, interventions, targets=targets)
            _dev = jax.devices()[0]
            dist_cf, dist_cf_log_probs = dist_obj.b_counterfactual(
                jax.device_put(factual_pgm, _dev), jax.device_put(interventions, _dev),
                targets=targets)
            cf_swds.append(compute_swd_jax(cf_keys[i],
                                           jax.device_put(_flatten_samples(pgm_cf), _dev),
                                           _flatten_samples(dist_cf), num_projections=100))
            cf_nlls.append(-jnp.mean(dist_cf_log_probs))
            cf_kls.append(jnp.mean(pgm_cf_log_probs - dist_cf_log_probs))
        else:
            cf_swds.append(jnp.inf)
            cf_nlls.append(jnp.inf)
            cf_kls.append(jnp.inf)
        print(f"      [cond {i+1}/{n_condition_samples}] {time.time()-_it_t0:.1f}s", flush=True)

    swd, nll, kl = (jnp.mean(jnp.array(swds)), jnp.mean(jnp.array(nlls)), jnp.mean(jnp.array(kls)))
    cf_swd, cf_nll, cf_kl = (jnp.mean(jnp.array(cf_swds)), jnp.mean(jnp.array(cf_nlls)), jnp.mean(jnp.array(cf_kls)))
    
    metrics = {
        "SWD": float(swd),
        "NLL": float(nll),
        "KL": float(kl),
        "Counterfactual_SWD": float(cf_swd),
        "Counterfactual_NLL": float(cf_nll),
        "Counterfactual_KL": float(cf_kl)
    }

    for k, v in metrics.items():
        print(f"      [evaluate_distribution_v3] {k}: {float(v):.6f}")
    
    print(f"      [evaluate_distribution_v3 SUCCESS] Total time: {time.time() - start_time:.2f}s")
    return metrics
