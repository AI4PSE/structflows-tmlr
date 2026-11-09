# Evaluation Tests and Capabilities

This directory contains tests for the `evaluate_distribution` function and related evaluation utilities.

## `test_eval_integration.py`

This test suite performs a comprehensive integration test using **exact Gaussian velocity functions**. It verifies that:
1.  The evaluation pipeline runs without crashing for various modes and queries.
2.  The computed metrics (SWD, NLL, KL) are numerically correct (close to 0 for exact models).
3.  Edge cases (e.g., empty conditions, unsupported marginal log_probs) are handled gracefully.

### Supported Modes
-   `static`: Static Conditional Flow (OneSidedField)
-   `dynamic`: Parallel/Dynamic Flow (ParallelBlockTriangular)
-   `guided`: Guidance Based Distribution (Joint Flow + Reconstruction Guidance)

## Capabilities Matrix

The following table summarizes the capabilities of each distribution mode regarding Sampling and Log Probability computation for different query types.

| Mode | Query Type | Sampling | Log Probability (Flow) | Notes |
|------|------------|----------|------------------------|-------|
| **Static / Dynamic** | Joint (All vars) | ✅ | ✅ | Full support. |
| | Conditional (Full) | ✅ | ✅ | |
| | Marginal / Partial | ✅ | ✅ | Can integrate/sample arbitrary subsets. |
| **Guided (Joint)** | Joint (All vars) | ✅ | ✅ | |
| | Conditional (Full) | ✅ | ✅ | e.g., `p(x0 \| x1)` where `{x0, x1}` are all vars. |
| | Marginal / Partial | ✅ | ❌ | Requires all variables to integrate ODE. |

### Metric Availability

Metrics depend on both the Flow and the Ground Truth PGM capabilities.

| Metric | Computation | Supported by Guided (Marginal) | Supported by Guided (Full) | Note on PGM |
|--------|-------------|------------------------------|----------------------------|-------------|
| **SWD** | Sampling (Flow) vs Sampling (PGM) | ✅ | ✅ | Always computable if sampling works. |
| **NLL** | Log Prob (Flow) | ❌ (`inf`) | ✅ | Computable if Flow supports log_p. |
| **KL** | Log Prob (PGM) - Log Prob (Flow) | ❌ (`inf`) | ⚠️ | **Fails (`inf`) if PGM cannot compute log_p** (e.g., non-ancestral queries). |
| **Counterfactual SWD** | Causal Task + Conditions | ✅ | ✅ | |
| **Counterfactual NLL** | Causal Task + Log Prob (Flow) | ❌ (`inf`) | ⚠️ | May report joint log_prob of non-intervened vars. |

> **Note on KL Divergence:** You may see cases where **NLL is finite** but **KL is `inf`**. This occurs when the Flow can compute the conditional log probability (e.g., `Guided` mode with all variables), but the **Ground Truth PGM** implementation considers that query intractable (e.g., `p(Cause | Effect)` in inference mode).

### Test Structure

Tests are parameterized by:
-   **Mode**: `static`, `dynamic`, `guided`
-   **Task**: `inference` (standard metrics), `causal` (adds counterfactual metrics)
-   **Query**:
    -   `marginal`: `p(x0)`
    -   `conditional`: `p(x0 | x1)`
    -   `joint`: `p(x0, x1)`
    -   `conditional_reverse`: `p(x1 | x0)`

Run the tests with:
```bash
pytest pgm_paper_utils/tests/test_eval_integration.py -v
```
