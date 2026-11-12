# Structured Flows for Causal Modelling and Statistical Inference

Experiments, evaluation, figures and LaTeX source for the TMLR paper

> Alexander Telfar and Stefan Sandfeld. *Structured Flows for Causal Modelling
> and Statistical Inference.* Transactions on Machine Learning Research, 2026.
> [OpenReview](https://openreview.net/forum?id=5cFwygAUFt)

The method itself is the library [structflows](https://github.com/AI4PSE/structflows),
built on [sinterp](https://github.com/act65/sinterp) (packages `sinterp` and `flow`).
This repository pins a structflow release (which in turn pins sinterp); the tag
`tmlr-camera-ready` reproduces the published version.

## Layout

```
paper/                 LaTeX source (TMLR style); `make camera-ready` in paper/ builds the
                       paper (`make` builds main.pdf with revision highlighting)
  images/              every figure the paper includes, and pgm-structure.svg, the Inkscape
                       canvas of the hand-drawn diagrams
pgm_paper_utils/       config -> PGM / learner / distribution factories, evaluation (SWD, KL, NLL,
                       counterfactuals, NUTS reference posteriors); paths.py sets where results go
scripts/sweep.py       train a grid of (mode, PGM seed, learner seed) runs
scripts/evaluate_sweep.py   evaluate every finished run of a results directory into a CSV
scripts/ablations/     the analysis experiments (chains, moralization, sample efficiency,
                       amortization vs NUTS, backbone NFE, NUTS validation, ...)
scripts/plotters/      figure scripts; make_benchmark_figures.py draws the benchmark figures
Makefile               training, evaluation and figure recipes for the experiments in the paper
constraints.txt        package versions used for the paper's results
```

## Install

Python 3.10+ and a CUDA 12 GPU (the results were produced on RTX A6000s;
everything also runs on CPU, slowly).

```bash
pip install "jax[cuda12]==0.6.2"      # or, CPU only: pip install "jax[cpu]==0.6.2"
pip install -e ".[dev]" -c constraints.txt
```

MNIST and Colored-MNIST download MNIST from OpenML (via scikit-learn) on first use.

## Reproduce

Results go to `$STRUCTFLOW_RESULTS` (default `./results`), one directory per experiment
named `<experiment>_kr` (for example `results/nonlinear4_kr/`); figures go to
`paper/images` (override with `$STRUCTFLOW_IMAGES`). Pass the results location to
`make` as `RESULTS=...`.

| paper | command |
| :--- | :--- |
| benchmarks, 4 nodes (Sec. 4, App. results) | `make linear4 nonlinear4` |
| benchmarks, 8 nodes (App. results) | `make nonlinear8` |
| MNIST, Colored-MNIST (Sec. 4.2, App.) | `make mnist colored-mnist colored-mnist-conv colored-mnist-rho1` |
| chains: price of parallelism (Sec. 4.3) | `make chain-rw` |
| NUTS reference validation (App.) | `make nuts-validation` |
| ordering ablation (App.) | `make ordering-ablation` |
| backbone ablation, CNF vs per-node DDIM (App.) | `make backbone-ablation` (after `make nonlinear4`) |
| moralization, sample efficiency, amortization (App.) | `python scripts/ablations/<name>.py` |
| benchmark figures (Sec. 4, App.) | `make figures` (after the benchmark targets) |

Each target trains with `scripts/sweep.py` and then scores with
`scripts/evaluate_sweep.py`; both skip finished runs, so a target can be
re-issued after an interruption. The noise–data coupling for each conditioning
mode is set by `COUPLING` in the `Makefile` (paper appendix "Noise–Data
Coupling"). Ground-truth samples for inference queries come from independent
NUTS chains and dominate evaluation time for the nonlinear benchmarks.

Figures: `make figures` draws the benchmark box plots from the evaluation CSVs
(`scripts/plotters/make_benchmark_figures.py`); the chain and backbone targets draw their
own figures; the remaining figure scripts are in `scripts/plotters/`.

## Tests

```bash
JAX_PLATFORMS=cpu pytest pgm_paper_utils -m "not slow"
```

## Errata and changes since review

The camera-ready version fixes three implementation issues found after
submission and re-runs every affected experiment:

1. the triangular masks of the dynamic network were trainable parameters
   (now frozen);
2. the default joint (Sinkhorn) OT coupling biased conditional models (now
   chosen per conditioning mode; see the paper's appendix);
3. two density evaluations (dynamic `log_prob` / counterfactual and the guided
   conditional `log_prob`) used the wrong velocity or base term.

The tests that guard against each are in `structflow`
(`structflow/conditional_flows/tests/test_claims.py`) and in
`pgm_paper_utils/tests`.

## Citation

```bibtex
@article{telfar2026structured,
  title   = {Structured Flows for Causal Modelling and Statistical Inference},
  author  = {Telfar, Alexander and Sandfeld, Stefan},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=5cFwygAUFt}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)). The paper text and figures are © the
authors.
