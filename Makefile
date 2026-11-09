# StructFlow experiment recipes. Run from the repo root (tmux on the cluster):
#   make nonlinear16                 # train + NUTS-evaluate the n=16 benchmark
#   make nonlinear16 NLRN=3          # widen the seed grid
#   make PY=python3 nonlinear8       # PY defaults to `python` (active env)
# Every sweep skips finished run directories, so all targets are resume-safe:
# re-issue the same `make` line to continue after an interruption.

PY      ?= python
RESULTS ?= $(or $(STRUCTFLOW_RESULTS),$(CURDIR)/results)
# Noise-data coupling per conditioning mode (paper Appendix "Noise--Data Coupling"):
# static independent, dynamic per-block C2OT (minibatch Knothe-Rosenblatt), guided
# exact OT. Learner seeds of a (mode, PGM seed) group train together (vmap).
COUPLING ?= --coupling static=false,dynamic=c2ot,guided=hungarian --c2ot_target_ratio=0.05 --seed_batch=True
SWEEP    = $(PY) scripts/sweep.py $(COUPLING)
EVAL     = $(PY) scripts/evaluate_sweep.py

.PHONY: linear4 nonlinear4 linear8 nonlinear8 linear16 nonlinear16 \
        mnist colored-mnist colored-mnist-conv colored-mnist-rho1 \
        chain-rw nuts-validation ordering-ablation

# ------------------------------------------------------------------ benchmarks
# Canonical recipe (matches the published n=4 / n=8 grids): random DAGs,
# sparsity 0.3, 5M-parameter budget, 100k steps, both tasks, then NUTS eval.
BENCH = linear4 nonlinear4 linear8 nonlinear8 linear16 nonlinear16

linear4 nonlinear4:   N  = 4
linear4 nonlinear4:   ND = 4
linear8 nonlinear8:   N  = 8
linear8 nonlinear8:   ND = 8
linear16 nonlinear16: N  = 16
linear16 nonlinear16: ND = 8
linear4 linear8 linear16:          MECH = linear
nonlinear4 nonlinear8 nonlinear16: MECH = universal
# Published grids: 5 modes x 5 PGM seeds x 5 learner seeds.
linear4 nonlinear4 linear8 nonlinear8: MODES = carefl,static,dynamic,guided,universal
linear4 nonlinear4 linear8 nonlinear8: NPGM  = 5
linear4 nonlinear4 linear8 nonlinear8: NLRN  = 5
# n=16: same 5x5 seed grid as the published benchmarks (override, e.g.
# NLRN=2, to time-budget); modes default to four -- add `universal` if time allows.
linear16 nonlinear16: MODES ?= carefl,static,dynamic,guided
linear16 nonlinear16: NPGM  ?= 5
linear16 nonlinear16: NLRN  ?= 5

$(BENCH):
	for T in inference causal; do \
	  $(SWEEP) $(RESULTS)/$@ --pgm_type random --mechanism_type $(MECH) \
	    --n_nodes=$(N) --node_dim=$(ND) --sparsity=0.3 \
	    --budget_params=5000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes $(MODES) \
	    --n_pgm_seeds=$(NPGM) --n_learner_seeds=$(NLRN) || exit 1; \
	done
	$(EVAL) $(RESULTS)/$@

# ----------------------------------------------------------------------- mnist
mnist:
	for T in inference causal; do \
	  $(SWEEP) $(RESULTS)/mnist --pgm_type mnist \
	    --budget_params=8000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes carefl,static,dynamic,guided,universal \
	    --n_pgm_seeds=1 --n_learner_seeds=5 || exit 1; \
	done
	$(EVAL) $(RESULTS)/mnist

# --------------------------------------------------------------- colored-MNIST
# MLP fleets behind the quantitative numbers: static/dynamic at 8 seeds,
# baselines at 3.
colored-mnist:
	for T in causal inference; do \
	  $(SWEEP) $(RESULTS)/colored_mnist/$$T --pgm_type colored_mnist \
	    --budget_params=8000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes static,dynamic --n_pgm_seeds=1 --n_learner_seeds=8 \
	    || exit 1; \
	  $(SWEEP) $(RESULTS)/colored_mnist/$$T --pgm_type colored_mnist \
	    --budget_params=8000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes carefl,guided,universal --n_pgm_seeds=1 \
	    --n_learner_seeds=3 || exit 1; \
	  $(EVAL) $(RESULTS)/colored_mnist/$$T || exit 1; \
	done

# Conv (U-Net) models behind the sample figures; 300k steps, static only.
colored-mnist-conv:
	for T in causal inference; do \
	  $(SWEEP) $(RESULTS)/colored_mnist/conv_$$T --pgm_type colored_mnist \
	    --budget_params=8000000 --train_steps=300000 --lr=0.0001 \
	    --task $$T --modes static --use_conv \
	    --n_pgm_seeds=1 --n_learner_seeds=1 || exit 1; \
	done

# rho = 1 variant (zero-shot counterfactual appendix).
colored-mnist-rho1:
	for T in causal inference; do \
	  $(SWEEP) $(RESULTS)/colored_mnist_rho1/conv_$$T --pgm_type colored_mnist \
	    --color_correlation=1.0 \
	    --budget_params=8000000 --train_steps=300000 --lr=0.0001 \
	    --task $$T --modes static --use_conv \
	    --n_pgm_seeds=1 --n_learner_seeds=1 || exit 1; \
	done

# ---------------------------------------------------------------------- chains
# Unit-root random-walk chain grid (Section "Price of Parallelism"):
# train 2 modes x 8 seeds x 5 depths, then the 25-draw solver grid + OOD evals.
chain-rw:
	for N in 2 4 8 16 32; do \
	  $(SWEEP) $(RESULTS)/chain_rw/n$$N --pgm_type chain --mechanism_type linear \
	    --weight_scale=1.0 --noise_scale=0.2 --root_scale=1.0 \
	    --n_nodes=$$N --node_dim=1 \
	    --budget_params=500000 --train_steps=30000 --lr=0.0001 \
	    --task inference --modes static,dynamic \
	    --n_pgm_seeds=1 --n_learner_seeds=8 || exit 1; \
	done
	for N in 2 4 8 16 32; do \
	  CHAIN_BASE=$(RESULTS)/chain_rw CHAIN_NS=$$N CHAIN_NCOND=25 \
	    CHAIN_TAG=full25-$$N \
	    $(PY) scripts/ablations/chain_stiffness_analysis.py || exit 1; \
	done
	CHAIN_BASE=$(RESULTS)/chain_rw $(PY) scripts/ablations/chain_ood_measure.py

# ------------------------------------------------------------------- ablations
nuts-validation:
	$(PY) scripts/ablations/nuts_validation.py

ordering-ablation:
	$(PY) scripts/ablations/ordering_ablation.py
