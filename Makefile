# StructFlow experiment recipes. Run from the repo root, e.g.
#   make nonlinear4                  # train + NUTS-evaluate the n=4 nonlinear benchmark
#   make nonlinear4 NLRN=2           # fewer learner seeds per PGM
#   make PY=python3 nonlinear8       # PY defaults to `python` (active env)
#   make figures                     # benchmark figures from the evaluation CSVs
# Every sweep skips finished run directories, so all targets are resume-safe:
# re-issue the same `make` line to continue after an interruption.
#
# Results go to $(RESULTS)/<experiment>_kr/ (``kr``: trained with the per-mode
# couplings below); the figure scripts read the same directories.

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
        chain-rw nuts-validation ordering-ablation backbone-ablation figures

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
	  $(SWEEP) $(RESULTS)/$@_kr --pgm_type random --mechanism_type $(MECH) \
	    --n_nodes=$(N) --node_dim=$(ND) --sparsity=0.3 \
	    --budget_params=5000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes $(MODES) \
	    --n_pgm_seeds=$(NPGM) --n_learner_seeds=$(NLRN) || exit 1; \
	done
	$(EVAL) $(RESULTS)/$@_kr

# ----------------------------------------------------------------------- mnist
mnist:
	for T in inference causal; do \
	  $(SWEEP) $(RESULTS)/mnist_kr --pgm_type mnist \
	    --budget_params=8000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes carefl,static,dynamic,guided,universal \
	    --n_pgm_seeds=1 --n_learner_seeds=5 || exit 1; \
	done
	$(EVAL) $(RESULTS)/mnist_kr

# --------------------------------------------------------------- colored-MNIST
# MLP fleets behind the quantitative numbers: static/dynamic at 8 seeds,
# baselines at 3.
colored-mnist:
	for T in causal inference; do \
	  $(SWEEP) $(RESULTS)/colored_mnist_kr/$$T --pgm_type colored_mnist \
	    --budget_params=8000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes static,dynamic --n_pgm_seeds=1 --n_learner_seeds=8 \
	    || exit 1; \
	  $(SWEEP) $(RESULTS)/colored_mnist_kr/$$T --pgm_type colored_mnist \
	    --budget_params=8000000 --train_steps=100000 --lr=0.0001 \
	    --task $$T --modes carefl,guided,universal --n_pgm_seeds=1 \
	    --n_learner_seeds=3 || exit 1; \
	  $(EVAL) $(RESULTS)/colored_mnist_kr/$$T || exit 1; \
	done

# Conv (U-Net) models behind the sample figures; 300k steps, static only.
colored-mnist-conv:
	for T in causal inference; do \
	  $(SWEEP) $(RESULTS)/colored_mnist_kr/conv_$$T --pgm_type colored_mnist \
	    --budget_params=8000000 --train_steps=300000 --lr=0.0001 \
	    --task $$T --modes static --use_conv \
	    --n_pgm_seeds=1 --n_learner_seeds=1 || exit 1; \
	done

# rho = 1 variant (zero-shot counterfactual appendix).
colored-mnist-rho1:
	for T in causal inference; do \
	  $(SWEEP) $(RESULTS)/colored_mnist_rho1_kr/conv_$$T --pgm_type colored_mnist \
	    --color_correlation=1.0 \
	    --budget_params=8000000 --train_steps=300000 --lr=0.0001 \
	    --task $$T --modes static --use_conv \
	    --n_pgm_seeds=1 --n_learner_seeds=1 || exit 1; \
	done

# ---------------------------------------------------------------------- chains
# Unit-root random-walk chain grid (Section "Price of Parallelism"): 3 variants x 5 depths x 8
# seeds (static/no coupling, dynamic/per-block C2OT, dynamic/independent), the 25-draw solver
# grid per variant, the population limit of the minibatch coupling, and the figure.
CHAIN_VARIANTS = static_none:static:static=false dynamic_c2ot:dynamic:dynamic=c2ot \
                 dynamic_none:dynamic:dynamic=false
chain-rw:
	for V in $(CHAIN_VARIANTS); do \
	  NAME=$${V%%:*}; R=$${V#*:}; MODE=$${R%%:*}; CPL=$${R#*:}; \
	  for N in 2 4 8 16 32; do \
	    $(PY) scripts/sweep.py $(RESULTS)/chain_rw_kr/$$NAME/n$$N --pgm_type chain --mechanism_type linear \
	      --weight_scale=1.0 --noise_scale=0.2 --root_scale=1.0 \
	      --n_nodes=$$N --node_dim=1 \
	      --budget_params=500000 --train_steps=30000 --lr=0.0001 \
	      --task inference --modes $$MODE --coupling $$CPL \
	      --c2ot_target_ratio=0.05 --seed_batch=True \
	      --n_pgm_seeds=1 --n_learner_seeds=8 || exit 1; \
	    CHAIN_BASE=$(RESULTS)/chain_rw_kr/$$NAME CHAIN_NS=$$N CHAIN_NCOND=25 \
	      CHAIN_TAG=full25-$$N \
	      $(PY) scripts/ablations/chain_stiffness_analysis.py || exit 1; \
	  done; \
	done
	CHAIN_POP_OUT=$(RESULTS)/chain_rw_kr/population.json $(PY) scripts/ablations/chain_population_check.py
	CHAIN_RW_BASE=$(RESULTS)/chain_rw_kr $(PY) scripts/plotters/plot_chain_rw.py

# ------------------------------------------------------------------- ablations
nuts-validation:
	$(PY) scripts/ablations/nuts_validation.py

ordering-ablation:
	$(PY) scripts/ablations/ordering_ablation.py

# Backbone ablation (static CNF vs per-node DDIM): needs the static causal runs of
# `make nonlinear4`; trains the DCM runs (no coupling option) into $(RESULTS)/nonlinear4,
# then sweeps network calls per sample and plots.
backbone-ablation:
	$(SWEEP) $(RESULTS)/nonlinear4 --pgm_type random --mechanism_type universal \
	  --n_nodes=4 --node_dim=4 --sparsity=0.3 \
	  --budget_params=5000000 --train_steps=100000 --lr=0.0001 \
	  --task causal --modes dcm --n_pgm_seeds=5 --n_learner_seeds=1
	NFE_BASE=$(RESULTS)/nonlinear4 NFE_STATIC_BASE=$(RESULTS)/nonlinear4_kr \
	  $(PY) scripts/ablations/backbone_nfe.py
	NFE_RESULTS=$(RESULTS)/nonlinear4_kr/backbone_nfe.json $(PY) scripts/plotters/plot_backbone_nfe.py

# Benchmark figures (4-node linear/nonlinear, 8-node inference) from the evaluation CSVs
# written by the benchmark targets above.
figures:
	$(PY) scripts/plotters/make_benchmark_figures.py --results $(RESULTS) --n8
