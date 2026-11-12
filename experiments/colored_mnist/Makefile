# Automation for Colored-MNIST Experiments
#
# Graph: digit -> color, (digit, color) -> image, with confounded coloring
# (rho = 0.8). Causal task: p(image | do(color)) etc. Inference task:
# p(digit, image | color), incl. the observational p(image | color) that
# differs from do(color) through the digit back-door.

# Paths
PYTHON = python
SCRIPTS_DIR = ../../scripts
PLOTTERS_DIR = ../../scripts/plotters
SWEEP_SCRIPT = $(SCRIPTS_DIR)/sweep.py
EVAL_SCRIPT = $(SCRIPTS_DIR)/evaluate_sweep.py
RESULTS_BASE ?= $(STRUCTFLOW_RESULTS)/colored_mnist
IMAGE_DIR = .

# Experiment Targets
.PHONY: all train_causal train_inference train_conv eval plot dgp

# Conv MVP: static mode only, U-Net image blocks, longer training.
# (Do NOT set xla_gpu_autotune_level=0 for training - 10x slowdown on convs.)
train_conv:
	mkdir -p $(RESULTS_BASE)/conv_causal $(RESULTS_BASE)/conv_inference
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/conv_causal \
		--pgm_type colored_mnist \
		--train_steps=300000 \
		--budget_params=8000000 \
		--lr=0.0001 \
		--task causal \
		--modes static \
		--use_conv \
		--n_pgm_seeds=1 \
		--n_learner_seeds=2
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/conv_inference \
		--pgm_type colored_mnist \
		--train_steps=300000 \
		--budget_params=8000000 \
		--lr=0.0001 \
		--task inference \
		--modes static \
		--use_conv \
		--n_pgm_seeds=1 \
		--n_learner_seeds=2

all: train_causal train_inference eval plot

dgp:
	$(PYTHON) $(PLOTTERS_DIR)/vis_colored_mnist_dgp.py

train_causal:
	mkdir -p $(RESULTS_BASE)/causal
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/causal \
		--pgm_type colored_mnist \
		--train_steps=100000 \
		--budget_params=8000000 \
		--lr=0.0001 \
		--task causal \
		--n_pgm_seeds=1 \
		--n_learner_seeds=3

train_inference:
	mkdir -p $(RESULTS_BASE)/inference
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/inference \
		--pgm_type colored_mnist \
		--train_steps=100000 \
		--budget_params=8000000 \
		--lr=0.0001 \
		--task inference \
		--n_pgm_seeds=1 \
		--n_learner_seeds=3

eval:
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/causal --output_path $(RESULTS_BASE)/causal/evaluation_results.csv --num_samples=1000 --only_last=False
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/inference --output_path $(RESULTS_BASE)/inference/evaluation_results.csv --num_samples=1000 --only_last=False

plot:
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path $(RESULTS_BASE)/causal/evaluation_results.csv --output_dir $(IMAGE_DIR) --metric SWD --task causal
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path $(RESULTS_BASE)/inference/evaluation_results.csv --output_dir $(IMAGE_DIR) --metric SWD --task inference
