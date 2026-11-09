# Automation for PGM-Flow Experiments

# Paths
PYTHON = python
SCRIPTS_DIR = ../../scripts
PLOTTERS_DIR = ../../scripts/plotters
SWEEP_SCRIPT = $(SCRIPTS_DIR)/sweep.py
EVAL_SCRIPT = $(SCRIPTS_DIR)/evaluate_sweep.py
RESULTS_BASE ?= $(STRUCTFLOW_RESULTS)/mnist
IMAGE_DIR = .

# Experiment Targets
.PHONY: all train sample eval plot

all: train sample eval plot 

train:
	mkdir -p $(RESULTS_BASE)
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE) \
		--pgm_type mnist \
		--train_steps=100000 \
		--budget_params=8000000 \
		--lr=0.0001 \
		--task causal \
		--n_pgm_seeds=1 \
		--n_learner_seeds=5

sample:
	for i in 0 1 2 3; do \
		$(PYTHON) $(PLOTTERS_DIR)/vis_image_samples.py --results_path $(RESULTS_BASE) --output_path ../mnist/mnist_samples$$i.png --seed=$$i; \
	done

eval:
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE) --output_path $(RESULTS_BASE)/evaluation_results.csv --num_samples=1000 --only_last=False
	
plot:
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path $(RESULTS_BASE)/evaluation_results.csv --output_dir $(IMAGE_DIR) --metric SWD --task causal
	
	# $(PYTHON) vis_sweep.py --results_path $(RESULTS_BASE)/evaluation_results.csv --output_dir $(IMAGE_DIR)