# Automation for PGM-Flow Experiments

# Paths
PYTHON = python
SCRIPTS_DIR = ../../scripts
PLOTTERS_DIR = ../../scripts/plotters
SWEEP_SCRIPT = $(SCRIPTS_DIR)/sweep.py
EVAL_SCRIPT = $(SCRIPTS_DIR)/evaluate_sweep.py
RESULTS_BASE ?= $(STRUCTFLOW_RESULTS)
IMAGE_BASE = .
MECHANISM_TYPE = linear

FOLDER ?= linear8
N_NODES ?= 8
N_PGM_SEEDS ?= 5
N_LEARNER_SEEDS ?= 5
NODE_DIM ?= 8
TRAIN_STEPS ?= 100000
BUDGET_PARAMS ?= 5000000

METRIC ?= KL

# Experiment Targets
.PHONY: all train eval plot

all: train eval plot

train:
	mkdir -p $(RESULTS_BASE)/$(FOLDER)
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/$(FOLDER) --pgm_type random --mechanism_type=$(MECHANISM_TYPE) --train_steps=$(TRAIN_STEPS) --budget_params=$(BUDGET_PARAMS) --n_nodes $(N_NODES) --n_pgm_seeds=$(N_PGM_SEEDS) --n_learner_seeds=$(N_LEARNER_SEEDS) --task=causal --node_dim=$(NODE_DIM)
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/$(FOLDER) --pgm_type random --mechanism_type=$(MECHANISM_TYPE) --train_steps=$(TRAIN_STEPS) --budget_params=$(BUDGET_PARAMS) --n_nodes $(N_NODES) --n_pgm_seeds=$(N_PGM_SEEDS) --n_learner_seeds=$(N_LEARNER_SEEDS) --task=inference --node_dim=$(NODE_DIM)
	
eval:
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/$(FOLDER) --output_path $(RESULTS_BASE)/$(FOLDER)/evaluation_results.csv --num_samples=500
	
plot:
	mkdir -p $(IMAGE_BASE)
	
	# causal
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path=$(RESULTS_BASE)/$(FOLDER)/evaluation_results.csv --output_dir=$(IMAGE_BASE) --metric=$(METRIC) --task=causal --conditional_only=True --avg_queries=True --modes=static,universal,carefl
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path=$(RESULTS_BASE)/$(FOLDER)/evaluation_results.csv --output_dir=$(IMAGE_BASE) --metric Counterfactual_KL --task=causal --avg_queries=True --conditional_only=True --modes=static,universal,carefl
	
	# inference
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path=$(RESULTS_BASE)/$(FOLDER)/evaluation_results.csv --output_dir=$(IMAGE_BASE) --metric=SWD --task=inference --conditional_only=True --avg_queries=True --modes=static,universal,guided,carefl

	# extras
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path=$(RESULTS_BASE)/$(FOLDER)/evaluation_results.csv --output_dir=$(IMAGE_BASE) --metric=SWD --task=inference
	$(PYTHON) $(PLOTTERS_DIR)/plot_final_box.py --results_path=$(RESULTS_BASE)/$(FOLDER)/evaluation_results.csv --output_dir=$(IMAGE_BASE)/ --metric=$(METRIC) --task=causal
	
	$(PYTHON) $(PLOTTERS_DIR)/plot_joint_task_comparison.py --results_path=$(RESULTS_BASE)/$(FOLDER)/evaluation_results.csv --output_dir=$(IMAGE_BASE)