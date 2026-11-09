# Automation for PGM-Flow Experiments

# Paths
PYTHON = python
SWEEP_SCRIPT = ../sweep.py
EVAL_SCRIPT = ../evaluate_sweep.py
IMAGE_DIR = ../../images
RESULTS_BASE = ../../../results

# Experiment Targets
.PHONY: all mnist posteriordb random sparse scaling

all: mnist posteriordb random sparse scaling

digits:
	mkdir -p $(RESULTS_BASE)/digits
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/digits \
		--pgm_type digits \
		--train_steps=10000 \
		--budget_params=1000000 \
		--task causal \
		--n_pgm_seeds=1 \
		--n_learner_seeds=5
	$(PYTHON) vis_image_samples.py --results_path $(RESULTS_BASE)/digits --output_path $(RESULTS_BASE)/digits/digits_samples0.png --seed=0
	$(PYTHON) vis_image_samples.py --results_path $(RESULTS_BASE)/digits --output_path $(RESULTS_BASE)/digits/digits_samples1.png --seed=1
	$(PYTHON) vis_image_samples.py --results_path $(RESULTS_BASE)/digits --output_path $(RESULTS_BASE)/digits/digits_samples2.png --seed=2
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/digits --output_path $(RESULTS_BASE)/digits/evaluation_results.csv --num_samples=250
	$(PYTHON) vis_sweep.py --results_path $(RESULTS_BASE)/digits/evaluation_results.csv --output_dir $(RESULTS_BASE)/digits/

mnist:
	# mkdir -p $(RESULTS_BASE)/mnist
	# $(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/mnist \
	# 	--pgm_type mnist \
	# 	--train_steps=100000 \
	# 	--budget_params=8000000 \
	# 	--lr=0.0001 \
	# 	--task causal \
	# 	--n_pgm_seeds=1 \
	# 	--n_learner_seeds=5
	# $(PYTHON) vis_image_samples.py --results_path $(RESULTS_BASE)/mnist --output_path $(RESULTS_BASE)/mnist/mnist_samples0.png --seed=4
	# $(PYTHON) vis_image_samples.py --results_path $(RESULTS_BASE)/mnist --output_path $(RESULTS_BASE)/mnist/mnist_samples1.png --seed=5
	# $(PYTHON) vis_image_samples.py --results_path $(RESULTS_BASE)/mnist --output_path $(RESULTS_BASE)/mnist/mnist_samples2.png --seed=6
	# $(PYTHON) vis_image_samples.py --results_path $(RESULTS_BASE)/mnist --output_path $(RESULTS_BASE)/mnist/mnist_samples3.png --seed=7
	# $(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/mnist --output_path $(RESULTS_BASE)/mnist/evaluation_results.csv --num_samples=1000 --only_last=False
	$(PYTHON) vis_sweep.py --results_path $(RESULTS_BASE)/mnist/evaluation_results.csv --output_dir $(RESULTS_BASE)/mnist/

	# $(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/mnist/evaluation_results.csv --output_dir $(RESULTS_BASE)/mnist --metric SWD


posteriordb_eight_schools:
	mkdir -p $(RESULTS_BASE)/posteriordb_eight_schools
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/posteriordb_eight_schools --pgm_type eight_schools --train_steps=10000 --budget_params=1000000 --task=inference --n_pgm_seeds=1 --n_learner_seeds=5
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/posteriordb_eight_schools --output_path $(RESULTS_BASE)/posteriordb_eight_schools/evaluation_results.csv --num_samples=10000
	$(PYTHON) vis_sweep.py --results_path $(RESULTS_BASE)/posteriordb_eight_schools/evaluation_results.csv --output_dir $(RESULTS_BASE)/posteriordb_eight_schools/

scaling:
	mkdir -p $(RESULTS_BASE)/scaling
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/scaling \
		--pgm_type chain --n_vars 64 --train_steps=10000 \
		--budget_params=5000000 n_pgm_seeds=3 --n_learner_seeds=3
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/scaling --output_path $(RESULTS_BASE)/scaling/evaluation_results.csv
	$(PYTHON) vis_chain_scaling.py --results_path $(RESULTS_BASE)/scaling/evaluation_results.csv --output_dir $(IMAGE_DIR)

random:
	mkdir -p $(RESULTS_BASE)/random
	
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/random --pgm_type random --train_steps=25000 --budget_params=5000000 --n_vars 4 --n_pgm_seeds=3 --n_learner_seeds=3 --task=inference
	$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/random --pgm_type random --train_steps=25000 --budget_params=5000000 --n_vars 4 --n_pgm_seeds=3 --n_learner_seeds=3 --task=causal
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/random --output_path $(RESULTS_BASE)/random/evaluation_results.csv --num_samples=500
	
	# for s in 0 1 2; do \
	# 	for t in inference causal; do \
	# 		$(PYTHON) vis_sweep.py --results_path $(RESULTS_BASE)/random/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear/ --pgm_seed=$$s --task=$$t --metric=SWD; \
	# 	done; \
	# done

	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/random/evaluation_results.csv --output_dir $(RESULTS_BASE)/random --metric SWD --task=causal
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/random/evaluation_results.csv --output_dir $(RESULTS_BASE)/random --metric SWD --task=inference
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/random/evaluation_results.csv --output_dir $(RESULTS_BASE)/random --metric SWD --avg_queries=True --task=inference
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/random/evaluation_results.csv --output_dir $(RESULTS_BASE)/random --metric Counterfactual_SWD --task=causal  --avg_queries=True
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/random/evaluation_results.csv --output_dir $(RESULTS_BASE)/random --metric SWD --avg_queries=True

	$(PYTHON) plot_joint_task_comparison.py --results_path $(RESULTS_BASE)/random/evaluation_results.csv --output_dir $(RESULTS_BASE)/random

nonlinear:
	# mkdir -p $(RESULTS_BASE)/nonlinear
	# $(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/nonlinear --pgm_type nonlinear --train_steps=50000 --budget_params=5000000 --n_vars 4 --n_pgm_seeds=3 --n_learner_seeds=3 --task=inference
	# $(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/nonlinear --pgm_type nonlinear --train_steps=50000 --budget_params=5000000 --n_vars 4 --n_pgm_seeds=3 --n_learner_seeds=3 --task=causal
	# $(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/nonlinear --output_path $(RESULTS_BASE)/nonlinear/evaluation_results-kv.csv --num_samples=500

	# for s in 0 1 2; do \
	# 	for t in inference causal; do \
	# 		$(PYTHON) vis_sweep.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear/ --pgm_seed=$$s --task=$$t --metric=SWD; \
	# 	done; \
	# done

	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear --metric SWD --task=causal
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear --metric SWD --task=inference
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear --metric SWD --avg_queries=True --task=inference  --conditional_only=True
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear --metric Counterfactual_SWD --task=causal  --avg_queries=True
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear --metric SWD --avg_queries=True

	$(PYTHON) plot_joint_task_comparison.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear

nonlinearnoise:
	# mkdir -p $(RESULTS_BASE)/nonlinearnoise
	# $(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/nonlinearnoise --pgm_type nonlinearnoise --train_steps=50000 --budget_params=5000000 --n_vars 4 --n_pgm_seeds=3 --n_learner_seeds=3 --task=inference
	# $(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/nonlinearnoise --pgm_type nonlinearnoise --train_steps=50000 --budget_params=5000000 --n_vars 4 --n_pgm_seeds=3 --n_learner_seeds=3 --task=causal --use_reversed_order=True
	# $(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/nonlinearnoise --output_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --num_samples=500

	# for s in 0 1 2; do \
	# 	for t in inference causal; do \
	# 		$(PYTHON) vis_sweep.py --results_path $(RESULTS_BASE)/nonlinear/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinear/ --pgm_seed=$$s --task=$$t --metric=SWD; \
	# 	done; \
	# done

	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric SWD --task=causal
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric SWD --task=causal --pgm_seed=0
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric SWD --task=inference
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric Counterfactual_SWD --task=causal
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric SWD --task=inference --pgm_seed=0
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric SWD --avg_queries=True --task=inference  --conditional_only=True
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric SWD --avg_queries=True --task=causal  --conditional_only=True
	$(PYTHON) plot_final_box.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise --metric Counterfactual_SWD --avg_queries=True --task=causal  --conditional_only=True

	$(PYTHON) plot_joint_task_comparison.py --results_path $(RESULTS_BASE)/nonlinearnoise/evaluation_results.csv --output_dir $(RESULTS_BASE)/nonlinearnoise

compare:
	python plot_pgm_type_comparison.py ~/Documents/results/random/evaluation_results.csv ~/Documents/results/nonlinearnoise/evaluation_results.csv ../../images/experimental_figures/ --metric=SWD --y_max=0.2

	python plot_pgm_type_comparison.py ~/Documents/results/random/evaluation_results.csv ~/Documents/results/nonlinearnoise/evaluation_results.csv ../../images/experimental_figures/ --metric=Counterfactual_SWD --y_max=0.05

sparse:
	mkdir -p $(RESULTS_BASE)/sparse
# 	for s in 0.1 0.3 0.5 0.7 0.9; do \
# 		echo "Running sweep for sparsity $$s..."; \
# 		$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/sparse \
# 			--pgm_type random \
# 			--train_steps=1000 \
# 			--budget_params=1000000 \
# 			--n_vars 16 \
# 			--n_pgm_seeds=5 \
# 			--n_learner_seeds=3 \
# 			--task inference \
# 			--sparsity $$s; \
# 	done
	for s in 0.3 0.7; do \
		echo "Running sweep for sparsity $$s..."; \
		$(PYTHON) $(SWEEP_SCRIPT) $(RESULTS_BASE)/sparse \
			--pgm_type random \
			--train_steps=100 \
			--budget_params=20000 \
			--n_vars 4 \
			--n_pgm_seeds=2 \
			--n_learner_seeds=2 \
			--sparsity $$s \
			--task inference; \
	done
	$(PYTHON) $(EVAL_SCRIPT) $(RESULTS_BASE)/sparse \
		--output_path $(RESULTS_BASE)/sparse/evaluation_results.csv \
		--num_samples 1000 \
		--n_steps 50
	$(PYTHON) vis_sparsity_scaling.py --results_path $(RESULTS_BASE)/sparse/evaluation_results.csv --output_dir $(IMAGE_DIR)


speed:
	$(PYTHON) ../benchmark_static_vs_dynamic.py
	$(PYTHON) vis_speedup.py --csv_path ../benchmark_results.csv