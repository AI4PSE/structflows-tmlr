
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
