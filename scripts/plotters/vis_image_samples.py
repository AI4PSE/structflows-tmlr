import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import json
import equinox as eqx
import fire
from jax import random

# Ensure CPU usage
# jax.config.update('jax_platform_name', 'cpu')

import sys
sys.path.append(os.path.abspath(".."))

from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.dist_factory import create_distribution

def load(path):
    config_path = os.path.join(path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 2. Setup PGM and Structure
    key = random.PRNGKey(0)
    pgm = create_pgm(config['pgm_type'], config, key)
    
    from structflow.pgms.structure import BlockTriangularStructure
    s_dict = config['structure']
    structure = BlockTriangularStructure(
        dependencies=s_dict['dependencies'],
        block_order=s_dict['block_order'],
        dims=s_dict['dims']
    )
    
    # 3. Initialize Leaner Skeleton and Load Weights
    # For MNIST, we usually have a normaliser
    model_skeleton, normaliser_skeleton, _, _ = create_learner(config, structure=structure)
    # find model_{step_num}.eqx in run_dir
    trained_model_path = [f for f in os.listdir(path) if f.startswith("model_")][0]
    ckpt_path = os.path.join(path, trained_model_path)
    model, normaliser = eqx.tree_deserialise_leaves(ckpt_path, (model_skeleton, normaliser_skeleton))
    
    # 4. Wrap in Distribution
    dist = create_distribution(config, model, normaliser)
    return dist, config, pgm

def plot_mnist_samples(results_path: str, output_path: str = "../../images/mnist_samples.png", seed=0):
    dists = []
    configs = []
    # Pick one run_dir for each mode (repeatable)
    mode_to_dir = {}
    for run_id in sorted(os.listdir(results_path)):
    # for run_id in reversed(sorted(os.listdir(results_path))):

        run_dir = os.path.join(results_path, run_id)
        if not os.path.isdir(run_dir):
            continue
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            mode = config.get('mode')
            if mode and mode not in mode_to_dir:
                mode_to_dir[mode] = run_dir

    for mode in sorted(mode_to_dir.keys()):
        d, c, pgm = load(mode_to_dir[mode])
        configs.append(c)
        dists.append(d)
    
    # 5. Sampling Grid
    img_dim = pgm.variable_dims['image']
    
    fig, axes = plt.subplots(len(dists), 10, figsize=(20, 8))
    sample_key = random.PRNGKey(seed)
    
    all_labels = jnp.arange(10)
    all_labels_oh = jax.nn.one_hot(all_labels, 10)
    
    for i, (d, c) in enumerate(zip(dists, configs)):
        # Prepare keys for all 10 labels for this distribution
        def get_key(label):
            label_key = random.fold_in(sample_key, label)
            # Use len(dists) instead of fixed 4 to avoid out-of-bounds
            return random.split(label_key, len(dists))[i]
        
        batch_keys = jax.vmap(get_key)(all_labels)
        
        # Vectorized sampling over labels
        # JIT the vmapped sampling function for this distribution
        @eqx.filter_jit
        def get_batch_samples(keys, labels_oh):
            return jax.vmap(lambda k, l: d.sample(k, conditions={'label': l}))(keys, labels_oh)
        
        samples = get_batch_samples(batch_keys, all_labels_oh)
        k = int(jnp.sqrt(img_dim))
        images = samples['image'].reshape(-1, k, k)
        
        for label in range(10):
            img = images[label]
            ax = axes[i, label]
            # ax.imshow(jnp.clip(img, 0, 255), cmap='gray', vmin=0, vmax=255)
            ax.imshow(img, cmap='gray')
            
            if i == 0:
                ax.set_title(f"Label {label}")
            
            if label == 0:
                # Keep ylabel visible, hide ticks and spines
                ax.set_ylabel(f"{c['mode'] if c['mode'] != 'joint' else 'guided'}", rotation=0, labelpad=40, verticalalignment='center')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved samples plot to {output_path}")

if __name__ == "__main__":
    fire.Fire(plot_mnist_samples)
