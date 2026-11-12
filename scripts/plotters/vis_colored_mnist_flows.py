"""Headline Colored-MNIST figure: intervening vs conditioning, with flows.

Loads a trained causal-aligned flow and a trained inference-aligned flow for
Colored-MNIST and renders:
  (a) model samples from p(image | do(color)) [causal flow: sample own joints,
      then counterfactual do(color) -- ancestral sampling of the mutilated flow]
      vs p(image | color) [inference flow: prefix conditioning],
  (b) the digit marginal under both regimes, model vs ground truth,
  (c) a counterfactual re-coloring strip: factual, flow do(color'), exact
      ground-truth re-tint,
  and prints the p(digit | color, image) classification accuracy of the
  inference flow on held-out data.

Usage (from repository root):
    ~/my_jax_env/bin/python scripts/plotters/vis_colored_mnist_flows.py \
        --causal_results $STRUCTFLOW_RESULTS/colored_mnist/causal \
        --inference_results $STRUCTFLOW_RESULTS/colored_mnist/inference \
        --mode static --learner_seed 0 \
        --output paper/images/colored_mnist_flows.png
"""
from pgm_paper_utils.paths import IMAGES
import os
import sys
import json

# Disable XLA GPU autotuning (same workaround as evaluate_sweep.py): the
# autotuner fails/produces bad kernels on the large vmapped image-block gemms.
# Triton GEMMs fail on the conv-U-Net contractions ("Too small divisible part of the contracting dimension").
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0 --xla_force_host_platform_device_count=1 --xla_gpu_enable_triton_gemm=false'


import fire
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from jax import random

from structflow.pgms.colored_mnist import ColoredMNISTPGM, DIGIT_PALETTE
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.dist_factory import create_distribution
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.utils import make_structure


def load_run(results_dir, mode, learner_seed):
    """Find the run dir for (mode, learner_seed), load its final checkpoint."""
    run_dir = None
    for item in sorted(os.listdir(results_dir)):
        cfg_path = os.path.join(results_dir, item, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg['mode'] == mode and cfg['learner_seed'] == learner_seed:
            run_dir = os.path.join(results_dir, item)
            config = cfg
            break
    if run_dir is None:
        raise FileNotFoundError(f"No run for mode={mode}, seed={learner_seed} in {results_dir}")

    pgm = create_pgm(config['pgm_type'], config, random.PRNGKey(config['pgm_seed']))
    structure = make_structure(pgm, config)
    model_skeleton, normaliser_skeleton, _, _ = create_learner(config, structure=structure)

    ckpts = [f for f in os.listdir(run_dir) if f.endswith(".eqx")
             and (f.startswith("checkpoint_") or f.startswith("model_"))]
    steps = [int(f.split("_")[-1].split(".")[0]) for f in ckpts]
    ckpt = ckpts[int(np.argmax(steps))]
    model, normaliser = eqx.tree_deserialise_leaves(
        os.path.join(run_dir, ckpt), (model_skeleton, normaliser_skeleton))
    config.setdefault('n_steps', 50)
    dist = create_distribution(config, model, normaliser)
    print(f"Loaded {config['mode']} (seed {learner_seed}) from {run_dir} @ {ckpt}")
    return pgm, dist, config


def b_sample_pgm(pgm, key, n, conditions=None, targets=None, training=False):
    """Batched PGM sampling with split control (base b_sample lacks `training`)."""
    keys = random.split(key, n)
    return jax.vmap(lambda k: pgm.sample(k, conditions, targets, training))(keys)


def b_sample_chunked(dist, key, n, chunk=32, **kw):
    """dist.b_sample in fixed-size chunks: a 256-wide batch of conv-U-Net ODE
    solves OOMs on a 12GB card; n must be divisible by chunk."""
    assert n % chunk == 0, (n, chunk)
    parts = [dist.b_sample(k2, chunk, **kw) for k2 in random.split(key, n // chunk)]
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *parts)


def image_strip(samples, n=10):
    imgs = np.clip(np.array(samples[:n]).reshape(-1, 28, 28, 3), 0, 1)
    return imgs.transpose(1, 0, 2, 3).reshape(28, -1, 3)


def main(causal_results, inference_results, mode='static', learner_seed=0,
         cond_digit=5, num_samples=256, n_class_eval=256, cf_strip=True,
         output=f'{IMAGES}/colored_mnist_flows.png'):
    key = random.PRNGKey(0)
    pgm, dist_causal, _ = load_run(causal_results, mode, learner_seed)
    _, dist_inf, _ = load_run(inference_results, mode, learner_seed)

    c = jnp.array(DIGIT_PALETTE[cond_digit])
    k_joint, k_cf, k_cond, k_gt_do, k_gt_cond, k_cls, k_cls_m = random.split(key, 7)
    tgt = frozenset({'digit', 'image'})

    # --- (a) model do(color) via two-stage ancestral sampling of the
    # mutilated graph: digit from its marginal block, then image with the
    # color block clamped. Equivalent to counterfactual-with-fresh-noise but
    # avoids b_counterfactual's log-prob Jacobian (OOM at image dims).
    digits = dist_causal.b_sample(k_joint, num_samples,
                                  targets=frozenset({'digit'}))['digit']
    _img_keys = random.split(k_joint, num_samples)
    imgs = jnp.concatenate([
        jax.vmap(lambda k, d: dist_causal.sample(
            k, conditions={'digit': d, 'color': c},
            targets=frozenset({'image'}))['image'])(
                _img_keys[i:i + 32], digits[i:i + 32])
        for i in range(0, num_samples, 32)])
    flow_do = {'digit': digits, 'image': imgs}

    # --- model p(. | color) from the inference-aligned flow ----------------
    flow_cond = b_sample_chunked(dist_inf, k_cond, num_samples,
                                 conditions={'color': c}, targets=tgt)

    # --- ground truth ------------------------------------------------------
    gt_do = pgm.b_sample_interventional(k_gt_do, num_samples, {'color': c}, training=False)
    gt_cond = b_sample_pgm(pgm, k_gt_cond, num_samples, conditions={'color': c})

    def digit_hist(s):
        return np.bincount(np.argmax(np.array(s['digit']), axis=-1), minlength=10) / len(s['digit'])

    # --- (c) counterfactual re-coloring strip (optional: the bundled
    # log-prob Jacobian OOMs on conv image blocks; skip until optimized) ----
    factual = b_sample_pgm(pgm, k_cf, 10)
    cf_int = {'color': jnp.broadcast_to(c, (10, 3))}
    # The PGM's dataset arrays are CPU-committed, so its sample outputs are
    # too; device_put them before any dist call or the flow ODE runs on CPU.
    dev = jax.devices()[0]
    if cf_strip:
        # One sample at a time: the inversion allocates ~2.4GB per sample on
        # the conv image block, so a 10-wide batch OOMs a 12GB card.
        factual_dev = jax.device_put(factual, dev)
        _parts = [dist_causal.b_counterfactual(
                      jax.tree_util.tree_map(lambda x: x[i:i + 1], factual_dev),
                      jax.tree_util.tree_map(lambda x: x[i:i + 1], cf_int),
                      targets=tgt, compute_log_prob=False)[0]
                  for i in range(10)]
        flow_cf = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *_parts)
        gt_cf, _ = jax.vmap(lambda d, i: pgm.counterfactual(d, i))(factual, cf_int)

    # --- p(digit | color, image) classification accuracy -------------------
    held_out = b_sample_pgm(pgm, k_cls, n_class_eval)
    def classify(k, color, image):
        s = dist_inf.sample(k, conditions={'color': color, 'image': image},
                            targets=frozenset({'digit'}))
        return jnp.argmax(s['digit'])
    preds = jax.vmap(classify)(random.split(k_cls_m, n_class_eval),
                               jax.device_put(held_out['color'], dev),
                               jax.device_put(held_out['image'], dev))
    truth = jnp.argmax(jax.device_put(held_out['digit'], dev), axis=-1)
    acc = float(jnp.mean(preds == truth))
    print(f"p(digit | color, image) accuracy [{mode}]: {acc:.3f}")

    # --- figures (two files: sample strips, and digit marginals) -----------
    # Display strips: one palette color per column, so the contrast is
    # self-explanatory (do: random digits in each color; conditioning: the
    # color-correlated digit appears in each color).
    pal = jnp.array(DIGIT_PALETTE)
    kd1, kd2, kd3 = random.split(k_joint, 3)
    disp_digits = dist_causal.b_sample(kd1, 10, targets=frozenset({'digit'}))['digit']
    do_disp = jax.vmap(lambda k, d, col: dist_causal.sample(
        k, conditions={'digit': d, 'color': col},
        targets=frozenset({'image'}))['image'])(random.split(kd2, 10), disp_digits, pal)
    cond_disp = jax.vmap(lambda k, col: dist_inf.sample(
        k, conditions={'color': col},
        targets=frozenset({'image'}))['image'])(random.split(kd3, 10), pal)

    rows = [
        (image_strip(factual['image']), "factual (joint) samples"),
        (image_strip(do_disp),
         r"$p(\mathrm{image} \mid \mathrm{do}(\mathrm{color}))$, one color per column"),
        (image_strip(cond_disp),
         r"$p(\mathrm{image} \mid \mathrm{color})$, observational"),
    ]
    if cf_strip:
        rows.append((np.concatenate([image_strip(flow_cf['image']),
                                     image_strip(gt_cf['image'])], axis=0),
                     r"counterfactual $\mathrm{do}(\mathrm{color}')$: flow (top), exact (bottom)"))

    stem = output[:-4] if output.endswith('.png') else output
    # The cf row stacks two strips (56px vs 28px): give it a double-height
    # axis so imshow's equal aspect renders every row at the same width.
    heights = [img.shape[0] // 28 for img, _ in rows]
    fig, axes = plt.subplots(len(rows), 1,
                             figsize=(3.9, 0.52 * sum(heights) + 0.3),  # printed at 0.6\\columnwidth
                             gridspec_kw={'hspace': 0.12, 'height_ratios': heights})
    for ax, (img, title) in zip(np.atleast_1d(axes), rows):
        ax.imshow(img)
        ax.set_title(title, fontsize=7.5, pad=2.5)
        ax.axis('off')
    fig.savefig(f"{stem}_strips.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved {stem}_strips.png")

    x = np.arange(10)
    fig2, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x - 0.3, digit_hist(flow_do), width=0.2, label=f"flow, $\\mathrm{{do}}(\\mathrm{{color}}{{=}}\\mathrm{{palette}}_{{{cond_digit}}})$")
    ax.bar(x - 0.1, digit_hist(gt_do), width=0.2, label=f"truth, $\\mathrm{{do}}(\\mathrm{{color}}{{=}}\\mathrm{{palette}}_{{{cond_digit}}})$")
    ax.bar(x + 0.1, digit_hist(flow_cond), width=0.2, label=f"flow, $\\mid \\mathrm{{color}}{{=}}\\mathrm{{palette}}_{{{cond_digit}}}$")
    ax.bar(x + 0.3, digit_hist(gt_cond), width=0.2, label=f"truth, $\\mid \\mathrm{{color}}{{=}}\\mathrm{{palette}}_{{{cond_digit}}}$")
    ax.set_xticks(x)
    ax.set_xlabel("digit")
    ax.set_ylabel("marginal probability")
    ax.set_title("Digit marginal: intervene vs observe", fontsize=11)
    ax.legend(fontsize=9)
    fig2.savefig(f"{stem}_marginals.png", dpi=200, bbox_inches='tight')
    print(f"Saved {stem}_marginals.png")

    # --- 10x10 heatmaps: p(digit | color=palette_j) for every j ------------
    def hist_of(onehots):
        return np.bincount(np.argmax(np.array(onehots), -1), minlength=10) / len(onehots)

    n_h = 128
    mats = {k2: np.zeros((10, 10)) for k2 in ['flow_do', 'truth_do', 'flow_cond']}
    for j in range(10):
        cj = pal[j]
        kj = random.fold_in(key, 100 + j)
        kh1, kh2, kh3 = random.split(kj, 3)
        mats['flow_do'][j] = hist_of(dist_causal.b_sample(
            kh1, n_h, targets=frozenset({'digit'}))['digit'])
        mats['truth_do'][j] = hist_of(pgm.b_sample_interventional(
            kh2, n_h, {'color': cj}, targets=frozenset({'digit'}), training=False)['digit'])
        mats['flow_cond'][j] = hist_of(b_sample_chunked(
            dist_inf, kh3, n_h, conditions={'color': cj},
            targets=frozenset({'digit', 'image'}))['digit'])
    truth_cond = np.stack([np.array(jax.nn.softmax(
        pgm._digit_posterior_logits(pal[j], training=False))) for j in range(10)])

    panels = [
        (mats['flow_do'], r"flow: do(color $j$)"),
        (mats['truth_do'], r"truth: do(color $j$)"),
        (mats['flow_cond'], r"flow: given color $j$"),
        (truth_cond, r"truth: given color $j$"),
    ]
    # printed at \\textwidth (6.5in): font sizes are the sizes on the page
    fig3, axes3 = plt.subplots(1, 4, figsize=(6.5, 1.85), sharey=True)
    for ax3, (m, title) in zip(axes3, panels):
        im = ax3.imshow(m, vmin=0.0, vmax=0.85, cmap='viridis')
        ax3.set_title(title, fontsize=8, pad=3)
        ax3.set_xlabel("digit $d$", fontsize=8, labelpad=1)
        ax3.set_xticks(range(10)); ax3.set_yticks(range(10))
        ax3.tick_params(labelsize=6.5, length=2, pad=1.5)
    axes3[0].set_ylabel(r"color $j$", fontsize=8, labelpad=1)
    cb = fig3.colorbar(im, ax=axes3, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=6.5)
    fig3.savefig(f"{stem}_heatmaps.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved {stem}_heatmaps.png")
    # numbers quoted in the text: diagonal (correlated-digit) mass per conditioned color
    for name, m in [('flow_cond', mats['flow_cond']), ('truth_cond', truth_cond),
                    ('flow_do', mats['flow_do']), ('truth_do', mats['truth_do'])]:
        print(f"diag {name}: " + " ".join(f"{v:.2f}" for v in np.diag(m)))


if __name__ == "__main__":
    fire.Fire(main)
