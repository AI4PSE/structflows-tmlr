"""Smoke-test and visualize the Colored-MNIST DGP (ground truth only, no flows).

Checks:
  1. Sampling works (joint, all supported conditionals, interventional) and
     shapes match variable_dims under vmap.
  2. The exact digit posterior p(digit | color) matches empirical frequencies.
  3. The compiler produces the expected causal and inference structures, and
     the evaluation harness would evaluate p(image | color) in inference mode.
  4. Renders the headline ground-truth figure: p(image | do(color)) keeps the
     digit marginal flat, p(image | color) concentrates it (back-door via digit).

Run from the repository root:
    python scripts/plotters/vis_colored_mnist_dgp.py
"""
from pgm_paper_utils.paths import IMAGES
import os
import sys

os.environ["JAX_PLATFORM_NAME"] = "cpu"


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from structflow.pgms.colored_mnist import ColoredMNISTPGM, DIGIT_PALETTE
from structflow.pgms.pgm_compiler import PGMCompiler
from pgm_paper_utils.utils import get_test_queries, format_dist_str

N = 2000
COND_DIGIT = 5  # showcase color: the palette color of digit 5 (blue)


def check_sampling(pgm, key):
    dims = pgm.variable_dims
    joint = pgm.b_sample(key, N)
    for k, d in dims.items():
        assert joint[k].shape == (N, d), (k, joint[k].shape)

    for conditions in [{'digit': jnp.array(3)},
                       {'color': jnp.array(DIGIT_PALETTE[COND_DIGIT])},
                       {'digit': jnp.array(3), 'color': jnp.array(DIGIT_PALETTE[0])}]:
        s = pgm.b_sample(key, 32, conditions=conditions)
        assert s['image'].shape == (32, dims['image'])

    s = pgm.b_sample_interventional(
        key, 32, interventions={'color': jnp.array(DIGIT_PALETTE[COND_DIGIT])})
    assert s['image'].shape == (32, dims['image'])
    print("[1] sampling: all shapes OK")
    return joint


def check_posterior(pgm, key):
    c = jnp.array(DIGIT_PALETTE[COND_DIGIT])
    logits = pgm._digit_posterior_logits(c, training=True)
    post = jax.nn.softmax(logits)
    cond = pgm.b_sample(key, N, conditions={'color': c}, targets=frozenset({'digit'}))
    emp = cond['digit'].mean(axis=0)
    err = float(jnp.max(jnp.abs(post - emp)))
    print(f"[2] p(digit | color=palette[{COND_DIGIT}]): analytic peak "
          f"{float(post[COND_DIGIT]):.3f} at digit {int(jnp.argmax(post))}, "
          f"max |analytic - empirical| = {err:.3f}")
    assert int(jnp.argmax(post)) == COND_DIGIT
    assert err < 0.05
    return post


def check_compilation(pgm):
    compiler = PGMCompiler(pgm.get_pgm_structure(), dims=pgm.variable_dims)
    causal = compiler.compile_causal()
    assert causal.block_order == ['digit', 'color', 'image']

    targets, conditions = pgm.get_inference_targets_and_conditions()
    inference = compiler.compile_inference(targets=targets, conditions=conditions)
    print(f"[3] causal order:    {causal.block_order}")
    print(f"    inference order: {inference.block_order}")
    print(f"    inference deps:  {inference.dependencies}")

    for task, structure in [('causal', causal), ('inference', inference)]:
        queries = get_test_queries(pgm, structure)
        strs = [format_dist_str(t, c) for (t, c, _) in queries] \
            if queries and len(queries[0]) == 3 else [str(q) for q in queries]
        print(f"    {task} task queries: {strs}")
    return causal, inference


def render_figure(pgm, key):
    c = jnp.array(DIGIT_PALETTE[COND_DIGIT])
    k1, k2 = jax.random.split(key)
    do_samples = pgm.b_sample_interventional(k1, N, interventions={'color': c})
    cond_samples = pgm.b_sample(k2, N, conditions={'color': c})

    do_hist = np.array(do_samples['digit'].mean(axis=0))
    cond_hist = np.array(cond_samples['digit'].mean(axis=0))

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.35)

    # Image grids: one row of samples per regime.
    for row, (samples, label) in enumerate(
            [(do_samples, r"$p(\mathrm{image} \mid \mathrm{do}(\mathrm{color}))$"),
             (cond_samples, r"$p(\mathrm{image} \mid \mathrm{color})$")]):
        ax = fig.add_subplot(gs[row, 0])
        imgs = np.array(samples['image'][:10]).reshape(-1, 28, 28, 3)
        ax.imshow(np.clip(imgs.transpose(1, 0, 2, 3).reshape(28, -1, 3), 0, 1))
        ax.set_title(label, fontsize=12)
        ax.axis('off')

    # Digit marginals under the two regimes.
    x = np.arange(10)
    ax = fig.add_subplot(gs[:, 1])
    ax.bar(x - 0.2, do_hist, width=0.4, label=r"under $\mathrm{do}(\mathrm{color})$")
    ax.bar(x + 0.2, cond_hist, width=0.4, label="under conditioning")
    ax.set_xticks(x)
    ax.set_xlabel("digit")
    ax.set_ylabel("marginal probability")
    ax.set_title("Digit marginal: intervening vs conditioning\n"
                 f"(color = palette color of digit {COND_DIGIT})", fontsize=12)
    ax.legend()

    out = f"{IMAGES}/colored_mnist_dgp.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"[4] do-vs-condition ground truth: do-marginal max {do_hist.max():.2f} "
          f"(flat), cond-marginal peak {cond_hist.max():.2f} at digit "
          f"{int(cond_hist.argmax())}. Saved {out}")


def main():
    pgm = ColoredMNISTPGM()
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    check_sampling(pgm, k1)
    check_posterior(pgm, k2)
    check_compilation(pgm)
    render_figure(pgm, k3)
    print("All Colored-MNIST DGP checks passed.")


if __name__ == "__main__":
    main()
