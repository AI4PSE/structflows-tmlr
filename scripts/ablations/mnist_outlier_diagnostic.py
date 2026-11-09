"""MNIST dynamic-conditioning outlier diagnostic.

Tests the conjecture (Section 4.8 / Future Work) that the elevated SWD of dynamic
conditioning on p(image|label) is driven by a small number of divergent pixels
rather than a perceptual failure. We:

  1. load one trained model per mode (carefl/static/dynamic/guided),
  2. draw conditional samples p(image|label) and the matching real test images,
  3. report the normalized SWD as-is,
  4. quantify pixel-value tails (fraction of pixels far outside the valid [0,1] range),
  5. recompute SWD after clipping samples to [0,1] -- if the dynamic gap collapses,
     the inflation is attributable to a few out-of-range pixels.

Run with the project jax env, e.g.
    ~/my_jax_env/bin/python scripts/ablations/mnist_outlier_diagnostic.py
"""
from pgm_paper_utils.paths import RESULTS
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0 --xla_force_host_platform_device_count=1'

import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.learner_factory import create_learner
from pgm_paper_utils.dist_factory import create_distribution
from structflow.bayes.metrics import compute_swd_jax

RESULTS = f"{RESULTS}/mnist"
N_PER_LABEL = 100
LABELS = list(range(10))


def load(path):
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    key = random.PRNGKey(0)
    pgm = create_pgm(config['pgm_type'], config, key)
    from structflow.pgms.structure import BlockTriangularStructure
    s = config['structure']
    structure = BlockTriangularStructure(
        dependencies=s['dependencies'], block_order=s['block_order'], dims=s['dims'])
    model_skeleton, normaliser_skeleton, _, _ = create_learner(config, structure=structure)
    ckpt = [f for f in os.listdir(path) if f.startswith("model_")][0]
    model, normaliser = eqx.tree_deserialise_leaves(
        os.path.join(path, ckpt), (model_skeleton, normaliser_skeleton))
    return create_distribution(config, model, normaliser), config, pgm


def first_dir_per_mode(results):
    mode_to_dir = {}
    for run_id in sorted(os.listdir(results)):
        run_dir = os.path.join(results, run_id)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isdir(run_dir) or not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            mode = json.load(f).get('mode')
        if mode and mode not in mode_to_dir:
            mode_to_dir[mode] = run_dir
    return mode_to_dir


def gen_samples(dist, pgm, key):
    """Returns (model_images, real_images) stacked over LABELS, shape (n, 784)."""
    model_imgs, real_imgs = [], []
    for label in LABELS:
        oh = jax.nn.one_hot(label, 10)
        mkeys = random.split(random.fold_in(key, label), N_PER_LABEL)

        @eqx.filter_jit
        def batch(keys):
            return jax.vmap(lambda k: dist.sample(k, conditions={'label': oh})['image'])(keys)

        model_imgs.append(np.asarray(batch(mkeys)))
        rkeys = random.split(random.fold_in(key, 1000 + label), N_PER_LABEL)
        real = jax.vmap(lambda k: pgm.sample(k, conditions={'label': oh}, training=False)['image'])(rkeys)
        real_imgs.append(np.asarray(real))
    return np.concatenate(model_imgs), np.concatenate(real_imgs)


def swd_terms(key, X, Y, num_projections=200):
    """Returns the per-element squared sorted-difference terms whose MEAN (then sqrt,
    then normalization) is the SWD. Lets us see how concentrated the SWD is."""
    X, Y = np.asarray(X), np.asarray(Y)
    d, N, M = X.shape[1], X.shape[0], Y.shape[0]
    dirs = np.asarray(jax.random.normal(key, (d, num_projections)))
    dirs = dirs / np.linalg.norm(dirs, axis=0)
    Xp, Yp = X @ dirs, Y @ dirs
    sq = (np.sort(Xp, axis=0) - np.sort(Yp, axis=0)) ** 2  # (N, num_projections)
    norm = np.sqrt(max(N, M) * d)
    return sq, norm


def trimmed_swd(sq, norm, trim_frac):
    """SWD recomputed after dropping the largest `trim_frac` of squared terms."""
    flat = np.sort(sq.ravel())
    keep = flat[: int(len(flat) * (1.0 - trim_frac))]
    return np.sqrt(keep.mean()) / norm


def main():
    mode_to_dir = first_dir_per_mode(RESULTS)
    key = random.PRNGKey(0)
    skey = random.PRNGKey(42)
    print(f"Modes found: {list(mode_to_dir)}\n")

    header =(f"{'mode':>10} | {'SWD raw':>9} | {'trim 0.1%':>9} | {'trim 1%':>9} | "
              f"{'top1% share':>11} | {'%px>realmax':>11} | {'max|px|':>9}")
    print(header)
    print("-" * len(header))
    rows = {}
    for mode, d in mode_to_dir.items():
        dist, config, pgm = load(d)
        model_x, real_x = gen_samples(dist, pgm, key)
        real_max = float(np.max(real_x))
        sq, norm = swd_terms(skey, model_x, real_x)
        swd_raw = np.sqrt(sq.mean()) / norm
        swd_t01 = trimmed_swd(sq, norm, 0.001)
        swd_t1 = trimmed_swd(sq, norm, 0.01)
        # fraction of total squared mass carried by the top 1% of terms
        flat = np.sort(sq.ravel())
        top1_share = 100.0 * flat[int(len(flat) * 0.99):].sum() / flat.sum()
        frac_oor = 100.0 * np.mean(model_x > real_max)
        mx = float(np.max(np.abs(model_x)))
        rows[mode] = dict(swd_raw=swd_raw, swd_trim01=swd_t01, swd_trim1=swd_t1,
                          top1_share=top1_share, frac_oor=frac_oor, max_px=mx, real_max=real_max)
        print(f"{mode:>10} | {swd_raw:9.5f} | {swd_t01:9.5f} | {swd_t1:9.5f} | "
              f"{top1_share:10.1f}% | {frac_oor:10.2f}% | {mx:9.1f}")

    print(f"\n(real-image max intensity ~= {rows[next(iter(rows))]['real_max']:.2f})")
    if 'dynamic' in rows and 'static' in rows:
        print(f"\nDynamic vs static, raw SWD ratio:        {rows['dynamic']['swd_raw']/rows['static']['swd_raw']:.2f}x")
        print(f"Dynamic vs static, after 1% trim ratio:  {rows['dynamic']['swd_trim1']/rows['static']['swd_trim1']:.2f}x")
    np.save(os.path.join(os.path.dirname(__file__), "mnist_outlier_diagnostic.npy"), rows, allow_pickle=True)


if __name__ == "__main__":
    main()
