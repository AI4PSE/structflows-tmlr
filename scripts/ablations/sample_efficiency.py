"""Sample-efficiency sweep: sparse structure as an inductive bias (Exp 2).

When the data are limited, embedding the correct conditional-independence
structure into the flow (hard sparsity) should generalize better than an
unstructured, fully-connected flow of the same family. We test this on a
linear-Gaussian chain  x0 -> x1 -> ... -> x5, performing posterior inference
p(x0..x4 | x5).

We compare two static flows that differ *only* in connectivity:
  * SPARSE  : the structure produced by StructFlow's inference compiler
              (for a chain this is the Markov chain itself; 5 edges).
  * DENSE   : the same block order but fully connected (15 edges) -- the
              unstructured baseline.

Both are trained identically on N joint samples, for a grid of N. Accuracy is
the 2-Wasserstein (Bures) distance between the flow's posterior and the exact
Gaussian posterior -- an interpretable, zero-based metric (no SWD normalization).
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import random, vmap
from scipy.linalg import sqrtm

from structflow.pgms.factories import make_linear_chain_pgm
from structflow.pgms.pgm_compiler import PGMCompiler
from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.flow_factory import create_static_flow
from structflow.conditional_flows.dist_factory import create_static_distribution

N_NODES = 6
NAMES = [f"x{i}" for i in range(N_NODES)]
COND = NAMES[-1]                 # observe the chain endpoint
TARGET_ORDER = NAMES[:-1]        # infer the rest, fixed order x0..x4


def joint_gaussian(pgm, key, n=200_000):
    """Estimate the joint Gaussian (mean, cov) over [x0..x5] by ancestral sampling."""
    s = pgm.b_sample(key, n)
    X = np.concatenate([np.asarray(s[k]).reshape(n, -1) for k in NAMES], axis=1)
    return X.mean(0), np.cov(X, rowvar=False)


def analytic_posterior(mu, Sigma, c):
    """Exact p(targets | x5=c) for the joint Gaussian over [x0..x5]."""
    t = np.arange(N_NODES - 1)          # target indices 0..4
    o = np.array([N_NODES - 1])         # condition index 5
    mu_t, mu_o = mu[t], mu[o]
    S_tt, S_to, S_oo = Sigma[np.ix_(t, t)], Sigma[np.ix_(t, o)], Sigma[np.ix_(o, o)]
    K = S_to @ np.linalg.inv(S_oo)
    mean = mu_t + (K @ (np.atleast_1d(c) - mu_o))
    cov = S_tt - K @ S_to.T
    return mean, cov


def bures_w2(m1, C1, m2, C2):
    """2-Wasserstein distance between two Gaussians."""
    diff = float(np.sum((m1 - m2) ** 2))
    s1 = sqrtm(C1)
    inner = sqrtm(s1 @ C2 @ s1)
    bures = float(np.trace(C1 + C2 - 2.0 * inner.real))
    return float(np.sqrt(max(diff + bures, 0.0)))


def structures():
    dims = {n: 1 for n in NAMES}
    dag = {n: ([NAMES[i - 1]] if i > 0 else []) for i, n in enumerate(NAMES)}
    sparse = PGMCompiler(dag, dims).compile_inference(
        targets=set(TARGET_ORDER), conditions={COND})
    dense_deps = {b: list(sparse.block_order[:i]) for i, b in enumerate(sparse.block_order)}
    dense = BlockTriangularStructure(
        block_order=sparse.block_order, dependencies=dense_deps, dims=dims)
    return sparse, dense


def train(structure, data, n, steps, width, depth, lr, batch, seed):
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.05))
    model, step_fn = create_static_flow(
        structure, depth=depth, width=width, model_key=random.PRNGKey(seed),
        optimizer=opt, use_coupling=False, normaliser=None)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    k = random.PRNGKey(seed + 100)
    bs = min(batch, n)
    for _ in range(steps):
        k, sk, bk = random.split(k, 3)
        idx = random.randint(bk, (bs,), 0, n)
        b = {name: data[name][idx] for name in data}
        model, opt_state, _, _ = step_fn(model, opt_state, None, b, sk)
    return model


def eval_w2(model, structure, mu, Sigma, c_values, n_post, seed):
    dist = create_static_distribution(model, structure, n_steps=20, integrator_name="rk4")

    @eqx.filter_jit
    def draw(keys, c):
        return vmap(lambda kk: dist.sample(kk, conditions={COND: c},
                                           targets=set(TARGET_ORDER)))(keys)
    w2s = []
    for j, c in enumerate(c_values):
        ks = random.split(random.PRNGKey(seed * 17 + j), n_post)
        out = draw(ks, jnp.array([float(c)]))
        S = np.stack([np.asarray(out[k]).reshape(-1) for k in TARGET_ORDER], axis=1)
        m_post, C_post = analytic_posterior(mu, Sigma, c)
        w2s.append(bures_w2(S.mean(0), np.cov(S, rowvar=False), m_post, C_post))
    return float(np.mean(w2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-post", type=int, default=2000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", type=str,
                    default="../../images/ablations/sample_efficiency.png")
    args = ap.parse_args()

    key = random.PRNGKey(0)
    pgm = make_linear_chain_pgm(n_nodes=N_NODES, key=key)
    mu, Sigma = joint_gaussian(pgm, random.PRNGKey(1))
    c_values = np.quantile(
        np.asarray(pgm.b_sample(random.PRNGKey(2), 20000)[COND]).reshape(-1),
        [0.2, 0.5, 0.8])

    sparse, dense = structures()
    print("sparse edges:", sum(len(v) for v in sparse.dependencies.values()),
          " dense edges:", sum(len(v) for v in dense.dependencies.values()))

    # large fixed pool of joint data; each run uses the first N rows
    pool = pgm.b_sample(random.PRNGKey(3), max(args.sizes))
    pool = {k: jnp.asarray(np.asarray(v).reshape(max(args.sizes), -1)) for k, v in pool.items()}

    results = {"sparse": {}, "dense": {}}
    for N in args.sizes:
        for label, struct in [("sparse", sparse), ("dense", dense)]:
            vals = []
            for seed in range(args.seeds):
                data = {k: v[:N] for k, v in pool.items()}
                model = train(struct, data, N, args.steps, args.width, args.depth,
                              args.lr, args.batch, seed=seed + 1)
                vals.append(eval_w2(model, struct, mu, Sigma, c_values, args.n_post, seed + 1))
            results[label][N] = (float(np.mean(vals)), float(np.std(vals)))
            m, s = results[label][N]
            print(f"  N={N:5d}  {label:6s}  W2={m:.3f} +/- {s:.3f}")

    plot(args.out, args.sizes, results)
    print(f"\nSaved figure to {os.path.abspath(args.out)}")
    # headline: low-N gap
    Nlo = args.sizes[0]
    print(f"\nHEADLINE @ N={Nlo}:  sparse W2={results['sparse'][Nlo][0]:.3f}  "
          f"dense W2={results['dense'][Nlo][0]:.3f}  "
          f"(structure is {results['dense'][Nlo][0]/max(results['sparse'][Nlo][0],1e-6):.2f}x better)")


def plot(out, sizes, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for label, col in [("sparse", "tab:blue"), ("dense", "tab:red")]:
        m = np.array([results[label][N][0] for N in sizes])
        s = np.array([results[label][N][1] for N in sizes])
        name = "StructFlow (sparse, compiled)" if label == "sparse" else "Unstructured (dense)"
        ax.plot(sizes, m, "-o", color=col, label=name)
        ax.fill_between(sizes, m - s, m + s, color=col, alpha=0.2)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)                       # zero-based, interpretable
    ax.set_xlabel("training samples (N)")
    ax.set_ylabel(r"posterior error: $W_2$ to exact posterior")
    ax.set_title("Sample efficiency of structured inference")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
