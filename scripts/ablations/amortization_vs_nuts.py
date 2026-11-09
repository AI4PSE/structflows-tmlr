"""Amortized inference vs. per-query MCMC (Exp 3).

StructFlow's inference-aligned flow is *amortized*: it is trained once on joint
samples, after which any posterior query p(targets | conditions=c) is a single
fast ODE solve. A classical Bayesian baseline (NUTS) must instead run a fresh
MCMC chain for every observed value c.

We compare the two on a nonlinear collider  x1 -> y <- x2  (query: infer the
causes x1, x2 from the observed effect y). We measure:
  * accuracy: Sliced-Wasserstein distance between StructFlow's amortized
    posterior samples and the NUTS reference, per query;
  * wall-clock: cumulative time as the number of distinct queries K grows.
    NUTS scales as K * t_chain; StructFlow as t_train + K * t_solve, so
    amortization wins beyond a crossover K*.
"""

import os
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import random, vmap

from structflow.pgms.factories import make_collider_pgm
from structflow.pgms.pgm_compiler import PGMCompiler
from structflow.conditional_flows.flow_factory import create_static_flow
from structflow.conditional_flows.dist_factory import create_static_distribution
from structflow.bayes.metrics import compute_swd_jax

TARGETS = {"x1", "x2"}
CONDITION = "y"


def block(x):
    """Block until JAX async dispatch completes (for honest timing)."""
    jax.block_until_ready(x)
    return x


def train(structure, data, steps, width, depth, lr, batch, seed=1):
    n = data["x1"].shape[0]
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.05))
    model, step_fn = create_static_flow(
        structure, depth=depth, width=width, model_key=random.PRNGKey(seed),
        optimizer=opt, use_coupling=False, normaliser=None,
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    k = random.PRNGKey(seed + 100)
    for _ in range(steps):
        k, sk, bk = random.split(k, 3)
        idx = random.randint(bk, (batch,), 0, n)
        b = {name: data[name][idx] for name in data}
        model, opt_state, _, loss = step_fn(model, opt_state, None, b, sk)
    block(loss)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-data", type=int, default=8192)
    ap.add_argument("--n-post", type=int, default=1000, help="posterior samples per query")
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--mechanism", type=str, default="nonlinear_parent")
    ap.add_argument("--out", type=str,
                    default="../../images/ablations/amortization_vs_nuts.png")
    args = ap.parse_args()

    key = random.PRNGKey(0)
    k_pgm, k_data, k_train, k_q = random.split(key, 4)

    # --- nonlinear collider PGM -------------------------------------------- #
    pgm = make_collider_pgm(n_causes=2, mechanism_type=args.mechanism, key=k_pgm)
    dims = {"x1": 1, "x2": 1, "y": 1}

    # Joint training data (ancestral, cheap).
    data = pgm.b_sample(k_data, args.n_data)
    data = {k: np.asarray(v).reshape(args.n_data, -1) for k, v in data.items()}
    data = {k: jnp.asarray(v) for k, v in data.items()}

    # --- train StructFlow once --------------------------------------------- #
    structure = PGMCompiler({"x1": [], "x2": [], "y": ["x1", "x2"]}, dims).compile_inference(
        targets=TARGETS, conditions={CONDITION})
    print("Inference structure deps:", structure.dependencies)

    t0 = time.perf_counter()
    model = train(structure, data, args.steps, args.width, args.depth, args.lr, args.batch)
    t_train = time.perf_counter() - t0
    dist = create_static_distribution(model, structure, n_steps=20, integrator_name="rk4")
    print(f"StructFlow one-time training: {t_train:.1f}s")

    @eqx.filter_jit
    def flow_sample(keys, y):
        return vmap(lambda kk: dist.sample(kk, conditions={CONDITION: y}, targets=TARGETS))(keys)

    # --- query points: observed y values ----------------------------------- #
    y_query = np.asarray(data["y"])[:args.queries].reshape(args.queries, 1)

    # warm-up (compile) both paths on the first query, untimed
    y0 = jnp.asarray(y_query[0])
    ks = random.split(random.PRNGKey(1), args.n_post)
    _ = block(flow_sample(ks, y0))
    _ = block(pgm.b_sample(random.PRNGKey(2), args.n_post,
                           conditions={CONDITION: y0}, targets=TARGETS))

    # --- run all queries, timing each path --------------------------------- #
    flow_times, nuts_times, swds = [], [], []
    for i in range(args.queries):
        yk = jnp.asarray(y_query[i])
        kq = random.fold_in(k_q, i)

        ksf = random.split(random.fold_in(kq, 0), args.n_post)
        t0 = time.perf_counter()
        sf = block(flow_sample(ksf, yk))
        flow_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        nuts = block(pgm.b_sample(random.fold_in(kq, 1), args.n_post,
                                  conditions={CONDITION: yk}, targets=TARGETS))
        nuts_times.append(time.perf_counter() - t0)

        sf_xy = jnp.concatenate([sf["x1"].reshape(-1, 1), sf["x2"].reshape(-1, 1)], axis=1)
        nuts_xy = jnp.concatenate([nuts["x1"].reshape(-1, 1), nuts["x2"].reshape(-1, 1)], axis=1)
        swds.append(float(compute_swd_jax(random.fold_in(kq, 2), sf_xy, nuts_xy)))

    flow_times = np.array(flow_times); nuts_times = np.array(nuts_times); swds = np.array(swds)
    t_flow = flow_times.mean(); t_nuts = nuts_times.mean()
    crossover = t_train / (t_nuts - t_flow) if t_nuts > t_flow else np.inf

    print(f"\nPer-query wall-clock ({args.n_post} samples):")
    print(f"  NUTS       : {t_nuts*1e3:8.1f} ms/query")
    print(f"  StructFlow : {t_flow*1e3:8.1f} ms/query   ({t_nuts/t_flow:.0f}x faster per query)")
    print(f"  amortization crossover: K* = {crossover:.1f} queries "
          f"(train cost {t_train:.1f}s amortizes after this many queries)")
    print(f"\nPosterior agreement (SWD StructFlow vs NUTS): "
          f"mean={swds.mean():.4f}  median={np.median(swds):.4f}")

    plot(args.out, t_train, t_flow, t_nuts, args.queries, swds, crossover, args.mechanism)
    print(f"\nSaved figure to {os.path.abspath(args.out)}")


def plot(out, t_train, t_flow, t_nuts, K, swds, crossover, mech):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)
    Kmax = max(int(crossover * 2.5) if np.isfinite(crossover) else K, K)
    ks = np.arange(0, Kmax + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    ax.plot(ks, t_nuts * ks, label="NUTS (per-query MCMC)", color="tab:red", lw=2)
    ax.plot(ks, t_train + t_flow * ks, label="StructFlow (amortized)", color="tab:blue", lw=2)
    if np.isfinite(crossover) and crossover <= Kmax:
        ax.axvline(crossover, ls="--", c="grey")
        ax.text(crossover, ax.get_ylim()[1] * 0.5, f"  crossover K*≈{crossover:.0f}",
                fontsize=9, color="grey")
    ax.set_xlabel("number of distinct posterior queries (K)")
    ax.set_ylabel("cumulative wall-clock (s)")
    ax.set_title("Amortization: total inference cost")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(swds, bins=15, color="tab:blue", alpha=0.8)
    ax.axvline(swds.mean(), c="k", ls="--", label=f"mean={swds.mean():.4f}")
    ax.set_xlabel("SWD(StructFlow, NUTS) per query")
    ax.set_ylabel("count")
    ax.set_title(f"Posterior agreement ({mech} collider)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
