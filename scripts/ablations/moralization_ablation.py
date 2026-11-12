"""Moralization ablation on a collider (Exp 1).

Demonstrates that the moral edge produced by StructFlow's inference compiler is
*necessary* to capture the explaining-away correlation in a posterior.

Setup: linear-Gaussian collider  A -> C <- B  with
    A, B ~ N(0, 1),   C = A + B + sigma * eps,   eps ~ N(0, 1).
Observing C induces a (strong, negative) correlation between A and B. The exact
posterior p(A, B | C=c) is Gaussian and available in closed form.

We compile the inference structure for query targets={A,B}, conditions={C}:
  * MORALIZED   (StructFlow): block order [C, B, A] with deps A:[C,B]  -> the
    A-B edge lets the flow represent Corr(A,B|C).
  * NON-MORAL   (ablation):   same order but deps A:[C], B:[C]  -> the flow
    factorizes as p(A|C)p(B|C) and *cannot* represent Corr(A,B|C).

Both are trained identically on the same joint samples. We report the recovered
posterior correlation / means against the analytic ground truth.
"""

from pgm_paper_utils.paths import IMAGES
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import random, vmap

from structflow.pgms.structure import BlockTriangularStructure
from structflow.pgms.pgm_compiler import PGMCompiler
from structflow.conditional_flows.flow_factory import create_static_flow
from structflow.conditional_flows.dist_factory import create_static_distribution

SIGMA = 0.3  # collider noise std


# --------------------------------------------------------------------------- #
# Analytic ground truth
# --------------------------------------------------------------------------- #
def analytic_posterior(c, sigma=SIGMA):
    """Exact p(A, B | C=c) for A,B~N(0,1), C=A+B+sigma*eps. Returns (mean, cov)."""
    var_c = 2.0 + sigma ** 2
    # Cov([A,B], C) = [1, 1]; condition the joint Gaussian on C=c.
    cov_xc = jnp.array([1.0, 1.0])
    mean = cov_xc / var_c * c                       # E[[A,B]|C=c]
    prior = jnp.eye(2)
    cov = prior - jnp.outer(cov_xc, cov_xc) / var_c  # Cov[[A,B]|C=c]
    return mean, cov


def corr_from_cov(cov):
    return cov[0, 1] / jnp.sqrt(cov[0, 0] * cov[1, 1])


# --------------------------------------------------------------------------- #
# Data + structures
# --------------------------------------------------------------------------- #
def make_data(key, n, sigma=SIGMA):
    ka, kb, ke = random.split(key, 3)
    A = random.normal(ka, (n, 1))
    B = random.normal(kb, (n, 1))
    C = A + B + sigma * random.normal(ke, (n, 1))
    return {"A": A, "B": B, "C": C}


def make_structures(dims):
    """Return (moralized, non_moralized) inference structures for targets {A,B}|{C}."""
    pgm_dag = {"A": [], "B": [], "C": ["A", "B"]}
    comp = PGMCompiler(pgm_dag, dims)
    moral = comp.compile_inference(targets={"A", "B"}, conditions={"C"})

    targets = {"A", "B"}
    deps_nm = {
        blk: [p for p in parents if not (blk in targets and p in targets)]
        for blk, parents in moral.dependencies.items()
    }
    non_moral = BlockTriangularStructure(
        block_order=moral.block_order, dependencies=deps_nm, dims=dims
    )
    return moral, non_moral


# --------------------------------------------------------------------------- #
# Training + sampling
# --------------------------------------------------------------------------- #
def train(structure, data, steps, width, depth, lr, batch, seed=1):
    n = data["A"].shape[0]
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    opt = optax.adam(sched)
    model, step_fn = create_static_flow(
        structure, depth=depth, width=width, model_key=random.PRNGKey(seed),
        optimizer=opt, use_coupling=False, normaliser=None,
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    k = random.PRNGKey(seed + 100)
    last = None
    for i in range(steps):
        k, sk, bk = random.split(k, 3)
        idx = random.randint(bk, (batch,), 0, n)
        b = {name: data[name][idx] for name in data}
        model, opt_state, _, loss = step_fn(model, opt_state, None, b, sk)
        last = loss
    return model, float(last)


def sample_posterior(dist, c_value, n_samples, seed=7):
    c = jnp.array([float(c_value)])
    ks = random.split(random.PRNGKey(seed), n_samples)
    out = vmap(lambda kk: dist.sample(kk, conditions={"C": c}, targets={"A", "B"}))(ks)
    return np.asarray(out["A"]).reshape(-1), np.asarray(out["B"]).reshape(-1)


def summarize(name, a, b, gt_mean, gt_cov):
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"  {name:10s}  E[A]={a.mean():+.3f} E[B]={b.mean():+.3f}  "
          f"sd[A]={a.std():.3f} sd[B]={b.std():.3f}  corr(A,B|C)={corr:+.3f}")
    return corr


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-data", type=int, default=8192)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--out", type=str,
                    default=os.path.join(IMAGES, "ablations", "moralization_collider.png"))
    args = ap.parse_args()

    dims = {"A": 1, "B": 1, "C": 1}
    data = make_data(random.PRNGKey(0), args.n_data)
    moral, non_moral = make_structures(dims)
    print("Moralized   deps:", moral.dependencies)
    print("Non-moral   deps:", non_moral.dependencies)

    gt_mean, gt_cov = analytic_posterior(args.c)
    gt_corr = float(corr_from_cov(gt_cov))
    print(f"\nAnalytic posterior at C={args.c}:  "
          f"E[A]=E[B]={float(gt_mean[0]):+.3f}  corr={gt_corr:+.3f}\n")

    print("Training moralized (StructFlow) ...")
    m_mor, l_mor = train(moral, data, args.steps, args.width, args.depth, args.lr, args.batch)
    print("Training non-moral (ablation) ...")
    m_non, l_non = train(non_moral, data, args.steps, args.width, args.depth, args.lr, args.batch)
    print(f"final loss: moral={l_mor:.4f}  non-moral={l_non:.4f}\n")

    d_mor = create_static_distribution(m_mor, moral, n_steps=50, integrator_name="diffrax")
    d_non = create_static_distribution(m_non, non_moral, n_steps=50, integrator_name="diffrax")

    a_mor, b_mor = sample_posterior(d_mor, args.c, args.n_samples)
    a_non, b_non = sample_posterior(d_non, args.c, args.n_samples)

    # Ground-truth samples
    rng = np.random.default_rng(0)
    gt = rng.multivariate_normal(np.asarray(gt_mean), np.asarray(gt_cov), size=args.n_samples)

    print(f"Posterior summaries at C={args.c} (analytic corr={gt_corr:+.3f}):")
    summarize("ground-truth", gt[:, 0], gt[:, 1], gt_mean, gt_cov)
    c_mor = summarize("MORALIZED", a_mor, b_mor, gt_mean, gt_cov)
    c_non = summarize("NON-MORAL", a_non, b_non, gt_mean, gt_cov)

    plot(args.out, args.c, gt, (a_mor, b_mor, c_mor), (a_non, b_non, c_non), gt_corr)
    print(f"\nSaved figure to {os.path.abspath(args.out)}")
    print(f"\nHEADLINE:  analytic corr={gt_corr:+.2f} | "
          f"moralized={c_mor:+.2f} | non-moral={c_non:+.2f}")


def plot(out, c, gt, mor, non, gt_corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out), exist_ok=True)
    a_mor, b_mor, c_mor = mor
    a_non, b_non, c_non = non
    panels = [
        ("Ground truth", gt[:, 0], gt[:, 1], gt_corr, "tab:green"),
        ("StructFlow (moralized)", a_mor, b_mor, c_mor, "tab:blue"),
        ("Ablation (no moral edge)", a_non, b_non, c_non, "tab:red"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, (title, a, b, corr, col) in zip(axes, panels):
        ax.scatter(a, b, s=5, alpha=0.25, color=col)
        ax.set_title(f"{title}\ncorr(A,B|C={c})={corr:+.2f}", fontsize=10)
        ax.set_xlabel("A"); ax.axhline(0, lw=.5, c="grey"); ax.axvline(0, lw=.5, c="grey")
    axes[0].set_ylabel("B")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
