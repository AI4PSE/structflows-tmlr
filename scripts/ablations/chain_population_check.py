"""Population check for the chain (Section 'Price of Parallelism'): what error does the coupling
alone allow dynamic conditioning, separately from learning?
Linear-Gaussian chain -> treat (z, x) as jointly Gaussian with cross-cov C = Cov(z, x).
Dynamic optimum (linear): v_k(t) = E[x_k - z_k | y_t^{1:k}], y_t = (1-t) z + t x.
Integrate the covariance ODE from N(0, I) and report W2 to the data (joint and root|leaf).
Couplings: exact KR (Cholesky), independent, empirical per-block C2OT (batch 128, as in training)."""
import os, sys, json, numpy as np
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
from scipy.linalg import sqrtm
from pgm_paper_utils.pgm_factory import create_pgm
from pgm_paper_utils.utils import make_structure
from sinterp.couplings import C2OTCoupling

def w2_gauss(S1, S2):
    r = np.real(sqrtm(S2)); return float(np.sqrt(max(np.trace(S1 + S2 - 2*np.real(sqrtm(r @ S1 @ r))), 0)))

def dyn_cov(Sx, C, nt=400):
    """Covariance at t=1 of the flow driven by the block-triangular linear projection."""
    n = Sx.shape[0]; I = np.eye(n); P = I.copy(); ts = np.linspace(0, 1, nt + 1)
    def V(t):
        Syy = (1-t)**2*I + t**2*Sx + t*(1-t)*(C + C.T)          # Cov(y_t)
        Svy = (t*Sx + (1-t)*C.T) - (t*C + (1-t)*I)              # Cov(x - z, y_t)
        Svy = Svy  # rows = velocity comps, cols = y comps
        Vm = np.zeros((n, n))
        for k in range(n):
            A = Syy[:k+1, :k+1]; b = Svy[k, :k+1]
            Vm[k, :k+1] = np.linalg.solve(A + 1e-10*np.eye(k+1), b)
        return Vm
    for a, b in zip(ts[:-1], ts[1:]):                           # RK4 on P' = VP + PV^T
        f = lambda t, P: V(t) @ P + P @ V(t).T
        h = b - a; k1 = f(a, P); k2 = f(a+h/2, P+h/2*k1); k3 = f(a+h/2, P+h/2*k2); k4 = f(b, P+h*k3)
        P = P + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return P

def root_post(Sx):   # Cov(root | leaf): order is [leaf, ..., root]
    return Sx[-1, -1] - Sx[-1, 0]**2 / Sx[0, 0]

def run(n, n_batches=300, standardise=True):
    cfg = dict(pgm_type='chain', mechanism_type='linear', weight_scale=1.0, noise_scale=0.2, root_scale=1.0,
               n_nodes=n, node_dim=1, task='inference', pgm_seed=0, sparsity=0.3)
    pgm = create_pgm('chain', cfg, jax.random.PRNGKey(0)); st = make_structure(pgm, cfg)
    order, deps = st.block_order, st.dependencies
    xs = jax.vmap(pgm.sample)(jax.random.split(jax.random.PRNGKey(1), 128*n_batches))
    X = np.stack([np.asarray(xs[b]).reshape(-1) for b in order], 1)
    X = X - X.mean(0)
    if standardise: X = X / X.std(0)
    Sx = np.cov(X.T)
    L = np.linalg.cholesky(Sx)
    cp = C2OTCoupling(target_ratio=0.05)
    f = jax.jit(lambda k, xd, ed: cp.couple_blocks(k, xd, ed, deps))
    Z = np.random.default_rng(2).standard_normal(X.shape)
    Zc = np.empty_like(Z)
    for i in range(n_batches):
        sl = slice(128*i, 128*(i+1))
        xd = {b: jnp.asarray(X[sl, j:j+1]) for j, b in enumerate(order)}
        ed = {b: jnp.asarray(Z[sl, j:j+1]) for j, b in enumerate(order)}
        out = f(jax.random.PRNGKey(i), xd, ed)
        Zc[sl] = np.stack([np.asarray(out[b])[:, 0] for b in order], 1)
    Cemp = (Zc - Zc.mean(0)).T @ X / len(X)                      # Cov(z, x)
    res = {}
    for name, C in [('KR', L.T), ('indep', np.zeros_like(Sx)), ('c2ot', Cemp)]:
        P = dyn_cov(Sx, C)
        aP, aS = P[-1, 0]/P[0, 0], Sx[-1, 0]/Sx[0, 0]
        sP, sS = np.sqrt(max(root_post(P), 0)), np.sqrt(root_post(Sx))
        res[name] = dict(w2_joint=round(w2_gauss(P, Sx), 4), root_post_sd=round(float(sP), 4),
                         w2_root_given_leaf=round(float(np.sqrt((aP-aS)**2*Sx[0, 0] + (sP-sS)**2)), 4))
    res['true_root_post_sd'] = round(float(np.sqrt(root_post(Sx))), 4)
    # how close is the empirical coupling to KR?  KR: z = L^{-1} x exactly
    Zkr = np.linalg.solve(L, X.T).T
    res['corr(z_c2ot, z_KR) per block'] = [round(float(np.corrcoef(Zc[:, j], Zkr[:, j])[0, 1]), 3) for j in range(n)]
    return res

if __name__ == "__main__":
    # Run: CHAIN_POP_OUT=$RESULTS/chain_rw/population.json python scripts/ablations/chain_population_check.py
    out = {}
    for n in [int(a) for a in sys.argv[1:]] or [2, 4, 8, 16, 32]:
        out[n] = run(n); print(n, json.dumps(out[n]), flush=True)
    if os.environ.get("CHAIN_POP_OUT"):
        json.dump(out, open(os.environ["CHAIN_POP_OUT"], "w"), indent=1)
