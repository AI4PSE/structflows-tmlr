import jax
import jax.numpy as jnp
import time
from structflow.conditional_flows.dynamic.layers import DenseMaskedLinear, SparseMaskedLinear

def benchmark_layers(in_features=1000, out_features=1000, sparsity=0.1):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Generate a random mask
    mask = (jax.random.uniform(k1, (out_features, in_features)) < sparsity).astype(jnp.float32)
    
    # Initialize layers
    dense_layer = DenseMaskedLinear(in_features, out_features, mask, k2)
    sparse_layer = SparseMaskedLinear(in_features, out_features, mask, k2)
    
    # Input
    x = jax.random.normal(k3, (in_features,))
    
    # JIT compile
    import equinox as eqx
    dense_fn = eqx.filter_jit(dense_layer)
    sparse_fn = eqx.filter_jit(sparse_layer)
    
    # Warmup
    dense_fn(x).block_until_ready()
    sparse_fn(x).block_until_ready()
    
    # Benchmark
    iters = 1000
    
    start = time.time()
    for _ in range(iters):
        dense_fn(x).block_until_ready()
    dense_time = (time.time() - start) / iters
    
    start = time.time()
    for _ in range(iters):
        sparse_fn(x).block_until_ready()
    sparse_time = (time.time() - start) / iters
    
    print(f"Benchmark Results (In={in_features}, Out={out_features}, Sparsity={sparsity})")
    print(f"DenseMaskedLinear: {dense_time*1000:.4f} ms")
    print(f"SparseMaskedLinear: {sparse_time*1000:.4f} ms")
    print(f"Ratio (Dense/Sparse): {dense_time/sparse_time:.2f}x")
    print("-" * 30)

if __name__ == "__main__":
    benchmark_layers(100, 100, 0.5)
    benchmark_layers(1000, 1000, 0.5)
    benchmark_layers(4000, 4000, 0.5)
    # benchmark_layers(4000, 4000, 0.01)
