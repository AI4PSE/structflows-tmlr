"""
Scaling benchmark for StructuredVelocityFn vs SimpleStructuredVelocityFn
Tests on random DAG structures with varying complexity and size.
"""

import sys
import os
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.static.net import (
    SimpleStructuredVelocityFn, 
    StructuredVelocityFn, 
    ConditionalBlock
)

# ============================================================================
# Random DAG Generator (from pgms)
# ============================================================================

def make_random_dag_structure(n_vars: int, density: float = 0.3) -> BlockTriangularStructure:
    """
    Generate a random DAG structure with specified density.
    
    Args:
        n_vars: Number of variables
        density: Probability that an edge exists between any two variables (0.0 to 1.0)
    
    Returns:
        BlockTriangularStructure representing the DAG
    """
    import random as py_random
    
    block_order = [f'x{i}' for i in range(n_vars)]
    dependencies = {}
    
    # For each variable, randomly select some earlier variables as parents
    for i in range(n_vars):
        name = block_order[i]
        possible_parents = block_order[:i]
        
        # Expected number of parents
        n_possible_parents = len(possible_parents)
        expected_parents = int(n_possible_parents * density)
        
        if expected_parents > 0 and possible_parents:
            # Add some randomness to the actual number
            n_parents = max(0, expected_parents + py_random.randint(-1, 1))
            n_parents = min(n_parents, len(possible_parents))
            
            if n_parents > 0:
                parents = py_random.sample(possible_parents, n_parents)
                dependencies[name] = sorted(parents)
            else:
                dependencies[name] = []
        else:
            dependencies[name] = []
    
    dims = {name: 1 for name in block_order}
    return BlockTriangularStructure(block_order, dependencies, dims)

# ============================================================================
# Benchmark Implementation
# ============================================================================

class StaticVelocityFunctionBenchmark:
    """Benchmarks static velocity function implementations."""
    
    def __init__(self, velocity_fn, name: str, structure: BlockTriangularStructure):
        self.velocity_fn = velocity_fn
        self.name = name
        self.structure = structure
        
    def create_dummy_input(self) -> Tuple[Dict[str, jnp.ndarray], float]:
        """Create a dummy input for the velocity function."""
        x_all = {name: jnp.ones((self.structure.dims[name],)) for name in self.structure.block_order}
        t = 0.5
        return x_all, t
    
    @eqx.filter_jit
    def call(self, x_all: Dict[str, jnp.ndarray], t: float):
        """Call the velocity function."""
        return self.velocity_fn(x_all, t)
    
    def benchmark(self, n_runs: int = 30, warmup: bool = True) -> float:
        """Run benchmark and return average time in seconds."""
        x_all, t = self.create_dummy_input()
        
        if warmup:
            # Warmup to trigger JIT
            self.call(x_all, t)
        
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = self.call(x_all, t)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - start)
        
        return np.mean(times)

def benchmark_static_on_structure(structure: BlockTriangularStructure, 
                                  struct_name: str,
                                  n_runs: int = 30) -> Dict:
    """Benchmark static implementations on a given structure."""
    print(f"  Benchmarking {struct_name} ({len(structure.block_order)} blocks)...")
    
    key = jax.random.PRNGKey(42)
    key_simple, key_struct = jax.random.split(key)
    
    # Create velocity functions
    simple_vel = SimpleStructuredVelocityFn(
        structure,
        key_simple,
        lambda **kwargs: ConditionalBlock(hidden_dim=32, depth=2, **kwargs)
    )
    
    struct_vel = StructuredVelocityFn(
        structure,
        key_struct,
        lambda **kwargs: ConditionalBlock(hidden_dim=32, depth=2, **kwargs)
    )
    
    # Benchmark
    simple_bench = StaticVelocityFunctionBenchmark(simple_vel, "Simple", structure)
    struct_bench = StaticVelocityFunctionBenchmark(struct_vel, "Structured", structure)
    
    simple_time = simple_bench.benchmark(n_runs=n_runs)
    struct_time = struct_bench.benchmark(n_runs=n_runs)
    
    speedup = simple_time / struct_time if struct_time > 0 else 0
    
    print(f"    Simple:      {simple_time*1000:.4f} ms")
    print(f"    Structured:  {struct_time*1000:.4f} ms")
    print(f"    Speedup:     {speedup:.2f}x")
    
    return {
        'structure': struct_name,
        'n_blocks': len(structure.block_order),
        'density': None,  # Will be filled below
        'simple_time': simple_time,
        'struct_time': struct_time,
        'speedup': speedup
    }

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("Scaling Benchmark: StructuredVelocityFn vs SimpleStructuredVelocityFn")
    print("Testing on random DAG structures with varying size and density")
    print("=" * 80)
    
    results = []
    
    # Test 1: Varying DAG size with fixed density
    print("\n" + "=" * 80)
    print("Test 1: Varying DAG Size (density=0.3)")
    print("=" * 80)
    
    sizes = [4, 8, 12, 16, 20, 24, 32]
    for n_vars in sizes:
        struct_name = f'random_dag_{n_vars}v_d03'
        structure = make_random_dag_structure(n_vars, density=0.3)
        
        # Adjust n_runs based on problem size
        n_runs = 40 if n_vars <= 16 else 25
        
        print(f"\nBenchmarking: {struct_name}")
        result = benchmark_static_on_structure(structure, struct_name, n_runs=n_runs)
        result['density'] = 0.3
        results.append(result)
    
    # Test 2: Varying DAG density with fixed size
    print("\n" + "=" * 80)
    print("Test 2: Varying DAG Density (size=20)")
    print("=" * 80)
    
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_vars = 20
    for density in densities:
        struct_name = f'random_dag_{n_vars}v_d{int(density*100):02d}'
        structure = make_random_dag_structure(n_vars, density=density)
        
        print(f"\nBenchmarking: {struct_name}")
        result = benchmark_static_on_structure(structure, struct_name, n_runs=25)
        result['density'] = density
        results.append(result)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving Results...")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'benchmark_static_scaling_results.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Mean Speedup:   {df['speedup'].mean():.3f}x")
    print(f"Median Speedup: {df['speedup'].median():.3f}x")
    print(f"Std Dev:        {df['speedup'].std():.3f}")
    print(f"Min Speedup:    {df['speedup'].min():.3f}x ({df.loc[df['speedup'].idxmin(), 'structure']})")
    print(f"Max Speedup:    {df['speedup'].max():.3f}x ({df.loc[df['speedup'].idxmax(), 'structure']})")
    
    # Scaling analysis
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS (by DAG size, density=0.3)")
    print("=" * 80)
    
    df_size = df[df['density'] == 0.3].copy()
    if len(df_size) > 0:
        print("Block Count | Speedup Trend")
        print("-" * 40)
        for _, row in df_size.sort_values('n_blocks').iterrows():
            print(f"{row['n_blocks']:3d}         | {row['speedup']:.3f}x")

if __name__ == "__main__":
    main()
