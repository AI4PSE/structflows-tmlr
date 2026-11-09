"""
Visualization script for scaling benchmark results
Plots scaling analysis for both static and parallel implementations
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_scaling_results():
    """Load scaling benchmark results from CSV files."""
    script_dir = os.path.dirname(__file__)
    
    static_path = os.path.join(script_dir, 'benchmark_static_scaling_results.csv')
    parallel_path = os.path.join(script_dir, 'benchmark_parallel_scaling_results.csv')
    
    df_static = pd.read_csv(static_path)
    df_parallel = pd.read_csv(parallel_path)
    
    return df_static, df_parallel

def plot_scaling_by_size(df_static, df_parallel):
    """Plot speedup scaling as DAG size increases (fixed density)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Static implementations - size scaling
    ax = axes[0]
    df_size = df_static[df_static['density'] == 0.3].copy()
    df_size_sorted = df_size.sort_values('n_blocks')
    
    ax.plot(df_size_sorted['n_blocks'], df_size_sorted['speedup'],
            marker='o', linewidth=2.5, markersize=10, color='#FF6B6B', 
            label='StructuredVelocityFn', zorder=3)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    
    ax.fill_between(df_size_sorted['n_blocks'], 1.0, df_size_sorted['speedup'], 
                     alpha=0.2, color='#FF6B6B')
    
    ax.set_xlabel('Number of Blocks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Simple / Struct)', fontsize=12, fontweight='bold')
    ax.set_title('Static: Scaling with DAG Size (density=0.3)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_ylim([0.7, 1.3])
    
    # Add value labels
    for _, row in df_size_sorted.iterrows():
        ax.text(row['n_blocks'], row['speedup'] + 0.02, f"{row['speedup']:.2f}x",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Parallel implementations - size scaling
    ax = axes[1]
    df_size = df_parallel[df_parallel['density'] == 0.3].copy()
    df_size_sorted = df_size.sort_values('n_blocks')
    
    ax.plot(df_size_sorted['n_blocks'], df_size_sorted['speedup'],
            marker='s', linewidth=2.5, markersize=10, color='#4ECDC4',
            label='ParallelVelocityFn (Sparse)', zorder=3)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    
    ax.fill_between(df_size_sorted['n_blocks'], 1.0, df_size_sorted['speedup'],
                     alpha=0.2, color='#4ECDC4')
    
    ax.set_xlabel('Number of Blocks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Dense / Sparse)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel: Scaling with DAG Size (density=0.3)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_ylim([0.7, 1.5])
    
    # Add value labels
    for _, row in df_size_sorted.iterrows():
        ax.text(row['n_blocks'], row['speedup'] + 0.03, f"{row['speedup']:.2f}x",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_sparsity_impact(df_static, df_parallel):
    """Plot speedup impact with varying DAG density (fixed size)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Static implementations - density impact
    ax = axes[0]
    df_dens = df_static[df_static['n_blocks'] == 20].copy()
    df_dens_sorted = df_dens.sort_values('density')
    
    densities_pct = df_dens_sorted['density'].values * 100
    ax.plot(densities_pct, df_dens_sorted['speedup'],
            marker='D', linewidth=2.5, markersize=10, color='#FFB86C',
            label='StructuredVelocityFn', zorder=3)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    
    ax.fill_between(densities_pct, 1.0, df_dens_sorted['speedup'],
                     alpha=0.2, color='#FFB86C')
    
    ax.set_xlabel('DAG Edge Density (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Simple / Struct)', fontsize=12, fontweight='bold')
    ax.set_title('Static: Sparsity Impact (20 blocks)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_ylim([0.8, 1.25])
    ax.set_xticks(densities_pct)
    
    # Add value labels
    for x, y in zip(densities_pct, df_dens_sorted['speedup']):
        ax.text(x, y + 0.02, f"{y:.2f}x", ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    # Parallel implementations - density impact
    ax = axes[1]
    df_dens = df_parallel[df_parallel['n_blocks'] == 20].copy()
    df_dens_sorted = df_dens.sort_values('density')
    
    densities_pct = df_dens_sorted['density'].values * 100
    ax.plot(densities_pct, df_dens_sorted['speedup'],
            marker='D', linewidth=2.5, markersize=10, color='#A6E22E',
            label='ParallelVelocityFn (Sparse)', zorder=3)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    
    ax.fill_between(densities_pct, 1.0, df_dens_sorted['speedup'],
                     alpha=0.2, color='#A6E22E')
    
    ax.set_xlabel('DAG Edge Density (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Dense / Sparse)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel: Sparsity Impact (20 blocks)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.set_ylim([0.8, 1.25])
    ax.set_xticks(densities_pct)
    
    # Add value labels
    for x, y in zip(densities_pct, df_dens_sorted['speedup']):
        ax.text(x, y + 0.02, f"{y:.2f}x", ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_comparison_summary(df_static, df_parallel):
    """Plot overall comparison between implementations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Static size-scaling times
    ax = axes[0, 0]
    df_size = df_static[df_static['density'] == 0.3].copy().sort_values('n_blocks')
    width = 1.0
    x_pos = np.arange(len(df_size))
    
    ax.bar(x_pos - width/2, df_size['simple_time']*1000, width/2, 
           label='Simple', alpha=0.8, color='#FF6B6B')
    ax.bar(x_pos + width/2, df_size['struct_time']*1000, width/2,
           label='Structured', alpha=0.8, color='#4ECDC4')
    
    ax.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Static: Absolute Times by Size', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_size['n_blocks'].values)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Parallel size-scaling times
    ax = axes[0, 1]
    df_size = df_parallel[df_parallel['density'] == 0.3].copy().sort_values('n_blocks')
    x_pos = np.arange(len(df_size))
    
    ax.bar(x_pos - width/2, df_size['dense_time']*1000, width/2,
           label='Dense', alpha=0.8, color='#95E1D3')
    ax.bar(x_pos + width/2, df_size['sparse_time']*1000, width/2,
           label='Sparse', alpha=0.8, color='#F38181')
    
    ax.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Parallel: Absolute Times by Size', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_size['n_blocks'].values)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Static density-scaling times
    ax = axes[1, 0]
    df_dens = df_static[df_static['n_blocks'] == 20].copy().sort_values('density')
    x_pos = np.arange(len(df_dens))
    densities_pct = (df_dens['density'].values * 100).astype(int)
    
    ax.bar(x_pos - width/2, df_dens['simple_time']*1000, width/2,
           label='Simple', alpha=0.8, color='#FF6B6B')
    ax.bar(x_pos + width/2, df_dens['struct_time']*1000, width/2,
           label='Structured', alpha=0.8, color='#4ECDC4')
    
    ax.set_xlabel('DAG Density (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Static: Absolute Times by Density (20 blocks)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(densities_pct)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Parallel density-scaling times
    ax = axes[1, 1]
    df_dens = df_parallel[df_parallel['n_blocks'] == 20].copy().sort_values('density')
    x_pos = np.arange(len(df_dens))
    densities_pct = (df_dens['density'].values * 100).astype(int)
    
    ax.bar(x_pos - width/2, df_dens['dense_time']*1000, width/2,
           label='Dense', alpha=0.8, color='#95E1D3')
    ax.bar(x_pos + width/2, df_dens['sparse_time']*1000, width/2,
           label='Sparse', alpha=0.8, color='#F38181')
    
    ax.set_xlabel('DAG Density (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Parallel: Absolute Times by Density (20 blocks)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(densities_pct)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main execution."""
    # Load results
    df_static, df_parallel = load_scaling_results()
    
    print("Loaded scaling benchmark results:")
    print(f"  Static implementations: {len(df_static)} structures")
    print(f"  Parallel implementations: {len(df_parallel)} structures")
    
    # Create plots
    fig1 = plot_scaling_by_size(df_static, df_parallel)
    fig2 = plot_sparsity_impact(df_static, df_parallel)
    fig3 = plot_comparison_summary(df_static, df_parallel)
    
    # Save plots
    script_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(script_dir, '..', 'images', 'benchmark_results')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot1_path = os.path.join(plots_dir, 'scaling_by_size.png')
    plot2_path = os.path.join(plots_dir, 'scaling_by_sparsity.png')
    plot3_path = os.path.join(plots_dir, 'scaling_comparison.png')
    
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot1_path}")
    
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot2_path}")
    
    fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot3_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nStatic Implementations (StructuredVelocityFn scaling):")
    print(f"  Mean Speedup:    {df_static['speedup'].mean():.3f}x")
    print(f"  Median Speedup:  {df_static['speedup'].median():.3f}x")
    print(f"  Std Dev:         {df_static['speedup'].std():.3f}")
    print(f"  Min Speedup:     {df_static['speedup'].min():.3f}x")
    print(f"  Max Speedup:     {df_static['speedup'].max():.3f}x")
    
    print("\nParallel Implementations (Sparse matrix speedup):")
    print(f"  Mean Speedup:    {df_parallel['speedup'].mean():.3f}x")
    print(f"  Median Speedup:  {df_parallel['speedup'].median():.3f}x")
    print(f"  Std Dev:         {df_parallel['speedup'].std():.3f}")
    print(f"  Min Speedup:     {df_parallel['speedup'].min():.3f}x")
    print(f"  Max Speedup:     {df_parallel['speedup'].max():.3f}x")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    
    # Analyze scaling trends
    df_static_size = df_static[df_static['density'] == 0.3].sort_values('n_blocks')
    if len(df_static_size) >= 2:
        trend_static = "increasing" if df_static_size['speedup'].iloc[-1] > df_static_size['speedup'].iloc[0] else "decreasing"
        print(f"Static scaling with size: {trend_static} ({df_static_size['speedup'].iloc[0]:.2f}x → {df_static_size['speedup'].iloc[-1]:.2f}x)")
    
    df_parallel_size = df_parallel[df_parallel['density'] == 0.3].sort_values('n_blocks')
    if len(df_parallel_size) >= 2:
        print(f"Parallel scaling with size: variable (peak {df_parallel_size['speedup'].max():.2f}x at {df_parallel_size.loc[df_parallel_size['speedup'].idxmax(), 'n_blocks']:.0f} blocks)")
    
    # Density analysis
    df_static_dens = df_static[df_static['n_blocks'] == 20].sort_values('density')
    if len(df_static_dens) >= 2:
        dens_mean = df_static_dens['speedup'].mean()
        print(f"Static speedup independence from density: mean={dens_mean:.2f}x (σ={df_static_dens['speedup'].std():.3f})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
