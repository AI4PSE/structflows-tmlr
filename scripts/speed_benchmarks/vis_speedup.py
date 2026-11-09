import fire
import pandas as pd
import matplotlib.pyplot as plt
import os

def main(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run the benchmark script first.")
        return

    df = pd.read_csv(csv_path)

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Nodes vs Time
    # ax1.plot(df['k'], df['dynamic'], 'o-', label='Dynamic')
    ax1.plot(df['k'], df['static'], 'o-', label='Static')
    ax1.plot(df['k'], df['parallel'], 'o-', label='Dynamic')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Number of Nodes ($N$)')
    ax1.set_ylabel('Average Sampling Time (s)')
    ax1.set_title('Inference Time Scaling')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    # Subplot 2: Nodes vs Speedup Ratio
    ax2.plot(df['k'], df['parallel_speedup_vs_static'], 's-', color='green', label='Speedup Ratio')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Number of Nodes ($N$)')
    ax2.set_ylabel('Speedup Factor (Static / Dynamic)')
    ax2.set_title('Ratio of Speedup')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()
    
    # Add a horizontal line at 1.0 for reference
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    
    # Save results
    output_path = '../../images/chain_benchmark_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
