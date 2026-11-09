import sys
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import fire
import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from pgm_paper_utils.utils import format_dist_str

# Add the code directory to path to allow importing project modules

from structflow.pgms.factories import make_random_pgm
from structflow.pgms.pgm_compiler import PGMCompiler

def plot_structure(ax, structure, title):
    """
    Plots a DAG structure using networkx.
    structure: Dict[str, List[str]] mapping child to parents.
    """
    G = nx.DiGraph()
    # Ensure all nodes are added even if they have no parents/children
    for child in structure.keys():
        G.add_node(child)
        
    for child, parents in structure.items():
        for parent in parents:
            G.add_edge(parent, child)
    
    # Use shell_layout or spring_layout for better visualization of small graphs
    pos = nx.shell_layout(G)
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lavender', 
            node_size=2000, arrowsize=20, font_size=14, font_weight='bold',
            edge_color='gray', width=2)
    
    ax.set_title(title, fontsize=16, pad=20)

def main(seed=42, n_nodes=6, sparsity=0.4):
    # 1. Initialize random PGM
    key = jax.random.PRNGKey(seed)
    
    print(f"Generating random PGM with {n_nodes} nodes and sparsity {sparsity}...")
    pgm = make_random_pgm(n_nodes, 'linear', key, sparsity=sparsity)
    pgm_structure = pgm.get_pgm_structure()
    
    # 2. Compile using PGMCompiler
    compiler = PGMCompiler(pgm_structure)
    
    # Causal compilation: Should match the original PGM structure 
    # but might be ordered topologically.
    print("Compiling causal structure...")
    causal_struct = compiler.compile_causal()
    causal_deps = causal_struct.dependencies
    
    # Inference compilation: p(roots | leaves)
    targets, conditions = pgm.get_inference_targets_and_conditions()
    print(f"Compiling inference structure for p({targets} | {conditions})...")
    inference_struct = compiler.compile_inference(targets, conditions)
    inference_deps = inference_struct.dependencies

    print("Tractable distributions:")
    print('Causal:')
    print([format_dist_str(targets=d['targets'], conditions=d['conditions']) for d in causal_struct.get_tractable_distributions()])
    print('Inference:')
    print([format_dist_str(targets=d['targets'], conditions=d['conditions']) for d in inference_struct.get_tractable_distributions()])

    
    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # plot_structure(axes[0], pgm_structure, "Original PGM DAG")
    plot_structure(axes[0], causal_deps, "Causal DAG")
    
    target_str = ", ".join(sorted(list(targets)))
    cond_str = ", ".join(sorted(list(conditions)))
    plot_structure(axes[1], inference_deps, f"Compiled Inference DAG\n$p({{{target_str}}} \mid {{{cond_str}}})$")
    
    plt.tight_layout()
    output_path = f"pgm_compilation_visualization_{seed}_{n_nodes}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
