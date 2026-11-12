"""Verify and visualize the worked example of Appendix "Worked Example:
Compiling a Dense Four-Node Graph" (reviewer jqSe's graph).

Graph: Z1 -> X, Z1 -> Y, Z1 -> Z2, X -> Y, X -> Z2, Z2 -> Y
Queries: p(z2 | x, y) (inference alignment) and p(y | do(x)) (causal alignment).

Run from anywhere:
    python scripts/plotters/vis_worked_example.py
"""
from pgm_paper_utils.paths import IMAGES
import sys
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"


import matplotlib.pyplot as plt
import networkx as nx

from structflow.pgms.pgm_compiler import PGMCompiler

PGM_DAG = {
    "z1": [],
    "x": ["z1"],
    "z2": ["z1", "x"],
    "y": ["z1", "x", "z2"],
}

LABELS = {"z1": "$Z_1$", "x": "$X$", "z2": "$Z_2$", "y": "$Y$"}
POS = {"z1": (0.0, 1.0), "x": (-1.1, 0.0), "z2": (1.1, 0.0), "y": (0.0, -1.0)}

COL_DEFAULT = "lavender"
COL_CONDITION = "lightgray"
COL_TARGET = "gold"
COL_CLAMPED = "lightcoral"


def verify():
    compiler = PGMCompiler(PGM_DAG)

    causal = compiler.compile_causal()
    print("Causal order:       ", causal.block_order)
    print("Causal dependencies:", causal.dependencies)
    assert causal.block_order == ["z1", "x", "z2", "y"]
    assert causal.dependencies == PGM_DAG

    inference = compiler.compile_inference(targets={"z2"}, conditions={"x", "y"})
    print("Inference order:       ", inference.block_order)
    print("Inference dependencies:", inference.dependencies)
    assert inference.block_order == ["y", "x", "z2", "z1"]
    assert inference.dependencies == {
        "y": [],
        "x": ["y"],
        "z2": ["y", "x"],
        "z1": ["y", "x", "z2"],
    }

    # p(z2 | x, y): tractable on the compiled structure, needs only the z2
    # block (z1 is never sampled); intractable on the causal structure.
    assert inference.is_tractable_query({"z2"}, {"x", "y"})
    assert inference.get_required_simulation_blocks({"z2"}, {"x", "y"}) == ["z2"]
    assert not causal.is_tractable_query({"z2"}, {"x", "y"})

    # p(y | do(x)) on the causal structure: clamp x, simulate z1 (from its
    # prior), z2, and y.
    assert causal.get_required_simulation_blocks({"y"}, {"x"}) == ["z1", "z2", "y"]

    print("All assertions passed: code matches the worked example.")
    return causal, inference


def draw_graph(ax, edges, colors, directed=True, removed_edges=(),
               order=None, title=""):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(PGM_DAG.keys())
    G.add_edges_from(edges)

    node_colors = [colors.get(n, COL_DEFAULT) for n in G.nodes()]
    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=node_colors,
                           node_size=1800, edgecolors="dimgray")
    nx.draw_networkx_labels(G, POS, labels=LABELS, ax=ax, font_size=15)
    nx.draw_networkx_edges(G, POS, ax=ax, edge_color="gray", width=2,
                           arrows=directed, arrowsize=16, node_size=1800)
    if removed_edges:
        nx.draw_networkx_edges(nx.DiGraph(list(removed_edges)), POS, ax=ax,
                               edge_color="red", style="dashed", width=2,
                               arrows=True, arrowsize=16, node_size=1800)
    if order is not None:
        for i, node in enumerate(order):
            x, y = POS[node]
            ax.text(x + 0.28, y + 0.28, str(i + 1), fontsize=12,
                    fontweight="bold", color="tab:blue",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle", fc="white", ec="tab:blue"))
    ax.set_title(title, fontsize=15)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")


def main():
    causal, inference = verify()

    dag_edges = [(p, c) for c, ps in PGM_DAG.items() for p in ps]
    moral_edges = [tuple(sorted(e)) for e in dag_edges]  # no new edges here
    inf_edges = [(p, c) for c, ps in inference.dependencies.items() for p in ps]
    do_edges = [(p, c) for (p, c) in dag_edges if c != "x"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))  # printed at \textwidth (6.5in): fonts ~8pt on the page

    draw_graph(axes[0], dag_edges, {}, directed=True,
               title="(a) Causal DAG $\\mathcal{G}$")
    draw_graph(axes[1], sorted(set(moral_edges)), {}, directed=False,
               title="(b) Moral graph $\\mathcal{G}_m$\n(no edges added)")
    draw_graph(axes[2], inf_edges,
               {"x": COL_CONDITION, "y": COL_CONDITION, "z2": COL_TARGET},
               directed=True, order=inference.block_order,
               title="(c) Inference flow for $p(z_2 \\mid x, y)$\n"
                     "order $[Y, X, Z_2, Z_1]$")
    draw_graph(axes[3], do_edges, {"x": COL_CLAMPED}, directed=True,
               removed_edges=[("z1", "x")], order=causal.block_order,
               title="(d) Causal flow, $\\mathrm{do}(x)$:\n"
                     "clamp $X$, sever $Z_1 \\to X$")

    plt.tight_layout()
    out = f"{IMAGES}/worked_example_compilation.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
