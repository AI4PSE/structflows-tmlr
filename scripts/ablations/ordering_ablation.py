"""Ordering-vs-d-separation ablation (compute-only; reviewer vwa6, critical change 2).

The compiler's dependency construction is pure d-separation on the original DAG
given a chosen block order; moralization enters only through the BFS that picks
the order. This script separates the two contributions: for each graph we
compile the leaf-conditioned posterior under
  (a) moral-BFS ordering (the compiler as shipped),
  (b) no-moralization ordering (conditions first, remainder in reverse
      topological order; same d-separation dependency step),
  (c) worst-case check: random valid orderings (conditions still a prefix).
and report compiled edge counts + whether the collider co-parent edge appears.

Run: python scripts/ablations/ordering_ablation.py
"""
import sys, itertools, random as pyrandom

import numpy as np
import networkx as nx
from jax import random
from structflow.pgms.pgm_compiler import compile_inference_structure
from structflow.pgms.structure import BlockTriangularStructure
from pgm_paper_utils.pgm_factory import create_pgm


def deps_for_order(pgm_dag, block_order):
    """The compiler's dependency step, verbatim: d-separation on the original
    DAG, candidates = all predecessors in the given order."""
    G = nx.DiGraph()
    for node, parents in pgm_dag.items():
        G.add_node(node)
        for p in parents:
            G.add_edge(p, node)
    deps = {}
    for i, block in enumerate(block_order):
        preds = set(block_order[:i])
        deps[block] = [p for p in preds
                       if not nx.is_d_separator(G, {block}, {p}, preds - {p})]
    return deps


def n_edges(deps):
    return sum(len(v) for v in deps.values())


def orders(pgm_dag, conditions):
    """(moral-BFS order via compiler, no-moralization order)."""
    st = BlockTriangularStructure(
        dependencies=pgm_dag,
        block_order=list(nx.topological_sort(nx.DiGraph(
            [(p, c) for c, ps in pgm_dag.items() for p in ps]).to_directed()))
        if False else None,
        dims={k: 1 for k in pgm_dag})
    return st


def main():
    rows = []
    graphs = []
    # collider
    graphs.append(('collider', {'A': [], 'B': [], 'C': ['A', 'B']}, {'C'}))
    # chains
    for n in (4, 8):
        deps = {f'x{i}': ([f'x{i-1}'] if i else []) for i in range(n)}
        graphs.append((f'chain{n}', deps, {f'x{n-1}'}))
    # random pgms from the paper's generator
    for seed in range(5):
        for n in (4, 6, 8):
            cfg = dict(pgm_type='random', mechanism_type='linear',
                       n_nodes=n, node_dim=1, sparsity=0.3)
            pgm = create_pgm('random', cfg, random.PRNGKey(seed))
            deps = {k: list(v) for k, v in pgm.structure.dependencies.items()}
            sinks = {k for k in deps if not any(k in v for v in deps.values())}
            graphs.append((f'rand-n{n}-s{seed}', deps, sinks))

    print(f"{'graph':14s} {'|E| moral-BFS':>13s} {'|E| no-moral':>12s} "
          f"{'identical':>9s} {'rand orders (min/med/max)':>26s}")
    agg = {'same': 0, 'total': 0, 'moral_edges': [], 'nomoral_edges': [],
           'rand_edges': []}
    for name, dag, conditions in graphs:
        dims = {k: 1 for k in dag}
        causal = BlockTriangularStructure(
            dependencies=dag,
            block_order=[str(x) for x in nx.topological_sort(
                nx.DiGraph([(p, c) for c, ps in dag.items() for p in ps]
                           ).to_undirected().to_directed())]
            if False else sorted(dag, key=lambda k: 0),
            dims=dims)
        # proper topological order for the causal structure
        G = nx.DiGraph(); G.add_nodes_from(dag)
        for c, ps in dag.items():
            for p in ps:
                G.add_edge(p, c)
        causal = BlockTriangularStructure(
            dependencies=dag, block_order=list(nx.topological_sort(G)), dims=dims)
        targets = set(dag) - conditions

        # (a) compiler as shipped (moral-BFS ordering)
        st = compile_inference_structure(causal, targets=targets,
                                         conditions=conditions)
        e_moral = n_edges(st.dependencies)

        # (b) no-moralization ordering: conditions (reverse topo), then the
        # rest in reverse topological order; same d-separation step
        topo = list(nx.topological_sort(G))
        idx = {v: i for i, v in enumerate(topo)}
        order_nm = (sorted(conditions, key=lambda v: idx[v], reverse=True)
                    + sorted(set(dag) - conditions, key=lambda v: idx[v],
                             reverse=True))
        e_nomoral = n_edges(deps_for_order(dag, order_nm))

        # (c) random valid orderings, conditions kept as a prefix
        pyrandom.seed(0)
        rand_counts = []
        rest = list(set(dag) - conditions)
        for _ in range(20):
            pyrandom.shuffle(rest)
            order_r = sorted(conditions, key=lambda v: idx[v], reverse=True) + list(rest)
            rand_counts.append(n_edges(deps_for_order(dag, order_r)))
        same = st.block_order == order_nm and e_moral == e_nomoral
        agg['same'] += int(e_moral == e_nomoral); agg['total'] += 1
        agg['moral_edges'].append(e_moral); agg['nomoral_edges'].append(e_nomoral)
        agg['rand_edges'].append(np.median(rand_counts))
        print(f"{name:14s} {e_moral:13d} {e_nomoral:12d} "
              f"{str(e_moral == e_nomoral):>9s} "
              f"{min(rand_counts):>8d}/{int(np.median(rand_counts))}/{max(rand_counts)}")
        if name == 'collider':
            print(f"  collider check: compiled deps = {st.dependencies}; "
                  f"order = {st.block_order}")

    print(f"\nedge-count equal (moral-BFS vs no-moral): "
          f"{agg['same']}/{agg['total']} graphs")
    print(f"mean |E|: moral-BFS {np.mean(agg['moral_edges']):.2f}, "
          f"no-moral {np.mean(agg['nomoral_edges']):.2f}, "
          f"random-order median {np.mean(agg['rand_edges']):.2f}")


if __name__ == '__main__':
    main()
