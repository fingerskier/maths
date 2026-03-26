"""
Problem: Trees and Spanning Trees

Implement tree-detection and minimum spanning tree algorithms.

Tasks:
  (a) Check if a graph is a tree (connected and acyclic)
  (b) Implement Kruskal's MST algorithm using union-find
  (c) Implement Prim's MST algorithm
  (d) Count spanning trees of a small graph using Kirchhoff's theorem
      (determinant of any cofactor of the Laplacian matrix)
"""

import math


def is_tree(n_vertices, edges):
    """
    Check whether an undirected graph is a tree.
    A tree is connected and has exactly n-1 edges (equivalently, connected
    and acyclic).

    Parameters:
        n_vertices: int, number of vertices (labeled 0..n-1)
        edges: list[tuple[int, int]], undirected edges

    Returns:
        bool: True if the graph is a tree
    """
    # TODO
    raise NotImplementedError


class UnionFind:
    """Disjoint-set (union-find) data structure with union by rank and
    path compression."""

    def __init__(self, n):
        """Initialize n singleton sets."""
        # TODO
        raise NotImplementedError

    def find(self, x):
        """Find the representative of the set containing x."""
        # TODO
        raise NotImplementedError

    def union(self, x, y):
        """
        Merge the sets containing x and y.

        Returns:
            bool: True if x and y were in different sets (merge performed),
                  False if already in the same set
        """
        # TODO
        raise NotImplementedError


def kruskal_mst(n_vertices, weighted_edges):
    """
    Compute the minimum spanning tree using Kruskal's algorithm.

    Parameters:
        n_vertices: int, number of vertices (labeled 0..n-1)
        weighted_edges: list[tuple[float, int, int]], edges as (weight, u, v)

    Returns:
        mst_edges: list[tuple[float, int, int]], edges in the MST
        total_weight: float, sum of edge weights in the MST
    """
    # TODO
    raise NotImplementedError


def prim_mst(n_vertices, adj):
    """
    Compute the minimum spanning tree using Prim's algorithm.

    Parameters:
        n_vertices: int, number of vertices (labeled 0..n-1)
        adj: dict[int, list[tuple[int, float]]], adjacency list mapping
             vertex -> [(neighbor, weight), ...]

    Returns:
        mst_edges: list[tuple[int, int, float]], edges in the MST as (u, v, w)
        total_weight: float, sum of edge weights in the MST
    """
    # TODO
    raise NotImplementedError


def count_spanning_trees(n_vertices, edges):
    """
    Count the number of spanning trees using Kirchhoff's theorem.
    Compute the Laplacian matrix L, then return the determinant of any
    (n-1) x (n-1) cofactor of L.

    Only practical for small graphs.

    Parameters:
        n_vertices: int
        edges: list[tuple[int, int]], undirected edges

    Returns:
        int: number of distinct spanning trees
    """
    # TODO
    raise NotImplementedError


def build_sample_weighted_graph():
    """Build a sample weighted undirected graph for MST testing."""
    # Edges: (weight, u, v)
    edges = [
        (1, 0, 1), (4, 0, 2), (3, 1, 2),
        (2, 1, 3), (5, 2, 3), (7, 2, 4),
        (6, 3, 4)
    ]
    adj = {i: [] for i in range(5)}
    for w, u, v in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return 5, edges, adj


if __name__ == "__main__":
    try:
        # A path graph on 4 vertices is a tree
        print("Path graph is tree:", is_tree(4, [(0, 1), (1, 2), (2, 3)]))
        # Adding an edge creates a cycle
        print("Cycle graph is tree:", is_tree(4, [(0, 1), (1, 2), (2, 3), (3, 0)]))
        # Disconnected graph
        print("Disconnected is tree:", is_tree(4, [(0, 1), (2, 3)]))
    except NotImplementedError:
        print("TODO: implement is_tree")

    n, edges, adj = build_sample_weighted_graph()

    try:
        mst_edges, total = kruskal_mst(n, edges)
        print(f"\nKruskal MST: edges={mst_edges}, total weight={total}")
    except NotImplementedError:
        print("TODO: implement kruskal_mst")

    try:
        mst_edges, total = prim_mst(n, adj)
        print(f"Prim MST: edges={mst_edges}, total weight={total}")
    except NotImplementedError:
        print("TODO: implement prim_mst")

    try:
        # Complete graph K4 has 4^2 = 16 spanning trees
        k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        count = count_spanning_trees(4, k4_edges)
        print(f"\nSpanning trees of K4: {count} (expected 16)")
    except NotImplementedError:
        print("TODO: implement count_spanning_trees")
