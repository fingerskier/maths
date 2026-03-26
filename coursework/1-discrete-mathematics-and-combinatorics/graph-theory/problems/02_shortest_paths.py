"""
Problem: Shortest Path Algorithms

Implement classical shortest-path algorithms on weighted graphs using
adjacency list representation (dict of dicts with weights).

Tasks:
  (a) Implement BFS for unweighted shortest path (number of edges)
  (b) Implement Dijkstra's algorithm for non-negative edge weights
  (c) Implement Bellman-Ford for graphs that may have negative edge weights
  (d) Detect negative-weight cycles using Bellman-Ford
"""

from collections import deque
import math


def bfs_shortest_path(graph, source):
    """
    Compute shortest distances (by number of edges) from source to all
    reachable vertices using BFS.

    Parameters:
        graph: dict[node, dict[node, weight]], adjacency list with weights
               (weights are ignored for BFS)
        source: the starting node

    Returns:
        dist: dict[node, int], shortest edge-count distance from source
              (unreachable nodes map to math.inf)
        parent: dict[node, node or None], predecessor map for path reconstruction
    """
    # TODO
    raise NotImplementedError


def dijkstra(graph, source):
    """
    Compute shortest distances from source to all vertices using Dijkstra's
    algorithm. All edge weights must be non-negative.

    Parameters:
        graph: dict[node, dict[node, weight]], adjacency list
        source: the starting node

    Returns:
        dist: dict[node, float], shortest weighted distance from source
        parent: dict[node, node or None], predecessor map for path reconstruction
    """
    # TODO
    raise NotImplementedError


def bellman_ford(graph, source):
    """
    Compute shortest distances from source using Bellman-Ford.
    Works with negative edge weights.

    Parameters:
        graph: dict[node, dict[node, weight]], adjacency list
        source: the starting node

    Returns:
        dist: dict[node, float], shortest distance from source
        parent: dict[node, node or None], predecessor map
        has_negative_cycle: bool, True if a negative-weight cycle is reachable
                            from source
    """
    # TODO
    raise NotImplementedError


def reconstruct_path(parent, source, target):
    """
    Reconstruct the shortest path from source to target using the parent map.

    Parameters:
        parent: dict[node, node or None]
        source: starting node
        target: ending node

    Returns:
        list[node]: path from source to target, or empty list if unreachable
    """
    # TODO
    raise NotImplementedError


def build_sample_graph():
    """Build a sample weighted directed graph for testing."""
    return {
        'A': {'B': 4, 'C': 2},
        'B': {'C': 1, 'D': 5},
        'C': {'D': 8, 'E': 10},
        'D': {'E': 2},
        'E': {}
    }


def build_negative_weight_graph():
    """Build a graph with negative edges (but no negative cycle)."""
    return {
        'A': {'B': 1, 'C': 4},
        'B': {'C': -2, 'D': 3},
        'C': {'D': 2},
        'D': {}
    }


def build_negative_cycle_graph():
    """Build a graph containing a negative-weight cycle."""
    return {
        'A': {'B': 1},
        'B': {'C': -1},
        'C': {'A': -1, 'D': 2},
        'D': {}
    }


if __name__ == "__main__":
    g = build_sample_graph()

    try:
        dist, parent = bfs_shortest_path(g, 'A')
        print("BFS distances from A:", dist)
        path = reconstruct_path(parent, 'A', 'E')
        print("BFS path A -> E:", path)
    except NotImplementedError:
        print("TODO: implement bfs_shortest_path")

    try:
        dist, parent = dijkstra(g, 'A')
        print("\nDijkstra distances from A:", dist)
        path = reconstruct_path(parent, 'A', 'E')
        print("Dijkstra path A -> E:", path)
    except NotImplementedError:
        print("TODO: implement dijkstra")

    try:
        g_neg = build_negative_weight_graph()
        dist, parent, neg_cycle = bellman_ford(g_neg, 'A')
        print("\nBellman-Ford distances from A:", dist)
        print("Negative cycle detected:", neg_cycle)

        g_neg_cycle = build_negative_cycle_graph()
        dist2, parent2, neg_cycle2 = bellman_ford(g_neg_cycle, 'A')
        print("\nNegative cycle graph - cycle detected:", neg_cycle2)
    except NotImplementedError:
        print("TODO: implement bellman_ford")
