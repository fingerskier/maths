"""
Problem 4: Graph Theory Basics

Implement fundamental graph algorithms from scratch (no networkx).

Tasks:
  (a) Represent a graph using an adjacency list
  (b) Implement BFS and DFS traversals
  (c) Detect whether a graph contains a cycle
  (d) Check if a graph is bipartite (2-colorable)
  (e) Prove that a tree on n vertices has exactly n-1 edges (verify computationally)
"""

from collections import deque


class Graph:
    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.adj = {i: [] for i in range(n_vertices)}

    def add_edge(self, u, v):
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)

    def bfs(self, start):
        """
        Return the list of vertices in BFS order starting from `start`.
        """
        # TODO
        raise NotImplementedError

    def dfs(self, start):
        """
        Return the list of vertices in DFS order starting from `start`.
        """
        # TODO
        raise NotImplementedError

    def has_cycle(self):
        """
        Return True if the graph contains a cycle, False otherwise.
        Handle both directed and undirected cases.
        """
        # TODO
        raise NotImplementedError

    def is_bipartite(self):
        """
        Return True if the graph is bipartite (2-colorable).
        Use BFS-based coloring.
        """
        # TODO
        raise NotImplementedError

    def is_connected(self):
        """Return True if all vertices are reachable from vertex 0."""
        visited = set()
        stack = [0]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(self.adj[v])
        return len(visited) == self.n

    def num_edges(self):
        total = sum(len(neighbors) for neighbors in self.adj.values())
        return total if self.directed else total // 2


def build_sample_graph():
    """Build a sample undirected graph for testing."""
    g = Graph(6)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def build_tree(n):
    """Build a tree on n vertices (path graph)."""
    g = Graph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def verify_tree_property():
    """Verify that a tree on n vertices has exactly n-1 edges."""
    for n in range(2, 20):
        tree = build_tree(n)
        assert tree.num_edges() == n - 1, f"Failed for n={n}"
        assert tree.is_connected(), f"Tree not connected for n={n}"
        assert not tree.has_cycle(), f"Tree has cycle for n={n}"
    print("Tree property verified: n vertices, n-1 edges, connected, no cycle.")


if __name__ == "__main__":
    g = build_sample_graph()
    try:
        print("BFS from 0:", g.bfs(0))
        print("DFS from 0:", g.dfs(0))
        print("Has cycle:", g.has_cycle())
        print("Is bipartite:", g.is_bipartite())
    except NotImplementedError:
        print("TODO: implement graph methods")

    try:
        verify_tree_property()
    except NotImplementedError:
        print("TODO: implement has_cycle() for tree verification")
