"""
Tests for Problem 1: Set Theory — Operations and Identities

Run with:  pytest coursework/0-foundations-of-mathematics/ -v
"""

import pytest

LEVEL = "0-foundations-of-mathematics"
TOPIC = "sets-and-number-theory"


@pytest.fixture
def mod(problem_loader):
    return problem_loader(LEVEL, TOPIC, "01_set_operations.py")


@pytest.mark.level0
class TestSetOperations:
    def test_union_basic(self, mod):
        assert mod.union([1, 2, 3], [3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_union_disjoint(self, mod):
        assert mod.union([1, 2], [3, 4]) == [1, 2, 3, 4]

    def test_union_empty(self, mod):
        assert mod.union([], [1, 2, 3]) == [1, 2, 3]
        assert mod.union([1, 2], []) == [1, 2]

    def test_intersection_basic(self, mod):
        assert mod.intersection([1, 2, 3], [2, 3, 4]) == [2, 3]

    def test_intersection_disjoint(self, mod):
        assert mod.intersection([1, 2], [3, 4]) == []

    def test_difference_basic(self, mod):
        assert mod.difference([1, 2, 3, 4], [2, 4]) == [1, 3]

    def test_difference_no_overlap(self, mod):
        assert mod.difference([1, 2], [3, 4]) == [1, 2]

    def test_symmetric_difference(self, mod):
        assert mod.symmetric_difference([1, 2, 3], [3, 4, 5]) == [1, 2, 4, 5]

    def test_complement(self, mod):
        assert mod.complement([0, 2, 4, 6, 8]) == [1, 3, 5, 7, 9]
        assert mod.complement([]) == list(range(10))

    def test_de_morgan_law_1(self, mod):
        A, B = [1, 2, 3], [3, 4, 5]
        lhs = mod.complement(mod.union(A, B))
        rhs = mod.intersection(mod.complement(A), mod.complement(B))
        assert lhs == rhs

    def test_de_morgan_law_2(self, mod):
        A, B = [1, 2, 3], [3, 4, 5]
        lhs = mod.complement(mod.intersection(A, B))
        rhs = mod.union(mod.complement(A), mod.complement(B))
        assert lhs == rhs
