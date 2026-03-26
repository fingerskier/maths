"""
Problem 1: Set Theory — Operations and Identities

Tasks:
  (a) Implement union, intersection, difference, and symmetric difference
      for sets represented as sorted lists (without using Python's built-in set)
  (b) Verify De Morgan's Laws:
        complement(A union B) = complement(A) intersect complement(B)
        complement(A intersect B) = complement(A) union complement(B)
  (c) Prove by exhaustive check that for finite sets A, B, C:
        A intersect (B union C) = (A intersect B) union (A intersect C)

Use a universal set U = {0, 1, 2, ..., 9} for complements.
"""

U = list(range(10))


def union(A, B):
    """Return the union of sorted lists A and B as a sorted list."""
    # TODO: implement without using Python set()
    raise NotImplementedError


def intersection(A, B):
    """Return the intersection of sorted lists A and B."""
    # TODO
    raise NotImplementedError


def difference(A, B):
    """Return A - B (elements in A but not in B)."""
    # TODO
    raise NotImplementedError


def symmetric_difference(A, B):
    """Return the symmetric difference of A and B."""
    # TODO
    raise NotImplementedError


def complement(A, universal=U):
    """Return the complement of A with respect to the universal set."""
    # TODO
    raise NotImplementedError


def verify_de_morgan():
    """
    Test De Morgan's Laws for several pairs of sets.
    Return True if all tests pass.
    """
    test_cases = [
        ([1, 2, 3], [3, 4, 5]),
        ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
        ([], [1, 2, 3]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5]),
    ]
    for A, B in test_cases:
        # Law 1: complement(A union B) == complement(A) intersect complement(B)
        lhs1 = complement(union(A, B))
        rhs1 = intersection(complement(A), complement(B))
        assert lhs1 == rhs1, f"De Morgan 1 failed for {A}, {B}"

        # Law 2: complement(A intersect B) == complement(A) union complement(B)
        lhs2 = complement(intersection(A, B))
        rhs2 = union(complement(A), complement(B))
        assert lhs2 == rhs2, f"De Morgan 2 failed for {A}, {B}"

    print("De Morgan's Laws verified for all test cases.")
    return True


def verify_distributive():
    """Verify A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) for sample sets."""
    from itertools import combinations

    elements = list(range(5))
    # test all possible triples of subsets of {0,1,2,3,4}
    from itertools import chain, combinations as combs

    def powerset(s):
        return list(chain.from_iterable(combs(s, r) for r in range(len(s) + 1)))

    all_sets = [sorted(list(s)) for s in powerset(elements)]
    count = 0
    for A in all_sets:
        for B in all_sets:
            for C in all_sets:
                lhs = intersection(A, union(B, C))
                rhs = union(intersection(A, B), intersection(A, C))
                assert lhs == rhs, f"Distributive law failed for {A}, {B}, {C}"
                count += 1
    print(f"Distributive law verified for {count} triples of subsets.")
    return True


if __name__ == "__main__":
    try:
        verify_de_morgan()
        verify_distributive()
    except NotImplementedError:
        print("TODO: implement set operations first")
