"""
Problem: Lattices and Partially Ordered Sets (Posets)

Represent and analyze partially ordered sets and lattice structures.

Tasks:
  (a) Represent a poset as a relation (set of pairs) and verify the partial
      order axioms (reflexive, antisymmetric, transitive)
  (b) Find minimal and maximal elements of a poset
  (c) Check if a poset is a lattice (every pair of elements has a meet and join)
  (d) Compute the Hasse diagram edges (the transitive reduction of the relation)
  (e) Check if a lattice is distributive:
      a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c) for all a, b, c
"""


def is_partial_order(elements, relation):
    """
    Check whether a relation on a set is a partial order.

    A partial order is reflexive, antisymmetric, and transitive.

    Parameters:
        elements: set, the ground set
        relation: set[tuple], set of (a, b) pairs meaning a <= b

    Returns:
        dict with keys:
            'is_partial_order': bool
            'is_reflexive': bool
            'is_antisymmetric': bool
            'is_transitive': bool
    """
    # TODO
    raise NotImplementedError


def find_minimal_maximal(elements, relation):
    """
    Find the minimal and maximal elements of a poset.

    An element x is minimal if there is no y != x with (y, x) in relation.
    An element x is maximal if there is no y != x with (x, y) in relation.

    Parameters:
        elements: set
        relation: set[tuple], partial order relation

    Returns:
        dict with keys:
            'minimal': set, the minimal elements
            'maximal': set, the maximal elements
    """
    # TODO
    raise NotImplementedError


def meet(a, b, elements, relation):
    """
    Compute the meet (greatest lower bound) of elements a and b.

    Parameters:
        a, b: elements of the poset
        elements: set
        relation: set[tuple], partial order relation

    Returns:
        The meet of a and b, or None if it does not exist
    """
    # TODO
    raise NotImplementedError


def join(a, b, elements, relation):
    """
    Compute the join (least upper bound) of elements a and b.

    Parameters:
        a, b: elements of the poset
        elements: set
        relation: set[tuple], partial order relation

    Returns:
        The join of a and b, or None if it does not exist
    """
    # TODO
    raise NotImplementedError


def is_lattice(elements, relation):
    """
    Check if a poset is a lattice: every pair of elements has both a meet
    and a join.

    Parameters:
        elements: set
        relation: set[tuple], partial order relation

    Returns:
        bool: True if the poset is a lattice
    """
    # TODO
    raise NotImplementedError


def hasse_diagram(elements, relation):
    """
    Compute the Hasse diagram of a poset (the transitive reduction).

    Remove all edges (a, b) where a != b and there exists c with
    a < c < b. Also remove reflexive edges.

    Parameters:
        elements: set
        relation: set[tuple], partial order relation

    Returns:
        set[tuple]: the covering relations (edges of the Hasse diagram)
    """
    # TODO
    raise NotImplementedError


def is_distributive_lattice(elements, relation):
    """
    Check if a lattice is distributive:
    a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c) for all a, b, c.

    Assumes the poset is already verified to be a lattice.

    Parameters:
        elements: set
        relation: set[tuple], partial order relation

    Returns:
        bool: True if the lattice is distributive
    """
    # TODO
    raise NotImplementedError


def build_divisibility_poset(n):
    """
    Build the divisibility poset on divisors of n.

    Elements are all positive divisors of n.
    Relation: (a, b) if a divides b.

    Parameters:
        n: int, positive integer

    Returns:
        elements: set[int]
        relation: set[tuple[int, int]]
    """
    divisors = {i for i in range(1, n + 1) if n % i == 0}
    relation = {(a, b) for a in divisors for b in divisors if b % a == 0}
    return divisors, relation


if __name__ == "__main__":
    # Divisibility poset of 12: {1, 2, 3, 4, 6, 12}
    elements, relation = build_divisibility_poset(12)
    print(f"Divisors of 12: {sorted(elements)}")

    try:
        result = is_partial_order(elements, relation)
        print(f"Is partial order: {result}")
    except NotImplementedError:
        print("TODO: implement is_partial_order")

    try:
        result = find_minimal_maximal(elements, relation)
        print(f"Minimal elements: {result['minimal']}")
        print(f"Maximal elements: {result['maximal']}")
    except NotImplementedError:
        print("TODO: implement find_minimal_maximal")

    try:
        m = meet(4, 6, elements, relation)
        j = join(4, 6, elements, relation)
        print(f"\nmeet(4, 6) = {m} (expected 2)")
        print(f"join(4, 6) = {j} (expected 12)")
    except NotImplementedError:
        print("TODO: implement meet/join")

    try:
        print(f"\nIs lattice: {is_lattice(elements, relation)}")
    except NotImplementedError:
        print("TODO: implement is_lattice")

    try:
        hasse = hasse_diagram(elements, relation)
        print(f"Hasse diagram edges: {sorted(hasse)}")
    except NotImplementedError:
        print("TODO: implement hasse_diagram")

    try:
        print(f"Is distributive: {is_distributive_lattice(elements, relation)}")
    except NotImplementedError:
        print("TODO: implement is_distributive_lattice")
