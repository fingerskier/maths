"""
Problem 5: Relations and Equivalences

A relation on a set S is represented as a set of pairs (a, b) where a, b in S.

Tasks:
  (a) Check if a relation is reflexive, symmetric, or transitive
  (b) Determine if a relation is an equivalence relation
      (reflexive + symmetric + transitive)
  (c) Compute equivalence classes of an equivalence relation
  (d) Check if a relation is a partial order
      (reflexive + antisymmetric + transitive)
  (e) Compute the transitive closure of a relation
"""


# ---------------------------------------------------------------------------
# Part (a): Properties of relations
# ---------------------------------------------------------------------------

def is_reflexive(S, R):
    """
    Check if relation R on set S is reflexive.

    A relation is reflexive if (a, a) in R for every a in S.

    Args:
        S: set of elements
        R: set of (a, b) tuples representing the relation

    Returns:
        True if R is reflexive on S.
    """
    raise NotImplementedError


def is_symmetric(R):
    """
    Check if relation R is symmetric.

    A relation is symmetric if (a, b) in R implies (b, a) in R.

    Args:
        R: set of (a, b) tuples

    Returns:
        True if R is symmetric.
    """
    raise NotImplementedError


def is_transitive(R):
    """
    Check if relation R is transitive.

    A relation is transitive if (a, b) in R and (b, c) in R implies
    (a, c) in R.

    Args:
        R: set of (a, b) tuples

    Returns:
        True if R is transitive.
    """
    raise NotImplementedError


def is_antisymmetric(R):
    """
    Check if relation R is antisymmetric.

    A relation is antisymmetric if (a, b) in R and (b, a) in R implies a == b.

    Args:
        R: set of (a, b) tuples

    Returns:
        True if R is antisymmetric.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (b): Equivalence relation check
# ---------------------------------------------------------------------------

def is_equivalence_relation(S, R):
    """
    Determine if R is an equivalence relation on S.

    An equivalence relation is reflexive, symmetric, and transitive.

    Args:
        S: set of elements
        R: set of (a, b) tuples

    Returns:
        True if R is an equivalence relation on S.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (c): Equivalence classes
# ---------------------------------------------------------------------------

def equivalence_classes(S, R):
    """
    Compute the equivalence classes of an equivalence relation R on S.

    Args:
        S: set of elements
        R: set of (a, b) tuples (must be an equivalence relation)

    Returns:
        list of sets, where each set is an equivalence class.
        The list should be sorted by the minimum element of each class.

    Example:
        S = {0, 1, 2, 3}
        R = {(0,0),(1,1),(2,2),(3,3),(0,2),(2,0),(1,3),(3,1)}
        => [{0, 2}, {1, 3}]
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (d): Partial order
# ---------------------------------------------------------------------------

def is_partial_order(S, R):
    """
    Check if R is a partial order on S.

    A partial order is reflexive, antisymmetric, and transitive.

    Args:
        S: set of elements
        R: set of (a, b) tuples

    Returns:
        True if R is a partial order on S.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (e): Transitive closure
# ---------------------------------------------------------------------------

def transitive_closure(R):
    """
    Compute the transitive closure of relation R using Warshall's algorithm
    (or repeated composition).

    The transitive closure is the smallest transitive relation containing R.

    Args:
        R: set of (a, b) tuples

    Returns:
        set of (a, b) tuples representing the transitive closure.

    Example:
        R = {(1, 2), (2, 3)}
        transitive_closure(R) => {(1, 2), (2, 3), (1, 3)}
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Part (a): Properties of Relations ===")
    try:
        S = {1, 2, 3}
        R_eq = {(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)}
        print(f"  R = {R_eq}")
        print(f"  Reflexive:     {is_reflexive(S, R_eq)}")
        print(f"  Symmetric:     {is_symmetric(R_eq)}")
        print(f"  Transitive:    {is_transitive(R_eq)}")
        print(f"  Antisymmetric: {is_antisymmetric(R_eq)}")
    except NotImplementedError:
        print("TODO: implement relation property checks")

    print()
    print("=== Part (b): Equivalence Relation ===")
    try:
        S = {0, 1, 2, 3}
        R = {(a, b) for a in S for b in S if a % 2 == b % 2}
        print(f"  'Same parity' on {{0,1,2,3}}: equivalence = "
              f"{is_equivalence_relation(S, R)}")
    except NotImplementedError:
        print("TODO: implement is_equivalence_relation")

    print()
    print("=== Part (c): Equivalence Classes ===")
    try:
        S = {0, 1, 2, 3, 4, 5}
        R = {(a, b) for a in S for b in S if a % 3 == b % 3}
        classes = equivalence_classes(S, R)
        print(f"  Equivalence classes (mod 3): {classes}")
    except NotImplementedError:
        print("TODO: implement equivalence_classes")

    print()
    print("=== Part (d): Partial Order ===")
    try:
        S = {1, 2, 3, 6}
        # Divisibility relation
        R_div = {(a, b) for a in S for b in S if b % a == 0}
        print(f"  Divisibility on {{1,2,3,6}}: partial order = "
              f"{is_partial_order(S, R_div)}")
    except NotImplementedError:
        print("TODO: implement is_partial_order")

    print()
    print("=== Part (e): Transitive Closure ===")
    try:
        R = {(1, 2), (2, 3), (3, 4)}
        tc = transitive_closure(R)
        print(f"  R = {{(1,2),(2,3),(3,4)}}")
        print(f"  Transitive closure = {tc}")
    except NotImplementedError:
        print("TODO: implement transitive_closure")
