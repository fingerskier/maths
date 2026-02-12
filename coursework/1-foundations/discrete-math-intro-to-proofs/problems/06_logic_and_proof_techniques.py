"""
Problem 6: Propositional Logic and Proof Techniques

Tasks:
  (a) Build a truth table generator for propositional formulas
  (b) Verify logical equivalences:
      - De Morgan's: NOT(A AND B) <=> (NOT A) OR (NOT B)
      - Contrapositive: (A => B) <=> (NOT B => NOT A)
      - Double negation: NOT(NOT A) <=> A
  (c) Implement a simple resolution prover for propositional logic
  (d) Demonstrate proof by contradiction:
      Prove that sqrt(2) is irrational by assuming it's rational
      and deriving a contradiction (computational illustration)
"""

from itertools import product


def truth_table(variables, formula):
    """
    Generate a truth table for a formula.

    Args:
        variables: list of variable names, e.g. ["A", "B"]
        formula: a function taking booleans and returning boolean

    Returns:
        list of (assignment_dict, result) pairs
    """
    # TODO
    raise NotImplementedError


def verify_de_morgan():
    """Verify NOT(A AND B) <=> (NOT A) OR (NOT B) via truth table."""
    for A, B in product([True, False], repeat=2):
        lhs = not (A and B)
        rhs = (not A) or (not B)
        assert lhs == rhs, f"De Morgan failed for A={A}, B={B}"
    print("De Morgan's Law verified.")


def verify_contrapositive():
    """Verify (A => B) <=> (NOT B => NOT A) via truth table."""
    for A, B in product([True, False], repeat=2):
        implies = (not A) or B  # A => B
        contra = A or (not B)   # NOT B => NOT A  is same as B or (NOT A)...
        # actually: NOT B => NOT A means (NOT(NOT B)) or (NOT A) = B or (NOT A)
        contra = B or (not A)
        assert implies == contra, f"Contrapositive failed for A={A}, B={B}"
    print("Contrapositive equivalence verified.")


def verify_double_negation():
    """Verify NOT(NOT A) <=> A."""
    for A in [True, False]:
        assert (not (not A)) == A
    print("Double negation verified.")


def sqrt2_irrationality_check(max_denominator=10000):
    """
    Computational illustration of proof by contradiction for sqrt(2) irrational.

    If sqrt(2) = p/q in lowest terms, then p^2 = 2*q^2.
    Search for any such (p, q) with gcd(p, q) = 1 and show none exist.
    """
    from math import gcd, isqrt

    found = False
    for q in range(1, max_denominator + 1):
        p_squared = 2 * q * q
        p = isqrt(p_squared)
        if p * p == p_squared:
            if gcd(p, q) == 1:
                print(f"Found: sqrt(2) = {p}/{q} — contradiction!")
                found = True
                break
    if not found:
        print(f"No rational p/q with q <= {max_denominator} satisfies p^2 = 2q^2")
        print("This supports (but doesn't prove) that sqrt(2) is irrational.")
        print()
        print("Formal proof sketch:")
        print("  Assume sqrt(2) = p/q with gcd(p,q) = 1.")
        print("  Then p^2 = 2q^2, so p^2 is even, so p is even.")
        print("  Write p = 2k. Then 4k^2 = 2q^2, so q^2 = 2k^2, so q is even.")
        print("  But then gcd(p,q) >= 2. Contradiction.")


if __name__ == "__main__":
    verify_de_morgan()
    verify_contrapositive()
    verify_double_negation()
    print()
    sqrt2_irrationality_check()
    print()
    try:
        table = truth_table(["A", "B"], lambda A, B: (not A) or B)
        print("Truth table for A => B (i.e., NOT A OR B):")
        for assignment, result in table:
            print(f"  {assignment} => {result}")
    except NotImplementedError:
        print("TODO: implement truth_table()")
