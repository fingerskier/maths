"""
Problem 2: Propositional Logic

Tasks:
  (a) Implement truth table generation for logical connectives:
      AND, OR, NOT, IMPLIES, IFF
  (b) Evaluate compound propositions given variable assignments
  (c) Verify logical equivalences:
      - Contrapositive: (A => B) <=> (NOT B => NOT A)
      - De Morgan's: NOT(A AND B) <=> (NOT A) OR (NOT B)
      - Material implication: (A => B) <=> (NOT A OR B)
      - Biconditional: (A <=> B) <=> ((A => B) AND (B => A))
  (d) Check validity of arguments using truth tables
      (an argument is valid iff whenever all premises are True,
       the conclusion is also True)
"""

from itertools import product


# ---------------------------------------------------------------------------
# Part (a): Logical connectives
# ---------------------------------------------------------------------------

def logical_and(a, b):
    """Return a AND b."""
    raise NotImplementedError


def logical_or(a, b):
    """Return a OR b."""
    raise NotImplementedError


def logical_not(a):
    """Return NOT a."""
    raise NotImplementedError


def logical_implies(a, b):
    """Return a => b (material implication)."""
    raise NotImplementedError


def logical_iff(a, b):
    """Return a <=> b (biconditional)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (a continued): Truth table generation
# ---------------------------------------------------------------------------

def generate_truth_table(variables, formula):
    """
    Generate a truth table for a propositional formula.

    Args:
        variables: list of variable names, e.g. ["P", "Q"]
        formula: a callable taking len(variables) booleans and returning bool

    Returns:
        list of tuples (assignment_dict, result) where assignment_dict maps
        variable names to bool values.

    Example:
        generate_truth_table(["P", "Q"], lambda p, q: p and q)
        => [({'P': True, 'Q': True}, True),
            ({'P': True, 'Q': False}, False), ...]
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (b): Evaluate compound propositions
# ---------------------------------------------------------------------------

def evaluate_proposition(expression_tree, assignment):
    """
    Evaluate a compound proposition represented as a nested tuple tree.

    The expression tree uses the format:
        - A string like "P" represents a variable (look up in assignment dict)
        - ("NOT", subexpr) represents negation
        - ("AND", left, right) represents conjunction
        - ("OR", left, right) represents disjunction
        - ("IMPLIES", left, right) represents implication
        - ("IFF", left, right) represents biconditional

    Args:
        expression_tree: nested tuple or string representing the proposition
        assignment: dict mapping variable names to bool values

    Returns:
        bool result of evaluating the proposition

    Example:
        evaluate_proposition(("IMPLIES", "P", "Q"), {"P": True, "Q": False})
        => False
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (c): Verify logical equivalences
# ---------------------------------------------------------------------------

def are_logically_equivalent(variables, formula1, formula2):
    """
    Check if two formulas are logically equivalent by comparing truth tables.

    Args:
        variables: list of variable names
        formula1: callable taking booleans, returning bool
        formula2: callable taking booleans, returning bool

    Returns:
        True if the formulas produce identical truth values for all assignments.
    """
    raise NotImplementedError


def verify_contrapositive():
    """
    Verify (P => Q) <=> (NOT Q => NOT P) for all truth value combinations.

    Returns:
        True if the equivalence holds for all assignments.
    """
    raise NotImplementedError


def verify_de_morgan():
    """
    Verify both De Morgan's Laws:
      NOT(P AND Q) <=> (NOT P) OR (NOT Q)
      NOT(P OR Q)  <=> (NOT P) AND (NOT Q)

    Returns:
        True if both equivalences hold for all assignments.
    """
    raise NotImplementedError


def verify_material_implication():
    """
    Verify (P => Q) <=> (NOT P OR Q) for all assignments.

    Returns:
        True if the equivalence holds.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (d): Argument validity
# ---------------------------------------------------------------------------

def is_valid_argument(variables, premises, conclusion):
    """
    Check whether an argument is valid using truth tables.

    An argument is valid iff for every assignment where ALL premises are True,
    the conclusion is also True.

    Args:
        variables: list of variable names
        premises: list of callables (each takes booleans, returns bool)
        conclusion: callable (takes booleans, returns bool)

    Returns:
        True if the argument is valid, False otherwise.

    Example (Modus Ponens):
        is_valid_argument(
            ["P", "Q"],
            premises=[lambda p, q: not p or q,   # P => Q
                      lambda p, q: p],            # P
            conclusion=lambda p, q: q             # therefore Q
        )
        => True
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Part (a): Truth Table Generation ===")
    try:
        table = generate_truth_table(
            ["P", "Q"], lambda p, q: logical_implies(p, q)
        )
        print("Truth table for P => Q:")
        for assignment, result in table:
            print(f"  {assignment} => {result}")
    except NotImplementedError:
        print("TODO: implement generate_truth_table and connectives")

    print()
    print("=== Part (b): Evaluate Compound Propositions ===")
    try:
        expr = ("IMPLIES", ("AND", "P", "Q"), "R")
        val = evaluate_proposition(expr, {"P": True, "Q": True, "R": False})
        print(f"(P AND Q) => R with P=T, Q=T, R=F: {val}")
    except NotImplementedError:
        print("TODO: implement evaluate_proposition")

    print()
    print("=== Part (c): Logical Equivalences ===")
    try:
        print(f"Contrapositive valid: {verify_contrapositive()}")
        print(f"De Morgan's valid:    {verify_de_morgan()}")
        print(f"Material implication: {verify_material_implication()}")
    except NotImplementedError:
        print("TODO: implement equivalence verifications")

    print()
    print("=== Part (d): Argument Validity ===")
    try:
        # Modus Ponens: P => Q, P |- Q
        valid = is_valid_argument(
            ["P", "Q"],
            premises=[lambda p, q: (not p) or q, lambda p, q: p],
            conclusion=lambda p, q: q,
        )
        print(f"Modus Ponens is valid: {valid}")

        # Invalid argument: P => Q, Q |- P  (affirming the consequent)
        invalid = is_valid_argument(
            ["P", "Q"],
            premises=[lambda p, q: (not p) or q, lambda p, q: q],
            conclusion=lambda p, q: p,
        )
        print(f"Affirming the consequent is valid: {invalid}")
    except NotImplementedError:
        print("TODO: implement is_valid_argument")
