"""
Problem: Boolean Algebra

Implement Boolean expression evaluation, simplification, and normal form
conversion.

Tasks:
  (a) Evaluate a Boolean expression given variable assignments
  (b) Simplify Boolean expressions using algebraic laws (absorption,
      distributive, idempotent, complement)
  (c) Convert a Boolean expression to Conjunctive Normal Form (CNF)
      and Disjunctive Normal Form (DNF) via truth table enumeration
  (d) Verify equivalence of two Boolean expressions by comparing truth tables
"""


def evaluate(expr, assignment):
    """
    Evaluate a Boolean expression represented as a nested tuple.

    Expression format:
        - A string like 'A', 'B' is a variable
        - ('NOT', e) is negation
        - ('AND', e1, e2) is conjunction
        - ('OR', e1, e2) is disjunction

    Parameters:
        expr: a Boolean expression (str or tuple)
        assignment: dict[str, bool], mapping variable names to values

    Returns:
        bool: the truth value of the expression
    """
    # TODO
    raise NotImplementedError


def truth_table(expr, variables):
    """
    Compute the complete truth table for a Boolean expression.

    Parameters:
        expr: Boolean expression (str or tuple)
        variables: list[str], ordered list of variable names

    Returns:
        list[tuple[dict[str, bool], bool]]: list of (assignment, result) pairs
    """
    # TODO
    raise NotImplementedError


def are_equivalent(expr1, expr2, variables):
    """
    Check if two Boolean expressions are logically equivalent by comparing
    their truth tables.

    Parameters:
        expr1, expr2: Boolean expressions
        variables: list[str], variable names used in both expressions

    Returns:
        bool: True if the expressions have identical truth tables
    """
    # TODO
    raise NotImplementedError


def to_dnf(expr, variables):
    """
    Convert a Boolean expression to Disjunctive Normal Form (DNF).

    Build the DNF by collecting all rows of the truth table where the
    expression evaluates to True, forming a minterm for each.

    Parameters:
        expr: Boolean expression
        variables: list[str]

    Returns:
        A Boolean expression in DNF (as a nested tuple), or the string
        representation of the DNF formula.
    """
    # TODO
    raise NotImplementedError


def to_cnf(expr, variables):
    """
    Convert a Boolean expression to Conjunctive Normal Form (CNF).

    Build the CNF by collecting all rows of the truth table where the
    expression evaluates to False, forming a maxterm for each.

    Parameters:
        expr: Boolean expression
        variables: list[str]

    Returns:
        A Boolean expression in CNF (as a nested tuple), or the string
        representation of the CNF formula.
    """
    # TODO
    raise NotImplementedError


def simplify(expr):
    """
    Apply basic Boolean simplification rules:
        - Idempotent: A AND A = A, A OR A = A
        - Complement: A AND NOT A = False, A OR NOT A = True
        - Identity: A AND True = A, A OR False = A
        - Domination: A AND False = False, A OR True = True
        - Absorption: A OR (A AND B) = A, A AND (A OR B) = A
        - Double negation: NOT (NOT A) = A

    Parameters:
        expr: Boolean expression (nested tuple)

    Returns:
        Simplified Boolean expression
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Example: A AND (B OR NOT A)
    expr1 = ('AND', 'A', ('OR', 'B', ('NOT', 'A')))
    # Simplified: A AND B
    expr2 = ('AND', 'A', 'B')
    variables = ['A', 'B']

    try:
        val = evaluate(expr1, {'A': True, 'B': False})
        print(f"A AND (B OR NOT A) with A=T, B=F: {val}")
    except NotImplementedError:
        print("TODO: implement evaluate")

    try:
        table = truth_table(expr1, variables)
        print("\nTruth table for A AND (B OR NOT A):")
        for assignment, result in table:
            vals = ', '.join(f"{k}={v}" for k, v in assignment.items())
            print(f"  {vals} -> {result}")
    except NotImplementedError:
        print("TODO: implement truth_table")

    try:
        equiv = are_equivalent(expr1, expr2, variables)
        print(f"\nA AND (B OR NOT A) == A AND B? {equiv}")
    except NotImplementedError:
        print("TODO: implement are_equivalent")

    try:
        dnf = to_dnf(expr1, variables)
        print(f"\nDNF: {dnf}")
    except NotImplementedError:
        print("TODO: implement to_dnf")

    try:
        cnf = to_cnf(expr1, variables)
        print(f"CNF: {cnf}")
    except NotImplementedError:
        print("TODO: implement to_cnf")

    try:
        simplified = simplify(('NOT', ('NOT', 'A')))
        print(f"\nSimplify NOT(NOT A): {simplified}")
    except NotImplementedError:
        print("TODO: implement simplify")
