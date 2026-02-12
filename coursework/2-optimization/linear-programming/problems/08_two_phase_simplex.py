"""
Problem 8: Two-Phase Simplex Method

Solve the LP where a basic feasible solution is not immediately available:

    Minimize   z = 2x1 + 3x2 + x3
    subject to:
        x1 + x2 + x3  = 40
        2x1 + x2 - x3 >= 10
        -x2 + x3      >= 10
        x1, x2, x3 >= 0

Tasks:
  (a) Phase I: introduce artificial variables and minimize their sum to find
      an initial BFS (if feasible)
  (b) Phase II: starting from the Phase I BFS, optimize the original objective
  (c) Implement both phases and track the tableaux
  (d) Verify with scipy.optimize.linprog
"""

import numpy as np


def phase_one(A_eq, b_eq):
    """
    Phase I of the two-phase simplex method.

    Add artificial variables and minimize their sum.
    Return the initial BFS (basis, tableau) if feasible.
    Raise ValueError if the problem is infeasible.

    Parameters:
        A_eq: constraint matrix (m x n)
        b_eq: RHS vector (m,)

    Returns:
        (basis, tableau) where basis is a list of basic variable indices.
    """
    # TODO
    raise NotImplementedError


def phase_two(basis, tableau, c):
    """
    Phase II: optimize the original objective c^T x starting from
    the BFS found in Phase I.

    Return (optimal_value, x_solution).
    """
    # TODO
    raise NotImplementedError


def solve_two_phase():
    """
    Set up and solve the full problem using the two-phase method.

    Convert >= constraints to equalities with surplus variables,
    then apply Phase I and Phase II.

    Return (optimal_value, x1, x2, x3).
    """
    # TODO
    raise NotImplementedError


def verify_with_scipy():
    """Verify using scipy.optimize.linprog."""
    from scipy.optimize import linprog
    c = [2, 3, 1]
    A_eq = [[1, 1, 1]]
    b_eq = [40]
    A_ub = [[-2, -1, 1], [0, 1, -1]]  # negate >= to <=
    b_ub = [-10, -10]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, None)] * 3)
    return result.fun, result.x


if __name__ == "__main__":
    print("SciPy verification:", verify_with_scipy())
    try:
        val, x1, x2, x3 = solve_two_phase()
        print(f"\nTwo-phase result: z* = {val:.4f}")
        print(f"  x1 = {x1:.4f}, x2 = {x2:.4f}, x3 = {x3:.4f}")
    except NotImplementedError:
        print("TODO: implement two-phase simplex")
