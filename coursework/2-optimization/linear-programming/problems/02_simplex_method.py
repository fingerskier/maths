"""
Problem 2: The Simplex Method

Implement the simplex algorithm to solve:

    Maximize   z = 5x1 + 4x2
    subject to:
        6x1 + 4x2 <= 24
        x1 + 2x2  <= 6
        x1, x2    >= 0

Tasks:
  (a) Convert to standard form by introducing slack variables
  (b) Construct the initial simplex tableau
  (c) Implement pivoting (entering and leaving variable selection)
  (d) Iterate until optimality; verify with scipy.optimize.linprog
"""

import numpy as np


def initial_tableau():
    """
    Return the initial simplex tableau as a 2D numpy array.

    Standard form:
        6x1 + 4x2 + s1          = 24
         x1 + 2x2      + s2     = 6
        -5x1 - 4x2              = 0  (objective row)

    Tableau layout: each row is [x1, x2, s1, s2 | rhs]
    Last row is the objective row.
    """
    # TODO
    raise NotImplementedError


def find_pivot_column(tableau):
    """
    Return the index of the entering variable (most negative coefficient
    in the objective row, excluding the rhs column).
    Return -1 if the current solution is optimal.
    """
    # TODO
    raise NotImplementedError


def find_pivot_row(tableau, col):
    """
    Return the index of the leaving variable using the minimum ratio test.
    Only consider rows with positive entries in the pivot column.
    Raise ValueError if the problem is unbounded.
    """
    # TODO
    raise NotImplementedError


def pivot(tableau, row, col):
    """
    Perform a pivot operation on the tableau at position (row, col).
    Return the updated tableau.
    """
    # TODO
    raise NotImplementedError


def simplex(tableau):
    """
    Run the simplex algorithm on the given tableau.
    Return (optimal_value, solution_dict) where solution_dict maps
    variable names to their values.
    """
    # TODO: iterate find_pivot_column, find_pivot_row, pivot until optimal
    raise NotImplementedError


def verify_with_scipy():
    """Verify using scipy.optimize.linprog (which minimizes, so negate objective)."""
    from scipy.optimize import linprog
    c = [-5, -4]  # negate for minimization
    A_ub = [[6, 4], [1, 2]]
    b_ub = [24, 6]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])
    return -result.fun, result.x


if __name__ == "__main__":
    print("SciPy verification:", verify_with_scipy())
    try:
        tab = initial_tableau()
        print("Initial tableau:\n", tab)
        opt_val, sol = simplex(tab)
        print(f"Optimal value: {opt_val}")
        print(f"Solution: {sol}")
    except NotImplementedError:
        print("TODO: implement simplex functions")
