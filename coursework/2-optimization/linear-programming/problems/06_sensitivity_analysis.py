"""
Problem 6: Sensitivity Analysis

Given the LP:

    Maximize   z = 5x1 + 4x2
    subject to:
        6x1 + 4x2 <= 24
         x1 + 2x2 <= 6
        x1, x2 >= 0

Tasks:
  (a) Solve the LP and record the optimal basis
  (b) Determine the range of c1 (objective coefficient of x1) for which
      the current basis remains optimal
  (c) Determine the range of b1 (RHS of first constraint) for which
      the current basis remains optimal (and find the shadow price)
  (d) Interpret the shadow prices (dual variables) economically
"""

import numpy as np
from scipy.optimize import linprog


def solve_lp():
    """
    Solve the LP. Return a dict with keys:
      'optimal_value', 'x', 'slack', 'shadow_prices'
    """
    # TODO
    raise NotImplementedError


def objective_coefficient_range(c_index=0):
    """
    Determine the range [c_low, c_high] for the objective coefficient c1
    such that the current optimal basis remains optimal.

    Return (c_low, c_high).

    Hint: vary c1 and re-solve; or use the reduced cost conditions
    from the optimal tableau.
    """
    # TODO
    raise NotImplementedError


def rhs_range(constraint_index=0, base_rhs=24):
    """
    Determine the range [b_low, b_high] for the RHS of the given constraint
    such that the current basis remains feasible.

    Return (b_low, b_high).

    Hint: the basis stays feasible as long as B^{-1} b >= 0.
    """
    # TODO
    raise NotImplementedError


def shadow_price_analysis():
    """
    Compute and interpret shadow prices for each constraint.
    Return a list of (constraint_name, shadow_price, interpretation) tuples.
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    try:
        result = solve_lp()
        print(f"Optimal value: {result['optimal_value']:.4f}")
        print(f"Solution: x = {result['x']}")
        print(f"Slack: {result['slack']}")
        print(f"Shadow prices: {result['shadow_prices']}")
    except NotImplementedError:
        print("TODO: implement solve_lp()")

    try:
        c_lo, c_hi = objective_coefficient_range()
        print(f"\nc1 range for basis optimality: [{c_lo:.4f}, {c_hi:.4f}]")
    except NotImplementedError:
        print("TODO: implement objective_coefficient_range()")

    try:
        b_lo, b_hi = rhs_range()
        print(f"b1 range for basis feasibility: [{b_lo:.4f}, {b_hi:.4f}]")
    except NotImplementedError:
        print("TODO: implement rhs_range()")
