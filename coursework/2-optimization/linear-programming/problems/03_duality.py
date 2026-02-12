"""
Problem 3: LP Duality

Consider the primal LP:

    Maximize   z = 4x1 + 3x2
    subject to:
        2x1 +  x2 <= 10
         x1 + 3x2 <= 12
        x1, x2 >= 0

Tasks:
  (a) Formulate the dual problem
  (b) Solve both the primal and dual using scipy.optimize.linprog
  (c) Verify strong duality: optimal primal value == optimal dual value
  (d) Verify complementary slackness conditions
"""

import numpy as np


def solve_primal():
    """
    Solve the primal LP. Return (optimal_value, x1, x2).
    """
    # TODO: use scipy.optimize.linprog (remember it minimizes)
    raise NotImplementedError


def formulate_dual():
    """
    Return the dual problem parameters as a dict:
    {
        'c': dual objective coefficients (to minimize),
        'A_lb': constraint matrix (>= constraints),
        'b_lb': RHS of >= constraints,
        'bounds': variable bounds
    }

    The dual of the given primal is:
        Minimize   w = 10y1 + 12y2
        subject to:
            2y1 +  y2 >= 4
             y1 + 3y2 >= 3
            y1, y2 >= 0
    """
    # TODO
    raise NotImplementedError


def solve_dual():
    """
    Solve the dual LP. Return (optimal_value, y1, y2).
    """
    # TODO
    raise NotImplementedError


def verify_strong_duality(primal_val, dual_val, tol=1e-8):
    """Check that primal and dual optimal values are equal."""
    assert abs(primal_val - dual_val) < tol, (
        f"Strong duality violated: primal={primal_val}, dual={dual_val}"
    )
    print(f"Strong duality holds: primal = dual = {primal_val:.4f}")
    return True


def verify_complementary_slackness(x, y, tol=1e-8):
    """
    Verify complementary slackness:
      - y_i * (b_i - A_i @ x) = 0 for each dual variable
      - x_j * (A^T_j @ y - c_j) = 0 for each primal variable

    x = (x1, x2), y = (y1, y2).
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    try:
        p_val, x1, x2 = solve_primal()
        d_val, y1, y2 = solve_dual()
        print(f"Primal: z* = {p_val:.4f}, x = ({x1:.4f}, {x2:.4f})")
        print(f"Dual:   w* = {d_val:.4f}, y = ({y1:.4f}, {y2:.4f})")
        verify_strong_duality(p_val, d_val)
        verify_complementary_slackness((x1, x2), (y1, y2))
    except NotImplementedError:
        print("TODO: implement duality functions")
