"""
Problem 7: Integer Linear Programming (Branch and Bound)

Solve the following ILP:

    Maximize   z = 8x1 + 5x2
    subject to:
        x1 + x2  <= 6
        9x1 + 5x2 <= 45
        x1, x2 >= 0 and integer

Tasks:
  (a) Solve the LP relaxation (ignore integrality)
  (b) Implement a basic branch-and-bound algorithm
  (c) Track the branching tree (nodes explored, bounds at each node)
  (d) Verify the solution with scipy.optimize.milp
"""

import numpy as np


def solve_relaxation():
    """
    Solve the LP relaxation (continuous variables).
    Return (optimal_value, x1, x2).
    """
    from scipy.optimize import linprog
    # TODO
    raise NotImplementedError


def branch_and_bound():
    """
    Implement branch-and-bound for the given ILP.

    Return a dict:
    {
        'optimal_value': float,
        'x': (x1, x2),
        'nodes_explored': int,
        'tree': [(node_id, bound, branch_var, branch_val, status), ...]
    }

    Use the LP relaxation at each node. Branch on the variable with the
    largest fractional part.
    """
    # TODO
    raise NotImplementedError


def verify_with_scipy():
    """Verify using scipy.optimize.milp."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    c = np.array([-8, -5])  # negate for minimization
    constraints = LinearConstraint(
        A=np.array([[1, 1], [9, 5]]),
        ub=np.array([6, 45]),
    )
    integrality = np.array([1, 1])
    bounds = Bounds(lb=0)
    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    return -result.fun, result.x


if __name__ == "__main__":
    print("LP relaxation:")
    try:
        val, x1, x2 = solve_relaxation()
        print(f"  z* = {val:.4f}, x = ({x1:.4f}, {x2:.4f})")
    except NotImplementedError:
        print("  TODO: implement solve_relaxation()")

    print("\nBranch and bound:")
    try:
        result = branch_and_bound()
        print(f"  z* = {result['optimal_value']}, x = {result['x']}")
        print(f"  Nodes explored: {result['nodes_explored']}")
    except NotImplementedError:
        print("  TODO: implement branch_and_bound()")

    print("\nSciPy MILP verification:", verify_with_scipy())
