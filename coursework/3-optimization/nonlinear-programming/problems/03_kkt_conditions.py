"""
Problem 3: Karush-Kuhn-Tucker (KKT) Conditions

Solve the constrained optimization problem:

    Minimize   f(x, y) = (x - 3)^2 + (y - 2)^2
    subject to:
        g1: x + y  <= 4
        g2: x^2 + y <= 6
        x >= 0, y >= 0

The KKT conditions are:
    1. Stationarity:   grad f = sum(lambda_i * grad g_i)
    2. Primal feasibility: g_i(x) <= 0
    3. Dual feasibility:   lambda_i >= 0
    4. Complementary slackness: lambda_i * g_i(x) = 0

Tasks:
  (a) Write out the KKT conditions explicitly for this problem
  (b) Find all candidate KKT points by considering active constraint cases
  (c) Determine the optimal solution from the KKT candidates
  (d) Verify numerically using scipy.optimize.minimize with constraints
"""

import numpy as np


def objective(x):
    """f(x, y) = (x - 3)^2 + (y - 2)^2"""
    return (x[0] - 3) ** 2 + (x[1] - 2) ** 2


def grad_objective(x):
    """Gradient of f."""
    return np.array([2 * (x[0] - 3), 2 * (x[1] - 2)])


def find_kkt_candidates():
    """
    Enumerate the possible active constraint sets and solve the
    corresponding KKT system for each.

    Return a list of dicts, each with:
    {
        'x': (x, y),
        'lambdas': dict of multiplier values,
        'active_constraints': list of active constraint names,
        'f_value': objective value,
        'kkt_satisfied': bool
    }
    """
    # TODO: consider cases: no active, g1 active, g2 active, both active, etc.
    raise NotImplementedError


def select_optimal(candidates):
    """
    From the KKT candidates, select the one that satisfies all KKT
    conditions and has the smallest objective value.
    """
    valid = [c for c in candidates if c["kkt_satisfied"]]
    return min(valid, key=lambda c: c["f_value"])


def verify_numerically():
    """Verify using scipy.optimize.minimize with SLSQP."""
    from scipy.optimize import minimize

    constraints = [
        {"type": "ineq", "fun": lambda x: 4 - x[0] - x[1]},
        {"type": "ineq", "fun": lambda x: 6 - x[0] ** 2 - x[1]},
    ]
    bounds = [(0, None), (0, None)]
    result = minimize(objective, x0=[1, 1], method="SLSQP",
                      constraints=constraints, bounds=bounds)
    return result.x, result.fun


if __name__ == "__main__":
    print("Numerical verification:", verify_numerically())
    try:
        candidates = find_kkt_candidates()
        print(f"\nFound {len(candidates)} KKT candidates:")
        for c in candidates:
            print(f"  x={c['x']}, f={c['f_value']:.4f}, "
                  f"active={c['active_constraints']}, valid={c['kkt_satisfied']}")
        opt = select_optimal(candidates)
        print(f"\nOptimal: x={opt['x']}, f={opt['f_value']:.4f}")
    except NotImplementedError:
        print("TODO: implement find_kkt_candidates()")
