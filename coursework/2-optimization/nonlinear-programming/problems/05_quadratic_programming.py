"""
Problem 5: Quadratic Programming

Solve the QP:

    Minimize   (1/2) x^T Q x + c^T x
    subject to:  A x <= b,  x >= 0

    where:
        Q = [[2, 0.5],     c = [-8, -10]
             [0.5, 4]]

        A = [[1, 1],       b = [10, 8, 4]
             [0, 1],
             [-1, 0]]

Tasks:
  (a) Verify that Q is positive definite (so the problem is convex)
  (b) Write out the KKT conditions for this QP
  (c) Solve the KKT system to find the optimal solution
  (d) Verify using scipy.optimize.minimize with method='trust-constr'
"""

import numpy as np


Q = np.array([[2.0, 0.5],
              [0.5, 4.0]])
c = np.array([-8.0, -10.0])
A = np.array([[1.0, 1.0],
              [0.0, 1.0],
              [-1.0, 0.0]])
b = np.array([10.0, 8.0, 4.0])


def check_positive_definite(Q):
    """
    Check if Q is positive definite by examining its eigenvalues.
    Return (is_pd, eigenvalues).
    """
    # TODO
    raise NotImplementedError


def qp_objective(x):
    """Evaluate (1/2) x^T Q x + c^T x."""
    return 0.5 * x @ Q @ x + c @ x


def qp_gradient(x):
    """Return Q x + c."""
    return Q @ x + c


def solve_kkt():
    """
    Solve the QP by solving the KKT system.

    The KKT conditions for this QP are:
        Q x + c + A^T lambda = 0  (stationarity, for active constraints)
        A x <= b                  (primal feasibility)
        lambda >= 0               (dual feasibility)
        lambda_i (A_i x - b_i) = 0 (complementary slackness)

    Return (x_opt, lambdas, active_set).
    """
    # TODO: use active set method or enumerate active constraint sets
    raise NotImplementedError


def verify_with_scipy():
    """Verify using scipy.optimize.minimize."""
    from scipy.optimize import minimize, LinearConstraint, Bounds

    constraints = LinearConstraint(A, ub=b)
    bounds = Bounds(lb=0)
    result = minimize(qp_objective, x0=[0.0, 0.0], jac=qp_gradient,
                      constraints=constraints, bounds=bounds,
                      method="trust-constr")
    return result.x, result.fun


if __name__ == "__main__":
    try:
        is_pd, eigs = check_positive_definite(Q)
        print(f"Q is positive definite: {is_pd}")
        print(f"Eigenvalues: {eigs}")
    except NotImplementedError:
        print("TODO: implement check_positive_definite()")

    print("\nSciPy verification:", verify_with_scipy())

    try:
        x_opt, lambdas, active = solve_kkt()
        print(f"\nKKT solution: x = {x_opt}")
        print(f"Multipliers: {lambdas}")
        print(f"Active constraints: {active}")
        print(f"Objective value: {qp_objective(x_opt):.6f}")
    except NotImplementedError:
        print("TODO: implement solve_kkt()")
