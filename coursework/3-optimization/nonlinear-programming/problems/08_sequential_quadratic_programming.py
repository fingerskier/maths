"""
Problem 8: Sequential Quadratic Programming (SQP)

Solve the nonlinearly constrained problem:

    Minimize   f(x, y) = (x - 1)^2 + (y - 1)^2
    subject to:
        g1: x^2 + y^2 <= 2
        g2: x + y >= 1

SQP solves a sequence of quadratic subproblems, each approximating
the original problem locally.

Tasks:
  (a) At a given iterate x_k, form the QP subproblem:
      min  grad_f^T d + (1/2) d^T H d
      s.t. g_i(x_k) + grad_g_i^T d <= 0  (for active/violated constraints)
  (b) Implement one SQP iteration: solve QP, update x
  (c) Run the full SQP loop until convergence
  (d) Verify with scipy.optimize.minimize using SLSQP
"""

import numpy as np


def f(x):
    return (x[0] - 1) ** 2 + (x[1] - 1) ** 2


def grad_f(x):
    return np.array([2 * (x[0] - 1), 2 * (x[1] - 1)])


def hess_f(x):
    return np.array([[2.0, 0.0], [0.0, 2.0]])


def g1(x):
    """Inequality constraint: x^2 + y^2 - 2 <= 0."""
    return x[0] ** 2 + x[1] ** 2 - 2


def grad_g1(x):
    return np.array([2 * x[0], 2 * x[1]])


def g2(x):
    """Inequality constraint: 1 - x - y <= 0 (i.e., x + y >= 1)."""
    return 1 - x[0] - x[1]


def grad_g2(x):
    return np.array([-1.0, -1.0])


def solve_qp_subproblem(x_k, H_k):
    """
    Form and solve the QP subproblem at x_k:
        min  grad_f(x_k)^T d + (1/2) d^T H_k d
        s.t. g1(x_k) + grad_g1(x_k)^T d <= 0
             g2(x_k) + grad_g2(x_k)^T d <= 0

    Return (d, lambdas) where d is the step direction and
    lambdas are the QP multipliers.
    """
    # TODO: solve using scipy.optimize.minimize or a QP solver
    raise NotImplementedError


def sqp_method(x0=None, tol=1e-8, max_iter=50):
    """
    Run the SQP algorithm.

    Parameters:
        x0: initial feasible point
        tol: convergence tolerance on ||d||
        max_iter: maximum iterations

    Returns:
        (x_opt, f_opt, path, multipliers_history)
    """
    if x0 is None:
        x0 = np.array([0.5, 0.5])
    # TODO
    raise NotImplementedError


def verify_with_scipy():
    """Verify using scipy.optimize.minimize with SLSQP."""
    from scipy.optimize import minimize

    constraints = [
        {"type": "ineq", "fun": lambda x: 2 - x[0] ** 2 - x[1] ** 2},
        {"type": "ineq", "fun": lambda x: x[0] + x[1] - 1},
    ]
    result = minimize(f, x0=[0.5, 0.5], method="SLSQP", constraints=constraints)
    return result.x, result.fun


if __name__ == "__main__":
    print("SciPy verification:", verify_with_scipy())
    try:
        x_opt, f_opt, path, _ = sqp_method()
        print(f"\nSQP result: x = {x_opt}, f = {f_opt:.6f}")
        print(f"Converged in {len(path) - 1} iterations")
        print(f"Constraints: g1 = {g1(x_opt):.6f}, g2 = {g2(x_opt):.6f}")
    except NotImplementedError:
        print("TODO: implement sqp_method()")
