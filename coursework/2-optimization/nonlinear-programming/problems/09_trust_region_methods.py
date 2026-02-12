"""
Problem 9: Trust Region Methods

Minimize the Himmelblau function using a trust-region approach:

    f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

This function has four local minima at approximately:
    (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)

Trust Region Method:
    At each step, solve the subproblem:
        min  m_k(d) = f_k + g_k^T d + (1/2) d^T B_k d
        s.t. ||d|| <= Delta_k
    where Delta_k is the trust region radius.

    Accept/reject step and adjust Delta_k based on the ratio:
        rho_k = (f(x_k) - f(x_k + d)) / (m_k(0) - m_k(d))

Tasks:
  (a) Compute gradient and Hessian of the Himmelblau function
  (b) Implement the Cauchy point (steepest descent within trust region)
  (c) Implement the dogleg method for the trust region subproblem
  (d) Run trust region optimization from different starting points to find
      all four minima
"""

import numpy as np


def himmelblau(x):
    """f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2"""
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def grad_himmelblau(x):
    """Return the gradient of the Himmelblau function."""
    # TODO
    raise NotImplementedError


def hess_himmelblau(x):
    """Return the Hessian of the Himmelblau function."""
    # TODO
    raise NotImplementedError


def cauchy_point(g, B, delta):
    """
    Compute the Cauchy point: the steepest descent direction
    restricted to the trust region ||d|| <= delta.

    Parameters:
        g: gradient
        B: Hessian (or approximation)
        delta: trust region radius

    Return d_c (the Cauchy point).
    """
    # TODO
    raise NotImplementedError


def dogleg_step(g, B, delta):
    """
    Compute the dogleg step.

    The dogleg path goes from 0 -> d_U (unconstrained Cauchy) -> d_N (Newton step).
    If ||d_N|| <= delta, return d_N.
    If ||d_U|| >= delta, return scaled Cauchy direction.
    Otherwise, interpolate along the dogleg path.

    Return d (the dogleg step).
    """
    # TODO
    raise NotImplementedError


def trust_region_method(x0, delta0=1.0, delta_max=10.0, eta=0.15,
                        tol=1e-10, max_iter=200):
    """
    Trust region optimization with dogleg step.

    Parameters:
        x0: initial point
        delta0: initial trust region radius
        delta_max: maximum radius
        eta: acceptance threshold for rho
        tol: convergence tolerance on ||grad||
        max_iter: maximum iterations

    Returns:
        (x_opt, f_opt, path) where path tracks all iterates.
    """
    # TODO
    raise NotImplementedError


def find_all_minima():
    """
    Run trust region method from multiple starting points to find
    all four local minima of Himmelblau's function.

    Starting points: (2, 2), (-3, 3), (-3, -3), (3, -2)
    """
    starting_points = [
        np.array([2.0, 2.0]),
        np.array([-3.0, 3.0]),
        np.array([-3.0, -3.0]),
        np.array([3.0, -2.0]),
    ]
    results = []
    for x0 in starting_points:
        try:
            x_opt, f_opt, path = trust_region_method(x0)
            results.append((x0, x_opt, f_opt, len(path)))
        except NotImplementedError:
            results.append((x0, None, None, 0))
    return results


if __name__ == "__main__":
    results = find_all_minima()
    for x0, x_opt, f_opt, iters in results:
        if x_opt is not None:
            print(f"Start: {x0} -> Min: ({x_opt[0]:.4f}, {x_opt[1]:.4f}), "
                  f"f = {f_opt:.2e}, iters = {iters}")
        else:
            print(f"Start: {x0} -> TODO: implement trust_region_method()")
