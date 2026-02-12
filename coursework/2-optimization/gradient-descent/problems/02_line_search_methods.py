"""
Problem 2: Line Search Methods

Instead of a fixed step size, use line search to choose the optimal
step size at each iteration. Minimize:

    f(x, y) = 100*(y - x^2)^2 + (1 - x)^2   (Rosenbrock)

Tasks:
  (a) Implement exact line search: at each step, minimize
      phi(alpha) = f(x_k - alpha * grad_f(x_k)) over alpha >= 0
  (b) Implement backtracking line search with the Armijo condition:
      f(x_k + alpha * d) <= f(x_k) + c * alpha * grad_f(x_k)^T d
  (c) Implement Wolfe conditions (Armijo + curvature condition)
  (d) Compare convergence speed of fixed step, exact, and Armijo line search
"""

import numpy as np


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_rosenbrock(x):
    """Gradient of the Rosenbrock function."""
    # TODO
    raise NotImplementedError


def exact_line_search(x, d, f_func):
    """
    Find alpha* = argmin_{alpha >= 0} f(x + alpha * d).
    Use scipy.optimize.minimize_scalar.

    Parameters:
        x: current point
        d: search direction
        f_func: objective function

    Return alpha*.
    """
    # TODO
    raise NotImplementedError


def armijo_backtracking(x, d, f_func, grad_func, c=1e-4, rho=0.5, alpha0=1.0):
    """
    Backtracking line search satisfying the Armijo condition:
        f(x + alpha * d) <= f(x) + c * alpha * grad_f(x)^T d

    Start with alpha0 and multiply by rho until condition is met.
    Return alpha.
    """
    # TODO
    raise NotImplementedError


def wolfe_line_search(x, d, f_func, grad_func, c1=1e-4, c2=0.9, alpha0=1.0):
    """
    Line search satisfying the strong Wolfe conditions:
        1. f(x + alpha*d) <= f(x) + c1*alpha*grad(x)^T*d     (Armijo)
        2. |grad(x + alpha*d)^T*d| <= c2*|grad(x)^T*d|       (curvature)

    Return alpha.
    """
    # TODO
    raise NotImplementedError


def gd_with_line_search(x0, f_func, grad_func, line_search="armijo",
                         tol=1e-8, max_iter=10000):
    """
    Gradient descent using the specified line search method.

    Parameters:
        line_search: one of "exact", "armijo", "wolfe", or a float (fixed step)

    Returns:
        (x_opt, path, f_values)
    """
    # TODO
    raise NotImplementedError


def compare_methods(x0=np.array([-1.0, 1.0])):
    """Compare convergence of different line search strategies."""
    import matplotlib.pyplot as plt

    methods = ["exact", "armijo", "wolfe"]
    for method in methods:
        try:
            _, path, f_vals = gd_with_line_search(
                x0.copy(), rosenbrock, grad_rosenbrock, line_search=method
            )
            plt.semilogy(f_vals, label=f"{method} ({len(f_vals)} iters)")
        except NotImplementedError:
            print(f"TODO: implement {method} line search")

    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title("Line Search Comparison on Rosenbrock")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_methods()
