"""
Problem 2: Newton's Method for Optimization

Use Newton's method to minimize:

    f(x, y) = x^4 + y^4 - 4*x*y + 1

The Newton update is:  x_{k+1} = x_k - H^{-1}(x_k) * grad f(x_k)

Tasks:
  (a) Compute the gradient and Hessian of f
  (b) Implement Newton's method for optimization
  (c) Analyze convergence: track ||grad f|| at each iteration
  (d) Compare with gradient descent and scipy.optimize.minimize
"""

import numpy as np


def f(x):
    """Evaluate f(x, y) = x[0]^4 + x[1]^4 - 4*x[0]*x[1] + 1."""
    return x[0] ** 4 + x[1] ** 4 - 4 * x[0] * x[1] + 1


def grad_f(x):
    """
    Return the gradient [df/dx, df/dy].
      df/dx = 4x^3 - 4y
      df/dy = 4y^3 - 4x
    """
    # TODO
    raise NotImplementedError


def hessian_f(x):
    """
    Return the 2x2 Hessian matrix.
      d2f/dx2 = 12x^2,  d2f/dxdy = -4
      d2f/dydx = -4,     d2f/dy2 = 12y^2
    """
    # TODO
    raise NotImplementedError


def newtons_method(x0, tol=1e-10, max_iter=100):
    """
    Newton's method for unconstrained minimization.

    Parameters:
        x0: initial guess (numpy array)
        tol: convergence tolerance on ||grad f||
        max_iter: maximum iterations

    Returns:
        (x_opt, f_opt, path, grad_norms) where
        - path is a list of iterates
        - grad_norms is a list of ||grad f|| at each step
    """
    # TODO
    raise NotImplementedError


def gradient_descent(x0, alpha=0.01, tol=1e-10, max_iter=5000):
    """
    Basic gradient descent for comparison.
    Return (x_opt, f_opt, path, grad_norms).
    """
    # TODO
    raise NotImplementedError


def compare_methods(x0=np.array([0.5, 0.5])):
    """Compare Newton's method and gradient descent convergence."""
    import matplotlib.pyplot as plt

    try:
        _, _, _, gn_newton = newtons_method(x0.copy())
        plt.semilogy(gn_newton, "b-o", label="Newton", markersize=3)
    except NotImplementedError:
        print("Newton's method not implemented")

    try:
        _, _, _, gn_gd = gradient_descent(x0.copy())
        plt.semilogy(gn_gd, "r-", label="Gradient Descent")
    except NotImplementedError:
        print("Gradient descent not implemented")

    plt.xlabel("Iteration")
    plt.ylabel(r"$\|\nabla f\|$")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    x0 = np.array([0.5, 0.5])
    try:
        x_opt, f_opt, path, gn = newtons_method(x0)
        print(f"Newton minimum: x = {x_opt}, f = {f_opt:.6f}")
        print(f"Converged in {len(path) - 1} iterations")
    except NotImplementedError:
        print("TODO: implement newtons_method()")

    compare_methods(x0)
