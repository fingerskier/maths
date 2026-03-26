"""
Problem 1: Steepest Descent (Basic Gradient Descent)

Implement gradient descent to minimize:

    f(x, y) = 5*x^2 + y^2 + 4*x*y - 14*x - 6*y + 20

This is a convex quadratic, so gradient descent is guaranteed to converge
to the unique global minimum.

Tasks:
  (a) Compute the gradient analytically
  (b) Implement gradient descent with a fixed step size
  (c) Experiment with different step sizes: too small (slow convergence),
      too large (divergence), and just right
  (d) Plot the convergence path on a contour plot and plot f(x_k) vs iteration
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Evaluate f at x = [x0, x1]."""
    return 5 * x[0] ** 2 + x[1] ** 2 + 4 * x[0] * x[1] - 14 * x[0] - 6 * x[1] + 20


def grad_f(x):
    """
    Return the gradient [df/dx, df/dy].
      df/dx = 10*x + 4*y - 14
      df/dy = 2*y + 4*x - 6
    """
    # TODO
    raise NotImplementedError


def gradient_descent(x0, alpha, tol=1e-8, max_iter=10000):
    """
    Gradient descent with fixed step size.

    Parameters:
        x0: starting point (numpy array)
        alpha: step size (learning rate)
        tol: stop when ||grad f|| < tol
        max_iter: maximum iterations

    Returns:
        (x_opt, path, f_values) where
        - path is a list of all iterates
        - f_values is a list of f(x_k) at each iteration
    """
    # TODO
    raise NotImplementedError


def experiment_step_sizes(x0=np.array([0.0, 0.0])):
    """
    Run gradient descent with several step sizes and compare.
    Return a dict mapping alpha -> (x_opt, num_iters, converged).
    """
    alphas = [0.001, 0.01, 0.05, 0.1, 0.15, 0.3]
    results = {}
    for alpha in alphas:
        try:
            x_opt, path, f_vals = gradient_descent(x0.copy(), alpha)
            converged = len(path) < 10000
            results[alpha] = (x_opt, len(path), converged)
        except (NotImplementedError, OverflowError):
            results[alpha] = (None, 0, False)
    return results


def plot_convergence(x0=np.array([0.0, 0.0]), alpha=0.05):
    """Plot contour + path and f vs iteration."""
    try:
        x_opt, path, f_vals = gradient_descent(x0, alpha)
    except NotImplementedError:
        print("TODO: implement gradient_descent()")
        return

    path = np.array(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Contour plot
    xr = np.linspace(-1, 4, 200)
    yr = np.linspace(-2, 5, 200)
    X, Y = np.meshgrid(xr, yr)
    Z = 5 * X ** 2 + Y ** 2 + 4 * X * Y - 14 * X - 6 * Y + 20
    ax1.contour(X, Y, Z, levels=30, cmap="viridis")
    ax1.plot(path[:, 0], path[:, 1], "r.-", markersize=3, label="GD path")
    ax1.plot(path[-1, 0], path[-1, 1], "r*", markersize=12, label="Final")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Gradient Descent Path (α={alpha})")
    ax1.legend()
    ax1.grid(True)

    # Convergence plot
    ax2.semilogy(f_vals, "b-")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)")
    ax2.set_title("Objective Value vs Iteration")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        results = experiment_step_sizes()
        print("Step size experiments:")
        for alpha, (x_opt, niters, converged) in sorted(results.items()):
            status = "converged" if converged else "did not converge"
            print(f"  alpha={alpha:.3f}: {niters} iters, {status}")
    except NotImplementedError:
        print("TODO: implement gradient_descent()")

    plot_convergence()
