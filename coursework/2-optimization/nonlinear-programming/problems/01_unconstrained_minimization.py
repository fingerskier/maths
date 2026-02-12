"""
Problem 1: Unconstrained Minimization

Find the minimum of the Rosenbrock function:

    f(x, y) = (1 - x)^2 + 100*(y - x^2)^2

Tasks:
  (a) Compute the gradient analytically: grad f = (df/dx, df/dy)
  (b) Compute the Hessian matrix analytically
  (c) Find the critical point by setting grad f = 0
  (d) Verify using scipy.optimize.minimize and plot the contour with the path
"""

import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x):
    """Evaluate f(x, y) = (1 - x[0])^2 + 100*(x[1] - x[0]^2)^2."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def gradient(x):
    """
    Return the gradient [df/dx, df/dy] at point x = [x0, x1].

    df/dx = -2(1 - x) - 400*x*(y - x^2)
    df/dy = 200*(y - x^2)
    """
    # TODO
    raise NotImplementedError


def hessian(x):
    """
    Return the 2x2 Hessian matrix at point x = [x0, x1].
    """
    # TODO
    raise NotImplementedError


def find_minimum_analytically():
    """
    Return the minimizer (x*, y*) by solving grad f = 0.
    Hint: the global minimum is well-known.
    """
    # TODO
    raise NotImplementedError


def verify_numerically(x0=np.array([-1.0, 1.0])):
    """Use scipy.optimize.minimize to find the minimum starting from x0."""
    from scipy.optimize import minimize
    result = minimize(rosenbrock, x0, method="BFGS", jac=gradient)
    return result.x, result.fun


def plot_contour_with_path(x0=np.array([-1.0, 1.0])):
    """Plot Rosenbrock contours and the optimization path."""
    from scipy.optimize import minimize

    path = [x0.copy()]
    def callback(xk):
        path.append(xk.copy())

    try:
        minimize(rosenbrock, x0, method="BFGS", jac=gradient, callback=callback)
    except NotImplementedError:
        minimize(rosenbrock, x0, method="Nelder-Mead", callback=callback)

    path = np.array(path)

    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    plt.figure(figsize=(10, 7))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 30), cmap="viridis")
    plt.colorbar(label="f(x, y)")
    plt.plot(path[:, 0], path[:, 1], "r.-", markersize=4, label="Optimization path")
    plt.plot(1, 1, "r*", markersize=15, label="Global minimum (1, 1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rosenbrock Function Contour")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        x_star = find_minimum_analytically()
        print(f"Analytic minimum at: {x_star}")
    except NotImplementedError:
        print("TODO: implement find_minimum_analytically()")

    try:
        x_opt, f_opt = verify_numerically()
        print(f"Numerical minimum at: {x_opt}, f = {f_opt:.6f}")
    except NotImplementedError:
        print("TODO: implement gradient() for numerical verification")

    plot_contour_with_path()
