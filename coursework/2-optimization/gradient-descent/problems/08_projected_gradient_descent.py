"""
Problem 8: Projected Gradient Descent

Solve constrained optimization via projected gradient descent:

    Minimize   f(x, y) = (x - 2)^2 + (y - 3)^2
    subject to:
        x^2 + y^2 <= 1   (unit disk constraint)

At each step:
    1. Take a gradient step: z = x_k - alpha * grad_f(x_k)
    2. Project back onto the feasible set: x_{k+1} = proj_C(z)

Tasks:
  (a) Implement the projection onto the unit disk:
      proj_C(z) = z / max(1, ||z||)
  (b) Implement projected gradient descent
  (c) Apply to a box-constrained problem as well:
      min f(x) s.t. 0 <= x_i <= 1 for all i
  (d) Compare with scipy.optimize.minimize using SLSQP
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """f(x, y) = (x - 2)^2 + (y - 3)^2."""
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2


def grad_f(x):
    """Gradient of f."""
    return np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])


def project_disk(z, radius=1.0):
    """
    Project z onto the disk ||x|| <= radius.
    If ||z|| <= radius, return z unchanged.
    Otherwise, return radius * z / ||z||.
    """
    # TODO
    raise NotImplementedError


def project_box(z, lower=0.0, upper=1.0):
    """
    Project z onto the box [lower, upper]^n.
    Simply clip each coordinate.
    """
    # TODO
    raise NotImplementedError


def projected_gradient_descent(f_func, grad_func, proj_func,
                                x0, alpha=0.1, tol=1e-8, max_iter=5000):
    """
    Projected gradient descent.

    At each step:
        z = x_k - alpha * grad_f(x_k)
        x_{k+1} = proj(z)

    Stop when ||x_{k+1} - x_k|| < tol.

    Return (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def verify_with_scipy():
    """Verify disk-constrained problem with scipy."""
    from scipy.optimize import minimize
    constraints = [{"type": "ineq", "fun": lambda x: 1 - x[0] ** 2 - x[1] ** 2}]
    result = minimize(f, x0=[0.0, 0.0], method="SLSQP", constraints=constraints)
    return result.x, result.fun


def plot_projected_gd():
    """Visualize projected gradient descent on the disk."""
    try:
        x_opt, path, _ = projected_gradient_descent(
            f, grad_f, project_disk, x0=np.array([0.0, 0.0])
        )
    except NotImplementedError:
        print("TODO: implement projected_gradient_descent()")
        return

    path = np.array(path)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Contours
    xr = np.linspace(-1.5, 2.5, 200)
    yr = np.linspace(-1.5, 3.5, 200)
    X, Y = np.meshgrid(xr, yr)
    Z = (X - 2) ** 2 + (Y - 3) ** 2
    ax.contour(X, Y, Z, levels=20, alpha=0.4, cmap="viridis")

    # Feasible region
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2, label="Feasible boundary")
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.1, color="blue")

    # Path
    ax.plot(path[:, 0], path[:, 1], "r.-", markersize=4, label="PGD path")
    ax.plot(path[-1, 0], path[-1, 1], "r*", markersize=15, label="Optimal")
    ax.plot(2, 3, "g^", markersize=10, label="Unconstrained min")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Projected Gradient Descent on Unit Disk")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    print("SciPy verification:", verify_with_scipy())
    try:
        x_opt, path, f_vals = projected_gradient_descent(
            f, grad_f, project_disk, x0=np.array([0.0, 0.0])
        )
        print(f"PGD result: x = {x_opt}, f = {f_vals[-1]:.6f}")
        print(f"Converged in {len(path) - 1} iterations")
        print(f"Constraint: ||x|| = {np.linalg.norm(x_opt):.6f}")
    except NotImplementedError:
        print("TODO: implement projected_gradient_descent()")

    plot_projected_gd()
