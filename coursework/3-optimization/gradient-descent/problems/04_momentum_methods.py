"""
Problem 4: Momentum Methods

Gradient descent with momentum accelerates convergence, especially
for ill-conditioned problems.

Minimize:  f(x, y) = 50*x^2 + y^2   (condition number = 50)

Methods to implement:
  - Heavy-ball (Polyak momentum):
      v_{k+1} = beta * v_k - alpha * grad_f(x_k)
      x_{k+1} = x_k + v_{k+1}

  - Nesterov accelerated gradient (NAG):
      y_k     = x_k + beta * v_k
      v_{k+1} = beta * v_k - alpha * grad_f(y_k)
      x_{k+1} = x_k + v_{k+1}

Tasks:
  (a) Implement standard gradient descent (no momentum)
  (b) Implement heavy-ball momentum
  (c) Implement Nesterov accelerated gradient
  (d) Compare convergence rates on the ill-conditioned quadratic
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """f(x, y) = 50*x^2 + y^2."""
    return 50 * x[0] ** 2 + x[1] ** 2


def grad_f(x):
    """Gradient: [100*x, 2*y]."""
    return np.array([100 * x[0], 2 * x[1]])


def vanilla_gd(x0, alpha=0.005, tol=1e-8, max_iter=5000):
    """
    Standard gradient descent (no momentum).
    Return (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def heavy_ball(x0, alpha=0.005, beta=0.9, tol=1e-8, max_iter=5000):
    """
    Heavy-ball (Polyak) momentum.

    v_{k+1} = beta * v_k - alpha * grad_f(x_k)
    x_{k+1} = x_k + v_{k+1}

    Return (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def nesterov_ag(x0, alpha=0.005, beta=0.9, tol=1e-8, max_iter=5000):
    """
    Nesterov Accelerated Gradient.

    y_k     = x_k + beta * v_k
    v_{k+1} = beta * v_k - alpha * grad_f(y_k)
    x_{k+1} = x_k + v_{k+1}

    Return (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def compare_methods(x0=np.array([5.0, 5.0])):
    """Compare all three methods on the ill-conditioned quadratic."""
    methods = [
        ("Vanilla GD", vanilla_gd),
        ("Heavy-ball", heavy_ball),
        ("Nesterov AG", nesterov_ag),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, method in methods:
        try:
            x_opt, path, f_vals = method(x0.copy())
            path = np.array(path)
            ax1.plot(path[:, 0], path[:, 1], ".-", markersize=2, label=name)
            ax2.semilogy(f_vals, label=f"{name} ({len(f_vals)} iters)")
        except NotImplementedError:
            print(f"TODO: implement {name}")

    # Contour background
    xr = np.linspace(-6, 6, 200)
    yr = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(xr, yr)
    Z = 50 * X ** 2 + Y ** 2
    ax1.contour(X, Y, Z, levels=20, alpha=0.3, cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Optimization Paths")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)")
    ax2.set_title("Convergence Comparison")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_methods()
