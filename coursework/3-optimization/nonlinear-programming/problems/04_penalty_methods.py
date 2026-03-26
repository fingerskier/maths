"""
Problem 4: Penalty and Barrier Methods

Solve the constrained problem using penalty/barrier approaches:

    Minimize   f(x, y) = x^2 + y^2
    subject to:
        x + y >= 2

Tasks:
  (a) Implement the quadratic penalty method:
      min f(x) + (rho/2) * max(0, 2 - x - y)^2
      Solve for increasing values of rho.
  (b) Implement the log-barrier (interior point) method:
      min f(x) - (1/t) * ln(x + y - 2)
      Solve for increasing values of t.
  (c) Track how solutions converge to the true optimum as rho -> inf / t -> inf
  (d) Plot the convergence paths for both methods
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Objective: x^2 + y^2."""
    return x[0] ** 2 + x[1] ** 2


def penalty_function(x, rho):
    """
    Quadratic penalty: f(x) + (rho/2) * max(0, 2 - x0 - x1)^2

    The constraint x + y >= 2 is violated when x + y < 2.
    """
    # TODO
    raise NotImplementedError


def barrier_function(x, t):
    """
    Log-barrier: f(x) - (1/t) * ln(x0 + x1 - 2)

    Only defined in the interior of the feasible region (x + y > 2).
    """
    # TODO
    raise NotImplementedError


def penalty_method(rho_values=None):
    """
    Solve a sequence of unconstrained problems with increasing rho.
    Return a list of (rho, x_opt, f_opt) for each rho.
    """
    if rho_values is None:
        rho_values = [1, 10, 100, 1000, 10000]
    # TODO: for each rho, minimize penalty_function using scipy.optimize.minimize
    raise NotImplementedError


def barrier_method(t_values=None):
    """
    Solve a sequence of unconstrained problems with increasing t.
    Start from a strictly feasible point (e.g., x0 = [1.5, 1.5]).
    Return a list of (t, x_opt, f_opt) for each t.
    """
    if t_values is None:
        t_values = [1, 10, 100, 1000, 10000]
    # TODO: for each t, minimize barrier_function from a feasible start
    raise NotImplementedError


def plot_convergence():
    """Plot how both methods converge to the true optimum (1, 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    try:
        pen_results = penalty_method()
        rhos, xs, fs = zip(*pen_results)
        xs = np.array(xs)
        axes[0].plot(xs[:, 0], xs[:, 1], "bo-", markersize=6)
        axes[0].plot(1, 1, "r*", markersize=15, label="True optimum")
        for rho, x in zip(rhos, xs):
            axes[0].annotate(f"ρ={rho}", (x[0], x[1]), fontsize=8)
        axes[0].set_title("Penalty Method Path")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid(True)
    except NotImplementedError:
        axes[0].set_title("Penalty Method (not implemented)")

    try:
        bar_results = barrier_method()
        ts, xs, fs = zip(*bar_results)
        xs = np.array(xs)
        axes[1].plot(xs[:, 0], xs[:, 1], "go-", markersize=6)
        axes[1].plot(1, 1, "r*", markersize=15, label="True optimum")
        for t, x in zip(ts, xs):
            axes[1].annotate(f"t={t}", (x[0], x[1]), fontsize=8)
        axes[1].set_title("Barrier Method Path")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend()
        axes[1].grid(True)
    except NotImplementedError:
        axes[1].set_title("Barrier Method (not implemented)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("True optimum: x = (1, 1), f = 2")
    try:
        results = penalty_method()
        print("\nPenalty method:")
        for rho, x, fval in results:
            print(f"  rho={rho:6d}: x=({x[0]:.6f}, {x[1]:.6f}), f={fval:.6f}")
    except NotImplementedError:
        print("TODO: implement penalty_method()")

    try:
        results = barrier_method()
        print("\nBarrier method:")
        for t, x, fval in results:
            print(f"  t={t:6d}: x=({x[0]:.6f}, {x[1]:.6f}), f={fval:.6f}")
    except NotImplementedError:
        print("TODO: implement barrier_method()")

    plot_convergence()
