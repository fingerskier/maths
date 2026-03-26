"""
Problem 7: Augmented Lagrangian Method

Solve the equality-constrained problem:

    Minimize   f(x, y) = (x - 1)^2 + (y - 2.5)^2
    subject to:
        h(x, y) = x + y - 4 = 0

The augmented Lagrangian is:
    L_A(x, lambda, rho) = f(x) + lambda * h(x) + (rho/2) * h(x)^2

The method alternates between:
    1. Minimize L_A(x, lambda_k, rho_k) over x
    2. Update lambda: lambda_{k+1} = lambda_k + rho_k * h(x_{k+1})
    3. Optionally increase rho

Tasks:
  (a) Implement the augmented Lagrangian function
  (b) Implement the alternating minimization-update loop
  (c) Track convergence of x, lambda, and h(x) across iterations
  (d) Compare with the direct solution via Lagrange multipliers
"""

import numpy as np


def f(x):
    """Objective function."""
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2


def h(x):
    """Equality constraint: x + y - 4 = 0."""
    return x[0] + x[1] - 4


def augmented_lagrangian(x, lam, rho):
    """
    Evaluate L_A = f(x) + lam * h(x) + (rho/2) * h(x)^2.
    """
    # TODO
    raise NotImplementedError


def solve_augmented_lagrangian(x0=None, lam0=0.0, rho0=1.0, rho_max=1e6,
                                tol=1e-8, max_outer=50):
    """
    Augmented Lagrangian method.

    Parameters:
        x0: initial point
        lam0: initial multiplier estimate
        rho0: initial penalty parameter
        rho_max: maximum penalty
        tol: tolerance on constraint violation
        max_outer: max outer iterations

    Returns:
        (x_opt, lam_opt, history) where history is a list of
        (x, lam, rho, h_val, f_val) at each outer iteration.
    """
    if x0 is None:
        x0 = np.array([0.0, 0.0])
    # TODO
    raise NotImplementedError


def direct_lagrange_solution():
    """
    Solve analytically using Lagrange multipliers.
    grad f = lambda * grad h
    => (2(x-1), 2(y-2.5)) = lambda * (1, 1)
    Plus h(x,y) = 0.

    Return (x_opt, y_opt, lambda_opt).
    """
    # TODO
    raise NotImplementedError


def plot_convergence(history):
    """Plot convergence of constraint violation and multiplier estimate."""
    import matplotlib.pyplot as plt

    iters = range(len(history))
    h_vals = [abs(h_val) for _, _, _, h_val, _ in history]
    lams = [lam for _, lam, _, _, _ in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogy(iters, h_vals, "b-o")
    ax1.set_xlabel("Outer iteration")
    ax1.set_ylabel("|h(x)|")
    ax1.set_title("Constraint Violation")
    ax1.grid(True)

    ax2.plot(iters, lams, "r-o")
    ax2.set_xlabel("Outer iteration")
    ax2.set_ylabel("lambda")
    ax2.set_title("Multiplier Estimate")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        x_direct = direct_lagrange_solution()
        print(f"Direct solution: x={x_direct[:2]}, lambda={x_direct[2]:.4f}")
    except NotImplementedError:
        print("TODO: implement direct_lagrange_solution()")

    try:
        x_opt, lam_opt, history = solve_augmented_lagrangian()
        print(f"\nAugmented Lagrangian result:")
        print(f"  x = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
        print(f"  lambda = {lam_opt:.6f}")
        print(f"  h(x) = {h(x_opt):.2e}")
        print(f"  f(x) = {f(x_opt):.6f}")
        plot_convergence(history)
    except NotImplementedError:
        print("TODO: implement solve_augmented_lagrangian()")
