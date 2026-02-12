"""
Problem 7: Convergence Rate Analysis

Study the convergence rates of gradient descent on quadratic functions:

    f(x) = (1/2) x^T A x - b^T x

The convergence rate depends on the condition number kappa = lambda_max / lambda_min.

Theoretical bound:
    f(x_k) - f(x*) <= ((kappa - 1)/(kappa + 1))^{2k} * (f(x_0) - f(x*))

Tasks:
  (a) Implement gradient descent for quadratic objectives with exact step size
  (b) Generate problems with different condition numbers (kappa = 2, 10, 100, 1000)
  (c) Measure actual convergence rates and compare with the theoretical bound
  (d) Plot convergence curves and verify the linear rate
"""

import numpy as np
import matplotlib.pyplot as plt


def make_quadratic(d=10, kappa=10, seed=42):
    """
    Create a d-dimensional quadratic f(x) = (1/2) x^T A x - b^T x
    with condition number kappa.

    Returns (A, b, x_star) where x_star = A^{-1} b is the minimizer.
    """
    rng = np.random.RandomState(seed)
    # Generate eigenvalues uniformly in [1, kappa]
    eigenvalues = np.linspace(1, kappa, d)
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(rng.randn(d, d))
    A = Q @ np.diag(eigenvalues) @ Q.T
    b = rng.randn(d)
    x_star = np.linalg.solve(A, b)
    return A, b, x_star


def gd_quadratic(A, b, x0=None, max_iter=500):
    """
    Gradient descent on (1/2) x^T A x - b^T x with exact step size:
        alpha_k = ||g_k||^2 / (g_k^T A g_k)

    Return (x_opt, errors) where errors[k] = f(x_k) - f(x*).
    """
    if x0 is None:
        x0 = np.zeros(len(b))
    # TODO
    raise NotImplementedError


def theoretical_bound(kappa, k, initial_error):
    """
    Return the theoretical upper bound on f(x_k) - f(x*):
        ((kappa - 1)/(kappa + 1))^{2k} * initial_error
    """
    # TODO
    raise NotImplementedError


def convergence_study(d=10, kappas=None):
    """
    Run GD for different condition numbers and compare with theory.
    """
    if kappas is None:
        kappas = [2, 10, 100, 1000]
    # TODO: for each kappa, run GD and collect convergence data
    raise NotImplementedError


def plot_convergence(d=10, kappas=None):
    """Plot actual vs theoretical convergence for different kappa values."""
    if kappas is None:
        kappas = [2, 10, 100, 1000]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, kappa in enumerate(kappas):
        A, b, x_star = make_quadratic(d, kappa)
        f_star = 0.5 * x_star @ A @ x_star - b @ x_star

        try:
            _, errors = gd_quadratic(A, b)
            iters = np.arange(len(errors))

            # Theoretical bound
            bound = [theoretical_bound(kappa, k, errors[0]) for k in iters]

            axes[idx].semilogy(iters, errors, "b-", label="Actual")
            axes[idx].semilogy(iters, bound, "r--", label="Theoretical bound")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("f(x_k) - f*")
            axes[idx].set_title(f"κ = {kappa}")
            axes[idx].legend()
            axes[idx].grid(True)
        except NotImplementedError:
            axes[idx].set_title(f"κ = {kappa} (not implemented)")

    plt.suptitle("GD Convergence vs Condition Number")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        data = convergence_study()
        for kappa, rate, theory_rate in data:
            print(f"kappa={kappa:4d}: measured rate={rate:.6f}, "
                  f"theoretical=({(kappa-1)/(kappa+1):.6f})^2="
                  f"{((kappa-1)/(kappa+1))**2:.6f}")
    except NotImplementedError:
        print("TODO: implement convergence_study()")

    plot_convergence()
