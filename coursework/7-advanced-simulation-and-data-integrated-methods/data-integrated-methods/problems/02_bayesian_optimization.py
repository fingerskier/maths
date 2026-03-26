"""
Bayesian Optimization
=====================

Bayesian optimization is a sequential strategy for optimizing expensive
black-box functions. It builds a probabilistic surrogate model (typically a
Gaussian Process) of the objective function and uses an acquisition function
to decide where to evaluate next, balancing exploration and exploitation.

The algorithm:
    1. Evaluate the objective at a few initial points.
    2. Fit a GP to all observations.
    3. Maximize the acquisition function to find the next query point.
    4. Evaluate the objective at that point and update the GP.
    5. Repeat until budget is exhausted.

Tasks
-----
1. Implement a Bayesian optimization loop using a Gaussian Process surrogate
   and an acquisition function. The GP should provide posterior mean and
   variance predictions at any test point.

2. Implement the Expected Improvement (EI) acquisition function:
       EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
   where z = (f_best - mu(x)) / sigma(x), Phi is the standard normal CDF,
   and phi is the standard normal PDF.

3. Optimize a noisy 1D black-box function (e.g., negative Branin or a
   multi-modal function like -(x * sin(x)) on [0, 10]). Track the best
   observed value over iterations.

4. Compare Bayesian optimization convergence with random search. Plot the
   best-found value vs number of evaluations for both methods.
"""

import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Compute the RBF (squared exponential) kernel matrix.

    Parameters
    ----------
    x1 : np.ndarray
        First set of points, shape (n1, d).
    x2 : np.ndarray
        Second set of points, shape (n2, d).
    length_scale : float
        Kernel length scale.
    variance : float
        Kernel signal variance.

    Returns
    -------
    K : np.ndarray
        Kernel matrix of shape (n1, n2).
    """
    raise NotImplementedError


def gp_predict(X_train, y_train, X_test, length_scale=1.0, variance=1.0,
               noise=1e-6):
    """
    Gaussian Process posterior prediction.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs, shape (n, d).
    y_train : np.ndarray
        Training outputs, shape (n,).
    X_test : np.ndarray
        Test inputs, shape (m, d).
    length_scale : float
        Kernel length scale.
    variance : float
        Kernel signal variance.
    noise : float
        Observation noise variance.

    Returns
    -------
    mu : np.ndarray
        Posterior mean at test points, shape (m,).
    sigma : np.ndarray
        Posterior standard deviation at test points, shape (m,).
    """
    raise NotImplementedError


def expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement acquisition function.

    Parameters
    ----------
    mu : np.ndarray
        GP posterior mean at candidate points, shape (m,).
    sigma : np.ndarray
        GP posterior std at candidate points, shape (m,).
    f_best : float
        Best (minimum) observed function value so far.

    Returns
    -------
    ei : np.ndarray
        Expected improvement values, shape (m,).
    """
    raise NotImplementedError


def objective_function(x, noise_std=0.1):
    """
    Noisy black-box objective function to minimize.

    f(x) = -(x * sin(x)) + noise,  x in [0, 10]

    Parameters
    ----------
    x : float or np.ndarray
        Input point(s).
    noise_std : float
        Standard deviation of observation noise.

    Returns
    -------
    float or np.ndarray
        Noisy function value(s).
    """
    raise NotImplementedError


def bayesian_optimization(objective, bounds, n_init=5, n_iter=25,
                          length_scale=1.0, variance=1.0, noise=0.1):
    """
    Run Bayesian optimization loop.

    Parameters
    ----------
    objective : callable
        Black-box function to minimize.
    bounds : tuple
        (lower, upper) bounds for the 1D input.
    n_init : int
        Number of initial random evaluations.
    n_iter : int
        Number of BO iterations after initialization.
    length_scale : float
        GP kernel length scale.
    variance : float
        GP kernel signal variance.
    noise : float
        Assumed observation noise level.

    Returns
    -------
    X_observed : np.ndarray
        All evaluated points, shape (n_init + n_iter, 1).
    y_observed : np.ndarray
        All observed values, shape (n_init + n_iter,).
    best_values : list of float
        Best observed value after each evaluation.
    """
    raise NotImplementedError


def random_search(objective, bounds, n_evals=30):
    """
    Random search baseline for comparison.

    Parameters
    ----------
    objective : callable
        Black-box function to minimize.
    bounds : tuple
        (lower, upper) bounds.
    n_evals : int
        Total number of function evaluations.

    Returns
    -------
    X_observed : np.ndarray
        All evaluated points, shape (n_evals, 1).
    y_observed : np.ndarray
        All observed values, shape (n_evals,).
    best_values : list of float
        Running best observed value.
    """
    raise NotImplementedError


def plot_comparison(bo_best, rs_best):
    """
    Plot convergence comparison of Bayesian optimization vs random search.

    Parameters
    ----------
    bo_best : list of float
        Best values from Bayesian optimization at each iteration.
    rs_best : list of float
        Best values from random search at each iteration.
    """
    raise NotImplementedError


if __name__ == "__main__":
    bounds = (0.0, 10.0)

    # Task 1 & 2: Bayesian optimization with EI
    X_bo, y_bo, bo_best = bayesian_optimization(
        objective_function, bounds, n_init=5, n_iter=25
    )
    print(f"BO best value: {min(bo_best):.4f} at x={X_bo[np.argmin(y_bo), 0]:.4f}")

    # Task 3: Show optimization progress
    print("\nBO convergence:")
    for i in [0, 5, 10, 15, 20, 29]:
        if i < len(bo_best):
            print(f"  After {i+1} evals: best = {bo_best[i]:.4f}")

    # Task 4: Compare with random search
    np.random.seed(0)
    X_rs, y_rs, rs_best = random_search(objective_function, bounds, n_evals=30)
    print(f"\nRandom search best value: {min(rs_best):.4f}")

    plot_comparison(bo_best, rs_best)
    plt.savefig("bo_vs_random_search.png", dpi=150, bbox_inches="tight")
    print("Comparison plot saved.")
