"""
Gaussian Processes
==================

Implement Gaussian Process regression from scratch, including kernel functions,
prior sampling, posterior inference, and hyperparameter optimization.

Tasks
-----
1. GP Prior Sampling: Given a mean function and a kernel (covariance function),
   sample functions from the GP prior. Implement the squared exponential
   (RBF) kernel k(x, x') = sigma^2 * exp(-||x - x'||^2 / (2 * l^2)).

2. GP Regression: Given training data (X, y) and test points X*, compute the
   posterior mean and covariance. The key equations are:
     mu* = K(X*, X) [K(X, X) + sigma_n^2 I]^{-1} y
     Sigma* = K(X*, X*) - K(X*, X) [K(X, X) + sigma_n^2 I]^{-1} K(X, X*)

3. Posterior Mean and Variance: Implement functions to compute the posterior
   predictive mean and variance at new test points. Visualize the mean with
   uncertainty bands (mean +/- 2*std).

4. Hyperparameter Optimization: Optimize kernel hyperparameters (length scale l,
   signal variance sigma^2, noise variance sigma_n^2) by maximizing the log
   marginal likelihood:
     log p(y|X, theta) = -0.5 * y^T K_y^{-1} y - 0.5 * log|K_y| - n/2 * log(2*pi)
   Use gradient-based optimization (e.g., scipy.optimize.minimize or manual
   gradient descent).
"""

import numpy as np
import matplotlib.pyplot as plt


def squared_exponential_kernel(X1, X2, length_scale=1.0, signal_variance=1.0):
    """
    Compute the squared exponential (RBF) kernel matrix.

    Parameters
    ----------
    X1 : np.ndarray
        First set of points, shape (n1, d).
    X2 : np.ndarray
        Second set of points, shape (n2, d).
    length_scale : float
        Kernel length scale.
    signal_variance : float
        Signal variance (output scale).

    Returns
    -------
    K : np.ndarray
        Kernel matrix of shape (n1, n2).
    """
    raise NotImplementedError


def sample_gp_prior(X, kernel_fn, n_samples=5, **kernel_kwargs):
    """
    Sample functions from a GP prior with zero mean.

    Parameters
    ----------
    X : np.ndarray
        Input points of shape (n, d).
    kernel_fn : callable
        Kernel function that takes (X1, X2, **kwargs) and returns K.
    n_samples : int
        Number of function samples to draw.
    **kernel_kwargs
        Keyword arguments for the kernel function.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, n) containing the sampled function values.
    """
    raise NotImplementedError


def gp_posterior(X_train, y_train, X_test, kernel_fn, noise_variance=1e-4,
                 **kernel_kwargs):
    """
    Compute GP posterior mean and covariance.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs of shape (n_train, d).
    y_train : np.ndarray
        Training targets of shape (n_train,).
    X_test : np.ndarray
        Test inputs of shape (n_test, d).
    kernel_fn : callable
        Kernel function.
    noise_variance : float
        Observation noise variance sigma_n^2.
    **kernel_kwargs
        Keyword arguments for the kernel function.

    Returns
    -------
    mu : np.ndarray
        Posterior mean at test points, shape (n_test,).
    cov : np.ndarray
        Posterior covariance matrix, shape (n_test, n_test).
    """
    raise NotImplementedError


def plot_gp_regression(X_train, y_train, X_test, mu, cov, true_fn=None):
    """
    Plot GP regression results with uncertainty bands.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs (1D), shape (n_train,).
    y_train : np.ndarray
        Training targets, shape (n_train,).
    X_test : np.ndarray
        Test inputs (1D), shape (n_test,).
    mu : np.ndarray
        Posterior mean, shape (n_test,).
    cov : np.ndarray
        Posterior covariance, shape (n_test, n_test).
    true_fn : callable or None
        True function for comparison.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    raise NotImplementedError


def log_marginal_likelihood(X_train, y_train, kernel_fn, noise_variance,
                            **kernel_kwargs):
    """
    Compute the log marginal likelihood of the GP model.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs of shape (n_train, d).
    y_train : np.ndarray
        Training targets of shape (n_train,).
    kernel_fn : callable
        Kernel function.
    noise_variance : float
        Noise variance.
    **kernel_kwargs
        Kernel hyperparameters.

    Returns
    -------
    lml : float
        Log marginal likelihood.
    """
    raise NotImplementedError


def optimize_hyperparameters(X_train, y_train, kernel_fn):
    """
    Optimize GP hyperparameters by maximizing the log marginal likelihood.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs.
    y_train : np.ndarray
        Training targets.
    kernel_fn : callable
        Kernel function.

    Returns
    -------
    best_params : dict
        Optimized hyperparameters (length_scale, signal_variance, noise_variance).
    lml_history : list of float
        Log marginal likelihood at each optimization step.
    """
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(42)

    print("=== GP Prior Samples ===")
    X_grid = np.linspace(-5, 5, 200).reshape(-1, 1)
    prior_samples = sample_gp_prior(X_grid, squared_exponential_kernel,
                                     n_samples=5, length_scale=1.0)
    print(f"Sampled {prior_samples.shape[0]} functions on {prior_samples.shape[1]} points")

    print("\n=== GP Regression ===")
    # True function
    true_fn = lambda x: np.sin(x)
    X_train = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=float).reshape(-1, 1)
    y_train = true_fn(X_train.ravel()) + 0.1 * np.random.randn(len(X_train))
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)

    mu, cov = gp_posterior(X_train, y_train, X_test,
                           squared_exponential_kernel, noise_variance=0.01,
                           length_scale=1.0, signal_variance=1.0)
    print(f"Posterior mean range: [{mu.min():.3f}, {mu.max():.3f}]")
    std = np.sqrt(np.diag(cov))
    print(f"Posterior std range: [{std.min():.3f}, {std.max():.3f}]")

    print("\n=== Plot GP Regression ===")
    fig = plot_gp_regression(X_train.ravel(), y_train, X_test.ravel(), mu, cov,
                              true_fn=lambda x: np.sin(x))

    print("\n=== Log Marginal Likelihood ===")
    lml = log_marginal_likelihood(X_train, y_train, squared_exponential_kernel,
                                   noise_variance=0.01, length_scale=1.0,
                                   signal_variance=1.0)
    print(f"Log marginal likelihood: {lml:.4f}")

    print("\n=== Hyperparameter Optimization ===")
    best_params, lml_hist = optimize_hyperparameters(X_train, y_train,
                                                      squared_exponential_kernel)
    print(f"Optimized parameters: {best_params}")
    print(f"Final LML: {lml_hist[-1]:.4f}")
    plt.show()
