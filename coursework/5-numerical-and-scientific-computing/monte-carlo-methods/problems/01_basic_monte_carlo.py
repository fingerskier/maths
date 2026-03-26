"""
Basic Monte Carlo Methods
=========================

Implement fundamental Monte Carlo techniques for estimation and integration.

Tasks
-----
1. Estimate Pi: Use the classic Monte Carlo method of sampling random points in
   the unit square and checking if they fall inside the unit circle. Estimate pi
   as 4 * (points inside circle) / (total points). Track convergence as a
   function of sample size.

2. Monte Carlo Integration: Estimate the integral of a multidimensional function
   over a hyperrectangular domain using simple Monte Carlo sampling. Implement for
   arbitrary dimension d and return the estimate with a confidence interval.

3. Importance Sampling: Implement importance sampling to estimate integrals more
   efficiently. Given a target integrand and a proposal distribution, compute the
   weighted estimate. Demonstrate on an integral where simple Monte Carlo has
   high variance (e.g., integral of exp(-x) * x^4 for x in [0, inf]).

4. Variance Reduction Techniques: Implement antithetic variates and control
   variates. Compare the variance of the estimator with and without these
   techniques on a test integral.
"""

import numpy as np


def estimate_pi(n_samples):
    """
    Estimate pi using Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
        Number of random points to sample.

    Returns
    -------
    pi_estimate : float
        Estimate of pi.
    estimates : np.ndarray
        Running estimate of pi after each sample (for convergence analysis).
    """
    raise NotImplementedError


def mc_integrate(f, bounds, n_samples=100000):
    """
    Monte Carlo integration of f over a hyperrectangular domain.

    Parameters
    ----------
    f : callable
        Function to integrate. Accepts an array of shape (d,) and returns a scalar.
    bounds : list of tuples
        List of (lower, upper) bounds for each dimension.
    n_samples : int
        Number of random samples.

    Returns
    -------
    estimate : float
        Estimated integral value.
    std_error : float
        Standard error of the estimate.
    """
    raise NotImplementedError


def importance_sampling(f, proposal_sampler, proposal_pdf, target_pdf, n_samples=100000):
    """
    Estimate E_target[f(X)] using importance sampling.

    Parameters
    ----------
    f : callable
        Function to integrate.
    proposal_sampler : callable
        Function that returns n_samples draws from the proposal distribution.
    proposal_pdf : callable
        PDF of the proposal distribution.
    target_pdf : callable
        PDF of the target distribution (unnormalized is fine if consistent).
    n_samples : int
        Number of samples.

    Returns
    -------
    estimate : float
        Importance sampling estimate.
    effective_sample_size : float
        Effective sample size (measure of sampling efficiency).
    """
    raise NotImplementedError


def antithetic_variates(f, n_samples=100000):
    """
    Estimate integral of f over [0,1] using antithetic variates.

    Parameters
    ----------
    f : callable
        Function to integrate over [0, 1].
    n_samples : int
        Number of base samples (total evaluations = 2 * n_samples).

    Returns
    -------
    estimate : float
        Antithetic variates estimate.
    variance : float
        Variance of the estimator.
    naive_variance : float
        Variance of the naive Monte Carlo estimator (for comparison).
    """
    raise NotImplementedError


def control_variates(f, g, expected_g, n_samples=100000):
    """
    Estimate integral of f over [0,1] using control variates.

    Parameters
    ----------
    f : callable
        Function to integrate over [0, 1].
    g : callable
        Control variate function (correlated with f, known expectation).
    expected_g : float
        Known expectation E[g(U)] for U ~ Uniform(0,1).
    n_samples : int
        Number of samples.

    Returns
    -------
    estimate : float
        Control variate estimate.
    variance : float
        Variance of the control variate estimator.
    naive_variance : float
        Variance of the naive estimator (for comparison).
    """
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Estimate Pi ===")
    pi_est, estimates = estimate_pi(1000000)
    print(f"Pi estimate: {pi_est:.6f} (error: {abs(pi_est - np.pi):.6f})")

    print("\n=== Monte Carlo Integration ===")
    # Integral of sin(x)*cos(y) over [0, pi] x [0, pi/2] = 1.0
    f = lambda x: np.sin(x[0]) * np.cos(x[1])
    est, se = mc_integrate(f, [(0, np.pi), (0, np.pi / 2)])
    print(f"Estimate: {est:.6f} +/- {se:.6f} (exact: 1.0)")

    print("\n=== Importance Sampling ===")
    # Estimate E[X^4] where X ~ Exp(1), exact = 24
    f_is = lambda x: x**4
    proposal_sampler = lambda n: np.random.exponential(0.5, n)
    proposal_pdf = lambda x: 2 * np.exp(-2 * x)
    target_pdf = lambda x: np.exp(-x)
    est_is, ess = importance_sampling(f_is, proposal_sampler, proposal_pdf, target_pdf)
    print(f"Estimate: {est_is:.4f} (exact: 24), ESS: {ess:.0f}")

    print("\n=== Antithetic Variates ===")
    f_av = lambda x: np.exp(x)  # integral over [0,1] = e - 1
    est_av, var_av, var_naive = antithetic_variates(f_av)
    print(f"Estimate: {est_av:.6f} (exact: {np.e - 1:.6f})")
    print(f"Variance reduction: {var_naive / var_av:.2f}x")

    print("\n=== Control Variates ===")
    f_cv = lambda x: np.exp(x)
    g_cv = lambda x: 1 + x  # E[g(U)] = 1.5
    est_cv, var_cv, var_naive = control_variates(f_cv, g_cv, 1.5)
    print(f"Estimate: {est_cv:.6f} (exact: {np.e - 1:.6f})")
    print(f"Variance reduction: {var_naive / var_cv:.2f}x")
