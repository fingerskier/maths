"""
Markov Chain Monte Carlo (MCMC) Sampling
=========================================

Implement MCMC algorithms for sampling from distributions that are difficult
to sample from directly.

Tasks
-----
1. Metropolis-Hastings Sampler: Implement the Metropolis-Hastings algorithm for
   sampling from a target distribution given only an unnormalized density. Use a
   symmetric Gaussian proposal. Test on a mixture of Gaussians.

2. Gibbs Sampling: Implement Gibbs sampling for a bivariate normal distribution
   by alternately sampling from the conditional distributions. Compare the
   resulting samples with the true distribution.

3. Convergence Diagnostics: For a chain produced by Metropolis-Hastings, produce
   trace plots, compute autocorrelation as a function of lag, and estimate the
   effective sample size.

4. Burn-in Analysis: Run multiple chains from different starting points and
   analyse how long it takes for them to converge to the stationary distribution.
   Implement the Gelman-Rubin diagnostic (R-hat statistic).
"""

import numpy as np
import matplotlib.pyplot as plt


def metropolis_hastings(log_target, x0, proposal_std, n_samples, burn_in=0):
    """
    Metropolis-Hastings sampler with Gaussian proposal.

    Parameters
    ----------
    log_target : callable
        Log of the (unnormalized) target density.
    x0 : np.ndarray
        Initial state (d-dimensional).
    proposal_std : float
        Standard deviation of the Gaussian proposal.
    n_samples : int
        Number of samples to generate (after burn-in).
    burn_in : int
        Number of initial samples to discard.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, d) containing the samples.
    acceptance_rate : float
        Fraction of proposals accepted.
    """
    raise NotImplementedError


def gibbs_bivariate_normal(mu, sigma, n_samples, burn_in=0):
    """
    Gibbs sampler for a bivariate normal distribution.

    Parameters
    ----------
    mu : np.ndarray
        Mean vector (2,).
    sigma : np.ndarray
        Covariance matrix (2, 2).
    n_samples : int
        Number of samples to generate (after burn-in).
    burn_in : int
        Number of initial samples to discard.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, 2).
    """
    raise NotImplementedError


def trace_plot(samples, param_names=None):
    """
    Create trace plots for MCMC samples.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n_samples, d).
    param_names : list of str or None
        Names for each parameter dimension.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    raise NotImplementedError


def autocorrelation(samples, max_lag=100):
    """
    Compute the autocorrelation function for each parameter.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n_samples,) or (n_samples, d).
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    acf : np.ndarray
        Autocorrelation values of shape (max_lag + 1,) or (max_lag + 1, d).
    """
    raise NotImplementedError


def effective_sample_size(samples):
    """
    Estimate the effective sample size from autocorrelation.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n_samples,) or (n_samples, d).

    Returns
    -------
    ess : float or np.ndarray
        Effective sample size for each parameter.
    """
    raise NotImplementedError


def gelman_rubin_diagnostic(chains):
    """
    Compute the Gelman-Rubin R-hat statistic for convergence assessment.

    Parameters
    ----------
    chains : list of np.ndarray
        List of m chains, each of shape (n_samples,).

    Returns
    -------
    r_hat : float
        Gelman-Rubin R-hat statistic. Values close to 1 indicate convergence.
    """
    raise NotImplementedError


def burn_in_analysis(log_target, starting_points, proposal_std, n_samples):
    """
    Run multiple chains and analyse burn-in using the Gelman-Rubin diagnostic.

    Parameters
    ----------
    log_target : callable
        Log of the target density.
    starting_points : list of np.ndarray
        Initial states for each chain.
    proposal_std : float
        Proposal standard deviation.
    n_samples : int
        Number of samples per chain.

    Returns
    -------
    chains : list of np.ndarray
        The sampled chains.
    r_hat_history : list of float
        R-hat computed using increasing fractions of the chains.
    fig : matplotlib.figure.Figure
        Figure showing convergence diagnostics.
    """
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Metropolis-Hastings ===")
    # Target: mixture of two Gaussians
    def log_target(x):
        return np.log(0.5 * np.exp(-0.5 * (x[0] - 2)**2) +
                       0.5 * np.exp(-0.5 * (x[0] + 2)**2))

    samples, acc_rate = metropolis_hastings(log_target, np.array([0.0]),
                                            proposal_std=1.0, n_samples=10000,
                                            burn_in=1000)
    print(f"Acceptance rate: {acc_rate:.3f}")
    print(f"Sample mean: {np.mean(samples):.3f}, std: {np.std(samples):.3f}")

    print("\n=== Gibbs Sampling (Bivariate Normal) ===")
    mu = np.array([1.0, 2.0])
    sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples_gibbs = gibbs_bivariate_normal(mu, sigma, n_samples=10000, burn_in=500)
    print(f"Sample mean: {np.mean(samples_gibbs, axis=0)}")
    print(f"Sample cov:\n{np.cov(samples_gibbs.T)}")

    print("\n=== Convergence Diagnostics ===")
    fig_trace = trace_plot(samples_gibbs, param_names=["x1", "x2"])
    acf = autocorrelation(samples_gibbs[:, 0], max_lag=50)
    ess = effective_sample_size(samples_gibbs)
    print(f"Effective sample sizes: {ess}")

    print("\n=== Burn-in Analysis ===")
    starts = [np.array([s]) for s in [-10.0, -5.0, 0.0, 5.0, 10.0]]
    chains, r_hat_hist, fig_burn = burn_in_analysis(log_target, starts,
                                                     proposal_std=1.0,
                                                     n_samples=5000)
    print(f"Final R-hat: {r_hat_hist[-1]:.4f}")
    plt.show()
