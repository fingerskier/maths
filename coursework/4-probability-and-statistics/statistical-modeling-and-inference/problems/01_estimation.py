"""
Problem: Statistical Estimation

Implement maximum likelihood estimation, method of moments, confidence
intervals, and bootstrap estimation.

Tasks:
  (a) Implement MLE for normal, Poisson, and exponential distributions
  (b) Implement method of moments estimation for the same distributions
  (c) Compute confidence intervals for the mean (known and unknown variance)
  (d) Implement bootstrap estimation of standard errors and confidence intervals
"""

import numpy as np


def mle_normal(data):
    """
    Maximum likelihood estimators for Normal(mu, sigma^2).

    MLE: mu_hat = sample mean, sigma_hat^2 = (1/n) * sum((x_i - mu_hat)^2)
    (Note: MLE uses 1/n, not 1/(n-1).)

    Parameters:
        data: np.ndarray, shape (n,), observed samples

    Returns:
        dict with keys:
            'mu': float, estimated mean
            'sigma2': float, estimated variance (MLE, biased)
    """
    # TODO
    raise NotImplementedError


def mle_poisson(data):
    """
    Maximum likelihood estimator for Poisson(lambda).

    MLE: lambda_hat = sample mean.

    Parameters:
        data: np.ndarray, shape (n,), observed counts

    Returns:
        float: estimated lambda
    """
    # TODO
    raise NotImplementedError


def mle_exponential(data):
    """
    Maximum likelihood estimator for Exponential(lambda).

    MLE: lambda_hat = 1 / sample_mean.

    Parameters:
        data: np.ndarray, shape (n,), observed samples (positive values)

    Returns:
        float: estimated rate parameter lambda
    """
    # TODO
    raise NotImplementedError


def mom_normal(data):
    """
    Method of moments estimators for Normal(mu, sigma^2).

    First moment: mu = sample mean.
    Second central moment: sigma^2 = sample variance (1/n version).

    Parameters:
        data: np.ndarray, shape (n,)

    Returns:
        dict with keys 'mu' and 'sigma2'
    """
    # TODO
    raise NotImplementedError


def mom_poisson(data):
    """
    Method of moments estimator for Poisson(lambda).

    First moment: lambda = sample mean.

    Parameters:
        data: np.ndarray, shape (n,)

    Returns:
        float: estimated lambda
    """
    # TODO
    raise NotImplementedError


def mom_exponential(data):
    """
    Method of moments estimator for Exponential(lambda).

    First moment: E[X] = 1/lambda, so lambda = 1/sample_mean.

    Parameters:
        data: np.ndarray, shape (n,)

    Returns:
        float: estimated lambda
    """
    # TODO
    raise NotImplementedError


def confidence_interval_mean(data, confidence=0.95, sigma_known=None):
    """
    Compute a confidence interval for the population mean.

    If sigma_known is provided, use the z-interval.
    Otherwise, use the t-interval with n-1 degrees of freedom
    (approximate the t critical value).

    Parameters:
        data: np.ndarray, shape (n,)
        confidence: float, confidence level (e.g., 0.95)
        sigma_known: float or None

    Returns:
        tuple[float, float]: (lower_bound, upper_bound)
    """
    # TODO
    raise NotImplementedError


def bootstrap_estimate(data, statistic_fn, n_bootstrap=10000, confidence=0.95,
                       rng=None):
    """
    Estimate the standard error and confidence interval of a statistic
    using the nonparametric bootstrap.

    Parameters:
        data: np.ndarray, shape (n,), original sample
        statistic_fn: callable, takes an array and returns a scalar
        n_bootstrap: int, number of bootstrap resamples
        confidence: float, confidence level for the interval
        rng: np.random.Generator or None

    Returns:
        dict with keys:
            'estimate': float, statistic computed on original data
            'std_error': float, bootstrap standard error
            'ci_lower': float, lower bound of bootstrap percentile CI
            'ci_upper': float, upper bound of bootstrap percentile CI
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Generate samples from known distributions
    normal_data = rng.normal(loc=5.0, scale=2.0, size=100)
    poisson_data = rng.poisson(lam=3.5, size=100)
    exp_data = rng.exponential(scale=2.0, size=100)  # scale = 1/lambda

    try:
        est = mle_normal(normal_data)
        print(f"MLE Normal: mu={est['mu']:.3f} (true 5.0), "
              f"sigma2={est['sigma2']:.3f} (true 4.0)")
    except NotImplementedError:
        print("TODO: implement mle_normal")

    try:
        lam = mle_poisson(poisson_data)
        print(f"MLE Poisson: lambda={lam:.3f} (true 3.5)")
    except NotImplementedError:
        print("TODO: implement mle_poisson")

    try:
        lam = mle_exponential(exp_data)
        print(f"MLE Exponential: lambda={lam:.3f} (true 0.5)")
    except NotImplementedError:
        print("TODO: implement mle_exponential")

    try:
        ci = confidence_interval_mean(normal_data, confidence=0.95)
        print(f"\n95% CI for mean: ({ci[0]:.3f}, {ci[1]:.3f})")
    except NotImplementedError:
        print("TODO: implement confidence_interval_mean")

    try:
        result = bootstrap_estimate(normal_data, np.median)
        print(f"\nBootstrap median: {result['estimate']:.3f}")
        print(f"Bootstrap SE: {result['std_error']:.3f}")
        print(f"Bootstrap 95% CI: ({result['ci_lower']:.3f}, {result['ci_upper']:.3f})")
    except NotImplementedError:
        print("TODO: implement bootstrap_estimate")
