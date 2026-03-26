"""
Problem: Continuous Probability Distributions

Implement PDF, CDF, moments, and sampling for standard continuous distributions
from scratch (no scipy.stats).

Tasks:
  (a) Implement PDF and CDF for normal, exponential, and uniform distributions
  (b) Compute moments (mean, variance) by numerical integration
  (c) Generate random samples via the inverse CDF (quantile) method
"""

import numpy as np


def uniform_pdf(x, a=0.0, b=1.0):
    """
    Probability density function of Uniform(a, b).

    Parameters:
        x: float or np.ndarray
        a: float, lower bound
        b: float, upper bound

    Returns:
        float or np.ndarray: density value(s)
    """
    # TODO
    raise NotImplementedError


def uniform_cdf(x, a=0.0, b=1.0):
    """
    Cumulative distribution function of Uniform(a, b).

    Parameters:
        x: float or np.ndarray
        a: float, lower bound
        b: float, upper bound

    Returns:
        float or np.ndarray: CDF value(s)
    """
    # TODO
    raise NotImplementedError


def exponential_pdf(x, lam=1.0):
    """
    Probability density function of Exponential(lambda).

    Parameters:
        x: float or np.ndarray
        lam: float, rate parameter (lambda > 0)

    Returns:
        float or np.ndarray: density value(s)
    """
    # TODO
    raise NotImplementedError


def exponential_cdf(x, lam=1.0):
    """
    Cumulative distribution function of Exponential(lambda).

    Parameters:
        x: float or np.ndarray
        lam: float, rate parameter

    Returns:
        float or np.ndarray: CDF value(s)
    """
    # TODO
    raise NotImplementedError


def normal_pdf(x, mu=0.0, sigma=1.0):
    """
    Probability density function of Normal(mu, sigma^2).

    Parameters:
        x: float or np.ndarray
        mu: float, mean
        sigma: float, standard deviation (> 0)

    Returns:
        float or np.ndarray: density value(s)
    """
    # TODO
    raise NotImplementedError


def normal_cdf(x, mu=0.0, sigma=1.0):
    """
    Cumulative distribution function of Normal(mu, sigma^2).

    Implement using numerical integration (e.g., trapezoidal rule on the PDF)
    or an approximation of the error function.

    Parameters:
        x: float or np.ndarray
        mu: float, mean
        sigma: float, standard deviation

    Returns:
        float or np.ndarray: CDF value(s)
    """
    # TODO
    raise NotImplementedError


def compute_moments(pdf_fn, a, b, n_points=10000):
    """
    Compute the mean and variance of a distribution by numerical integration
    of its PDF over [a, b] using the trapezoidal rule.

    Parameters:
        pdf_fn: callable, the PDF function f(x)
        a: float, lower integration bound
        b: float, upper integration bound
        n_points: int, number of quadrature points

    Returns:
        dict with keys:
            'mean': float, E[X]
            'variance': float, Var(X) = E[X^2] - (E[X])^2
    """
    # TODO
    raise NotImplementedError


def inverse_cdf_sample(inverse_cdf_fn, n_samples, rng=None):
    """
    Generate random samples using the inverse CDF (quantile function) method.

    Draw u ~ Uniform(0, 1), then return inverse_cdf_fn(u).

    Parameters:
        inverse_cdf_fn: callable, the quantile function F^{-1}(u)
        n_samples: int
        rng: np.random.Generator or None

    Returns:
        np.ndarray, shape (n_samples,): random samples
    """
    # TODO
    raise NotImplementedError


def exponential_inverse_cdf(u, lam=1.0):
    """
    Inverse CDF (quantile function) for Exponential(lambda).

    F^{-1}(u) = -ln(1 - u) / lambda

    Parameters:
        u: float or np.ndarray, values in (0, 1)
        lam: float, rate parameter

    Returns:
        float or np.ndarray
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    try:
        # Uniform(0, 1) PDF at 0.5
        print(f"Uniform PDF at 0.5: {uniform_pdf(0.5)}")
        print(f"Uniform CDF at 0.5: {uniform_cdf(0.5)}")
    except NotImplementedError:
        print("TODO: implement uniform PDF/CDF")

    try:
        print(f"\nExponential(1) PDF at 1: {exponential_pdf(1.0):.4f}")
        print(f"Exponential(1) CDF at 1: {exponential_cdf(1.0):.4f}")
    except NotImplementedError:
        print("TODO: implement exponential PDF/CDF")

    try:
        print(f"\nNormal(0,1) PDF at 0: {normal_pdf(0.0):.4f}")
        print(f"Normal(0,1) CDF at 0: {normal_cdf(0.0):.4f} (expected ~0.5)")
    except NotImplementedError:
        print("TODO: implement normal PDF/CDF")

    try:
        moments = compute_moments(lambda x: exponential_pdf(x, 2.0), 0, 20)
        print(f"\nExponential(2) mean: {moments['mean']:.4f} (expected 0.5)")
        print(f"Exponential(2) variance: {moments['variance']:.4f} (expected 0.25)")
    except NotImplementedError:
        print("TODO: implement compute_moments")

    try:
        samples = inverse_cdf_sample(
            lambda u: exponential_inverse_cdf(u, 1.0), 10000
        )
        print(f"\nInverse CDF samples (Exp(1)): mean={np.mean(samples):.3f}, "
              f"std={np.std(samples):.3f}")
    except NotImplementedError:
        print("TODO: implement inverse_cdf_sample / exponential_inverse_cdf")
