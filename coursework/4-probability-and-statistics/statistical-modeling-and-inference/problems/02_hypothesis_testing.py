"""
Problem: Hypothesis Testing

Implement classical hypothesis tests from scratch, computing test statistics
and p-values using numerical integration rather than library functions.

Tasks:
  (a) Implement the z-test for a population mean (known variance)
  (b) Implement the one-sample t-test (unknown variance)
  (c) Implement the chi-squared goodness-of-fit test
  (d) Compute p-values from scratch using numerical integration of the
      test statistic's distribution
  (e) Simulate Type I and Type II error rates
"""

import numpy as np


def z_test(data, mu_0, sigma, alternative="two-sided"):
    """
    Perform a z-test for the population mean (known variance).

    H0: mu = mu_0  vs  H1: depends on `alternative`

    Parameters:
        data: np.ndarray, shape (n,), observed samples
        mu_0: float, hypothesized population mean
        sigma: float, known population standard deviation
        alternative: str, one of "two-sided", "greater", "less"

    Returns:
        dict with keys:
            'z_statistic': float
            'p_value': float
            'reject': bool (at alpha=0.05)
    """
    # TODO
    raise NotImplementedError


def t_test(data, mu_0, alternative="two-sided"):
    """
    Perform a one-sample t-test for the population mean (unknown variance).

    H0: mu = mu_0

    Parameters:
        data: np.ndarray, shape (n,)
        mu_0: float, hypothesized mean
        alternative: str, "two-sided", "greater", or "less"

    Returns:
        dict with keys:
            't_statistic': float
            'degrees_of_freedom': int
            'p_value': float
            'reject': bool (at alpha=0.05)
    """
    # TODO
    raise NotImplementedError


def chi_squared_goodness_of_fit(observed, expected):
    """
    Perform a chi-squared goodness-of-fit test.

    H0: the observed frequencies follow the expected distribution.

    Parameters:
        observed: np.ndarray, shape (k,), observed counts
        expected: np.ndarray, shape (k,), expected counts

    Returns:
        dict with keys:
            'chi2_statistic': float, sum((O_i - E_i)^2 / E_i)
            'degrees_of_freedom': int, k - 1
            'p_value': float
            'reject': bool (at alpha=0.05)
    """
    # TODO
    raise NotImplementedError


def normal_cdf_numerical(x):
    """
    Compute the standard normal CDF using numerical integration
    (trapezoidal rule on the PDF).

    Parameters:
        x: float

    Returns:
        float: P(Z <= x)
    """
    # TODO
    raise NotImplementedError


def chi2_cdf_numerical(x, df):
    """
    Compute the chi-squared CDF by numerical integration of the
    chi-squared PDF.

    Parameters:
        x: float, the test statistic value
        df: int, degrees of freedom

    Returns:
        float: P(X <= x) where X ~ chi2(df)
    """
    # TODO
    raise NotImplementedError


def simulate_type1_error(n_samples, true_mu, sigma, alpha=0.05, n_simulations=10000,
                         rng=None):
    """
    Estimate Type I error rate by repeatedly testing H0: mu = true_mu
    when H0 is actually true.

    Parameters:
        n_samples: int, sample size per test
        true_mu: float, the true (and hypothesized) mean
        sigma: float, known standard deviation
        alpha: float, significance level
        n_simulations: int

    Returns:
        float: estimated Type I error rate (should be close to alpha)
    """
    # TODO
    raise NotImplementedError


def simulate_type2_error(n_samples, mu_0, true_mu, sigma, alpha=0.05,
                         n_simulations=10000, rng=None):
    """
    Estimate Type II error rate (probability of failing to reject H0
    when H1 is true).

    Parameters:
        n_samples: int, sample size per test
        mu_0: float, hypothesized mean under H0
        true_mu: float, actual population mean (different from mu_0)
        sigma: float, known standard deviation
        alpha: float, significance level
        n_simulations: int

    Returns:
        dict with keys:
            'type2_error': float, estimated beta
            'power': float, 1 - beta
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Test data: sample from Normal(5.2, 1)
    data = rng.normal(loc=5.2, scale=1.0, size=50)

    try:
        result = z_test(data, mu_0=5.0, sigma=1.0)
        print(f"Z-test: z={result['z_statistic']:.3f}, "
              f"p={result['p_value']:.4f}, reject={result['reject']}")
    except NotImplementedError:
        print("TODO: implement z_test")

    try:
        result = t_test(data, mu_0=5.0)
        print(f"T-test: t={result['t_statistic']:.3f}, "
              f"df={result['degrees_of_freedom']}, "
              f"p={result['p_value']:.4f}, reject={result['reject']}")
    except NotImplementedError:
        print("TODO: implement t_test")

    try:
        # Chi-squared: test if a die is fair
        observed = np.array([18, 15, 22, 17, 12, 16])  # 100 rolls
        expected = np.array([100 / 6] * 6)
        result = chi_squared_goodness_of_fit(observed, expected)
        print(f"\nChi-squared test: chi2={result['chi2_statistic']:.3f}, "
              f"p={result['p_value']:.4f}, reject={result['reject']}")
    except NotImplementedError:
        print("TODO: implement chi_squared_goodness_of_fit")

    try:
        rate = simulate_type1_error(30, true_mu=0.0, sigma=1.0)
        print(f"\nType I error rate: {rate:.4f} (expected ~0.05)")
    except NotImplementedError:
        print("TODO: implement simulate_type1_error")

    try:
        result = simulate_type2_error(30, mu_0=0.0, true_mu=0.5, sigma=1.0)
        print(f"Type II error: {result['type2_error']:.4f}, "
              f"Power: {result['power']:.4f}")
    except NotImplementedError:
        print("TODO: implement simulate_type2_error")
