"""
Problem: Bayesian Inference

Implement Bayesian estimation with conjugate priors and simple MCMC sampling.

Tasks:
  (a) Compute the posterior distribution for Beta-Binomial conjugate model
  (b) Compute the posterior for Normal-Normal conjugate model (known variance)
  (c) Implement the Metropolis-Hastings MCMC algorithm for arbitrary posteriors
  (d) Compare Bayesian credible intervals with frequentist confidence intervals
"""

import numpy as np


def beta_binomial_posterior(n_successes, n_trials, prior_alpha=1.0,
                            prior_beta=1.0):
    """
    Compute the posterior parameters for a Beta-Binomial model.

    Prior: theta ~ Beta(alpha, beta)
    Likelihood: X | theta ~ Binomial(n, theta)
    Posterior: theta | X ~ Beta(alpha + k, beta + n - k)

    Parameters:
        n_successes: int, number of successes observed
        n_trials: int, total number of trials
        prior_alpha: float, Beta prior alpha parameter
        prior_beta: float, Beta prior beta parameter

    Returns:
        dict with keys:
            'posterior_alpha': float
            'posterior_beta': float
            'posterior_mean': float, alpha / (alpha + beta) for posterior
            'posterior_mode': float, (alpha - 1) / (alpha + beta - 2) if valid
            'credible_interval_95': tuple[float, float], via Beta quantiles
    """
    # TODO
    raise NotImplementedError


def normal_normal_posterior(data, prior_mu, prior_sigma2, likelihood_sigma2):
    """
    Compute the posterior for the Normal-Normal conjugate model
    (known likelihood variance).

    Prior: mu ~ N(prior_mu, prior_sigma2)
    Likelihood: X_i | mu ~ N(mu, likelihood_sigma2)
    Posterior: mu | data ~ N(posterior_mu, posterior_sigma2)

    Parameters:
        data: np.ndarray, shape (n,), observed samples
        prior_mu: float, prior mean
        prior_sigma2: float, prior variance
        likelihood_sigma2: float, known variance of the likelihood

    Returns:
        dict with keys:
            'posterior_mu': float
            'posterior_sigma2': float
            'credible_interval_95': tuple[float, float]
    """
    # TODO
    raise NotImplementedError


def metropolis_hastings(log_posterior_fn, initial, n_samples, proposal_std=1.0,
                        burn_in=1000, rng=None):
    """
    Run the Metropolis-Hastings MCMC algorithm with a Gaussian proposal.

    Parameters:
        log_posterior_fn: callable, takes a parameter value and returns
                          the log of the (unnormalized) posterior density
        initial: float, starting value
        n_samples: int, number of samples to collect (after burn-in)
        proposal_std: float, standard deviation of the Gaussian proposal
        burn_in: int, number of initial samples to discard
        rng: np.random.Generator or None

    Returns:
        dict with keys:
            'samples': np.ndarray, shape (n_samples,), posterior samples
            'acceptance_rate': float, fraction of proposals accepted
    """
    # TODO
    raise NotImplementedError


def bayesian_vs_frequentist(data, prior_mu, prior_sigma2, likelihood_sigma2,
                            confidence=0.95):
    """
    Compare Bayesian credible interval with frequentist confidence interval
    for the mean.

    Parameters:
        data: np.ndarray, shape (n,)
        prior_mu: float
        prior_sigma2: float
        likelihood_sigma2: float
        confidence: float

    Returns:
        dict with keys:
            'bayesian_ci': tuple[float, float], Bayesian credible interval
            'frequentist_ci': tuple[float, float], frequentist confidence interval
            'bayesian_mean': float, posterior mean
            'frequentist_mean': float, sample mean
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Beta-Binomial: coin flipping
    try:
        result = beta_binomial_posterior(
            n_successes=7, n_trials=10,
            prior_alpha=2, prior_beta=2
        )
        print("Beta-Binomial posterior:")
        print(f"  Alpha={result['posterior_alpha']}, Beta={result['posterior_beta']}")
        print(f"  Mean={result['posterior_mean']:.4f}")
        print(f"  95% CI: {result['credible_interval_95']}")
    except NotImplementedError:
        print("TODO: implement beta_binomial_posterior")

    # Normal-Normal
    data = rng.normal(loc=5.0, scale=1.0, size=20)
    try:
        result = normal_normal_posterior(
            data, prior_mu=0.0, prior_sigma2=10.0, likelihood_sigma2=1.0
        )
        print(f"\nNormal-Normal posterior:")
        print(f"  Mean={result['posterior_mu']:.4f}")
        print(f"  Variance={result['posterior_sigma2']:.4f}")
        print(f"  95% CI: ({result['credible_interval_95'][0]:.4f}, "
              f"{result['credible_interval_95'][1]:.4f})")
    except NotImplementedError:
        print("TODO: implement normal_normal_posterior")

    # Metropolis-Hastings: sample from a normal posterior
    try:
        def log_post(x):
            # Log of N(3, 1) density (up to constant)
            return -0.5 * (x - 3.0) ** 2

        result = metropolis_hastings(log_post, initial=0.0, n_samples=5000)
        samples = result['samples']
        print(f"\nMetropolis-Hastings:")
        print(f"  Acceptance rate: {result['acceptance_rate']:.3f}")
        print(f"  Sample mean: {np.mean(samples):.3f} (expected ~3.0)")
        print(f"  Sample std: {np.std(samples):.3f} (expected ~1.0)")
    except NotImplementedError:
        print("TODO: implement metropolis_hastings")

    # Comparison
    try:
        comp = bayesian_vs_frequentist(
            data, prior_mu=0.0, prior_sigma2=10.0, likelihood_sigma2=1.0
        )
        print(f"\nBayesian vs Frequentist:")
        print(f"  Bayesian CI: ({comp['bayesian_ci'][0]:.3f}, "
              f"{comp['bayesian_ci'][1]:.3f})")
        print(f"  Frequentist CI: ({comp['frequentist_ci'][0]:.3f}, "
              f"{comp['frequentist_ci'][1]:.3f})")
    except NotImplementedError:
        print("TODO: implement bayesian_vs_frequentist")
