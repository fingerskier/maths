"""
Variational Inference
=====================

Implement variational inference for a Gaussian mixture model, compare with
the EM algorithm, and analyse convergence.

Tasks
-----
1. Mean-Field Variational Inference for GMM: Implement mean-field variational
   inference for a Gaussian mixture model with K components. The variational
   distribution factorizes as q(z, mu, pi) = q(z) * q(mu) * q(pi), where z
   are cluster assignments, mu are component means, and pi are mixing weights.
   Derive and implement the coordinate ascent update equations.

2. Evidence Lower Bound (ELBO): Implement computation of the ELBO:
     ELBO = E_q[log p(X, Z, mu, pi)] - E_q[log q(Z, mu, pi)]
   Monitor the ELBO across iterations to verify monotonic increase (a key
   property of coordinate ascent VI).

3. EM Algorithm Comparison: Implement the EM algorithm for the same GMM.
   Compare the solutions found by VI and EM: cluster assignments, learned
   parameters, and convergence speed.

4. Convergence Analysis: Track the ELBO and parameter values across iterations.
   Experiment with different initializations and K values. Plot convergence
   curves and analyse sensitivity to initialization.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_elbo(X, r, means, covariances, weights):
    """
    Compute the Evidence Lower Bound for a Gaussian mixture model.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features).
    r : np.ndarray
        Responsibilities of shape (n_samples, K).
    means : np.ndarray
        Component means of shape (K, n_features).
    covariances : np.ndarray
        Component covariances of shape (K, n_features, n_features).
    weights : np.ndarray
        Mixing weights of shape (K,).

    Returns
    -------
    elbo : float
        The evidence lower bound.
    """
    raise NotImplementedError


class VariationalGMM:
    """Mean-field variational inference for Gaussian mixture models."""

    def __init__(self, n_components=3, max_iter=100, tol=1e-6, random_state=None):
        """
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        max_iter : int
            Maximum number of VI iterations.
        tol : float
            Convergence tolerance on ELBO change.
        random_state : int or None
            Random seed.
        """
        raise NotImplementedError

    def fit(self, X):
        """
        Fit the variational GMM to data.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Assign data points to clusters.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        labels : np.ndarray
            Cluster assignments of shape (n_samples,).
        """
        raise NotImplementedError

    @property
    def elbo_history(self):
        """List of ELBO values across iterations."""
        raise NotImplementedError


class EMGMM:
    """Expectation-Maximization for Gaussian mixture models."""

    def __init__(self, n_components=3, max_iter=100, tol=1e-6, random_state=None):
        """
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        max_iter : int
            Maximum number of EM iterations.
        tol : float
            Convergence tolerance on log-likelihood change.
        random_state : int or None
            Random seed.
        """
        raise NotImplementedError

    def fit(self, X):
        """
        Fit the GMM using the EM algorithm.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Assign data points to clusters.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        labels : np.ndarray
            Cluster assignments.
        """
        raise NotImplementedError

    @property
    def log_likelihood_history(self):
        """List of log-likelihood values across iterations."""
        raise NotImplementedError


def compare_vi_and_em(X, n_components=3, random_state=None):
    """
    Compare variational inference and EM on the same data.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    random_state : int or None
        Random seed.

    Returns
    -------
    vi_model : VariationalGMM
        Fitted VI model.
    em_model : EMGMM
        Fitted EM model.
    fig : matplotlib.figure.Figure
        Comparison plots (convergence curves, cluster assignments).
    """
    raise NotImplementedError


def convergence_analysis(X, n_components=3, n_runs=10):
    """
    Analyse convergence sensitivity to initialization.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features).
    n_components : int
        Number of mixture components.
    n_runs : int
        Number of random restarts.

    Returns
    -------
    elbo_curves : list of list
        ELBO history for each run.
    final_elbos : list of float
        Final ELBO for each run.
    fig : matplotlib.figure.Figure
        Convergence curves plot.
    """
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data from 3 Gaussians
    n = 300
    X0 = np.random.randn(n // 3, 2) * 0.5 + np.array([0, 0])
    X1 = np.random.randn(n // 3, 2) * 0.5 + np.array([3, 3])
    X2 = np.random.randn(n // 3, 2) * 0.5 + np.array([3, 0])
    X = np.vstack([X0, X1, X2])

    print("=== Variational GMM ===")
    vi = VariationalGMM(n_components=3, random_state=42)
    vi.fit(X)
    labels_vi = vi.predict(X)
    print(f"VI cluster sizes: {[np.sum(labels_vi == i) for i in range(3)]}")
    print(f"Final ELBO: {vi.elbo_history[-1]:.2f}")
    print(f"Converged in {len(vi.elbo_history)} iterations")

    print("\n=== EM GMM ===")
    em = EMGMM(n_components=3, random_state=42)
    em.fit(X)
    labels_em = em.predict(X)
    print(f"EM cluster sizes: {[np.sum(labels_em == i) for i in range(3)]}")
    print(f"Final log-likelihood: {em.log_likelihood_history[-1]:.2f}")
    print(f"Converged in {len(em.log_likelihood_history)} iterations")

    print("\n=== Comparison ===")
    vi_model, em_model, fig_comp = compare_vi_and_em(X, n_components=3, random_state=42)

    print("\n=== Convergence Analysis ===")
    elbo_curves, final_elbos, fig_conv = convergence_analysis(X, n_components=3, n_runs=5)
    print(f"ELBO range: [{min(final_elbos):.2f}, {max(final_elbos):.2f}]")
    plt.show()
