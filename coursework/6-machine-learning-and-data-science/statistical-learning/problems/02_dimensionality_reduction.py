"""
Dimensionality Reduction
========================

Implement Principal Component Analysis (PCA) from scratch and explore its
properties for data compression and visualization.

Tasks
-----
1. PCA via Eigendecomposition: Implement PCA by computing the covariance matrix
   of the centered data, then finding its eigenvalues and eigenvectors. Return
   the principal components (directions), explained variance, and projections.

2. Scree Plot: Create a scree plot showing the proportion of variance explained
   by each principal component. Include a cumulative variance curve. This aids
   in deciding how many components to retain.

3. Data Reconstruction: Project data onto the top k principal components, then
   reconstruct back to the original space. Compute the reconstruction error as
   a function of k.

4. Comparison with SVD: Implement PCA via the SVD of the centered data matrix.
   Verify that the results match the eigendecomposition approach. Discuss when
   SVD is numerically preferable.
"""

import numpy as np
import matplotlib.pyplot as plt


def pca_eigendecomposition(X, n_components=None):
    """
    Perform PCA via eigendecomposition of the covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_components : int or None
        Number of components to retain (all if None).

    Returns
    -------
    components : np.ndarray
        Principal component directions of shape (n_components, n_features).
    explained_variance : np.ndarray
        Variance explained by each component.
    explained_variance_ratio : np.ndarray
        Proportion of total variance explained by each component.
    projected : np.ndarray
        Data projected onto principal components (n_samples, n_components).
    """
    raise NotImplementedError


def scree_plot(explained_variance_ratio):
    """
    Create a scree plot with cumulative variance explained.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Proportion of variance explained by each component.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    raise NotImplementedError


def reconstruct_from_components(projected, components, mean):
    """
    Reconstruct data from principal component projections.

    Parameters
    ----------
    projected : np.ndarray
        Projected data of shape (n_samples, k).
    components : np.ndarray
        Principal component directions of shape (k, n_features).
    mean : np.ndarray
        Original data mean of shape (n_features,).

    Returns
    -------
    reconstructed : np.ndarray
        Reconstructed data of shape (n_samples, n_features).
    """
    raise NotImplementedError


def reconstruction_error_vs_k(X):
    """
    Compute reconstruction error as a function of number of components k.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).

    Returns
    -------
    k_values : np.ndarray
        Array of k values from 1 to n_features.
    errors : np.ndarray
        Mean squared reconstruction error for each k.
    """
    raise NotImplementedError


def pca_svd(X, n_components=None):
    """
    Perform PCA via SVD of the centered data matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_components : int or None
        Number of components to retain (all if None).

    Returns
    -------
    components : np.ndarray
        Principal component directions of shape (n_components, n_features).
    explained_variance : np.ndarray
        Variance explained by each component.
    projected : np.ndarray
        Data projected onto principal components (n_samples, n_components).
    """
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(42)

    # Generate correlated 5D data
    n = 300
    true_dim = 2
    W = np.random.randn(true_dim, 5)
    X = np.random.randn(n, true_dim) @ W + 0.5 * np.random.randn(n, 5)

    print("=== PCA via Eigendecomposition ===")
    components, var, var_ratio, projected = pca_eigendecomposition(X)
    print(f"Explained variance ratios: {var_ratio}")
    print(f"Cumulative: {np.cumsum(var_ratio)}")

    print("\n=== Scree Plot ===")
    fig = scree_plot(var_ratio)

    print("\n=== Reconstruction ===")
    mean = X.mean(axis=0)
    X_recon = reconstruct_from_components(projected[:, :2], components[:2], mean)
    mse = np.mean((X - X_recon)**2)
    print(f"MSE with k=2 components: {mse:.6f}")

    print("\n=== Reconstruction Error vs k ===")
    k_vals, errors = reconstruction_error_vs_k(X)
    for k, err in zip(k_vals, errors):
        print(f"k={k}: MSE={err:.6f}")

    print("\n=== PCA via SVD ===")
    comp_svd, var_svd, proj_svd = pca_svd(X)
    # Check agreement with eigendecomposition
    diff = np.max(np.abs(np.abs(comp_svd) - np.abs(components)))
    print(f"Max component difference (up to sign): {diff:.2e}")
    plt.show()
