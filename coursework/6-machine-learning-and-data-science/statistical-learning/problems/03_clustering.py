"""
Clustering Methods
==================

Implement unsupervised clustering algorithms from scratch and evaluate
cluster quality using internal metrics.

Tasks
-----
1. K-Means Clustering: Implement Lloyd's algorithm for k-means clustering.
   Initialize centroids randomly from data points (k-means++ optional bonus).
   Iterate assignment and update steps until convergence.

2. Hierarchical Clustering: Implement agglomerative hierarchical clustering with
   both single linkage (minimum distance between clusters) and complete linkage
   (maximum distance). Return the merge history (dendrogram data).

3. Silhouette Score: Implement the silhouette score for evaluating cluster
   quality. For each point, compute a(i) (mean intra-cluster distance) and
   b(i) (mean nearest-cluster distance). The score is (b-a)/max(a,b).

4. Elbow Method: Run k-means for k = 1, 2, ..., k_max and compute the total
   within-cluster sum of squares (inertia) for each k. Plot the elbow curve
   to help identify the optimal number of clusters.
"""

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """K-Means clustering."""

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6, random_state=None):
        """
        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance on centroid movement.
        random_state : int or None
            Random seed for reproducibility.
        """
        raise NotImplementedError

    def fit(self, X):
        """
        Fit k-means to the data.

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
        Assign clusters to new data points.

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
    def inertia(self):
        """Total within-cluster sum of squares."""
        raise NotImplementedError


def hierarchical_clustering(X, linkage="single"):
    """
    Agglomerative hierarchical clustering.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features).
    linkage : str
        Linkage criterion: 'single' or 'complete'.

    Returns
    -------
    merge_history : list of tuples
        List of (cluster_i, cluster_j, distance) at each merge step.
    labels_at_k : callable
        Function that takes k (number of clusters) and returns labels.
    """
    raise NotImplementedError


def silhouette_score(X, labels):
    """
    Compute the mean silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).

    Returns
    -------
    score : float
        Mean silhouette score in [-1, 1].
    sample_scores : np.ndarray
        Per-sample silhouette scores.
    """
    raise NotImplementedError


def elbow_method(X, k_max=10, random_state=None):
    """
    Run the elbow method for choosing the number of clusters.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features).
    k_max : int
        Maximum number of clusters to try.
    random_state : int or None
        Random seed.

    Returns
    -------
    k_values : list of int
        Values of k from 1 to k_max.
    inertias : list of float
        Inertia (within-cluster sum of squares) for each k.
    fig : matplotlib.figure.Figure
        Elbow plot.
    """
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(42)

    # Generate 3 clusters
    X0 = np.random.randn(100, 2) + np.array([0, 0])
    X1 = np.random.randn(100, 2) + np.array([5, 5])
    X2 = np.random.randn(100, 2) + np.array([5, 0])
    X = np.vstack([X0, X1, X2])
    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    print("=== K-Means ===")
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)
    labels = km.predict(X)
    print(f"Inertia: {km.inertia:.2f}")
    print(f"Cluster sizes: {[np.sum(labels == i) for i in range(3)]}")

    print("\n=== Hierarchical Clustering ===")
    merge_hist, get_labels = hierarchical_clustering(X, linkage="single")
    h_labels = get_labels(3)
    print(f"Cluster sizes: {[np.sum(h_labels == i) for i in range(3)]}")

    print("\n=== Silhouette Score ===")
    score, sample_scores = silhouette_score(X, labels)
    print(f"Mean silhouette score: {score:.4f}")

    print("\n=== Elbow Method ===")
    k_vals, inertias, fig = elbow_method(X, k_max=8, random_state=42)
    for k, inertia in zip(k_vals, inertias):
        print(f"k={k}: inertia={inertia:.2f}")
    plt.show()
