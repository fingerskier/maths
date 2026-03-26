"""
Classification Methods
======================

Implement fundamental classification algorithms from scratch and evaluate
them using proper methodology.

Tasks
-----
1. K-Nearest Neighbors: Implement a KNN classifier from scratch. Support
   arbitrary k and use Euclidean distance. Include a predict method that returns
   class labels based on majority vote among the k nearest training points.

2. Logistic Regression via Gradient Descent: Implement binary logistic regression
   trained with gradient descent. Include the sigmoid function, the negative
   log-likelihood loss, and its gradient. Support a configurable learning rate
   and number of iterations.

3. Decision Boundary Visualization: For a 2D dataset, plot the decision boundary
   of both classifiers. Generate a mesh grid over the feature space, predict at
   each grid point, and use a contour plot.

4. Cross-Validation: Implement k-fold cross-validation from scratch. Split the
   data into k folds, train on k-1 folds, evaluate on the held-out fold, and
   report the mean and standard deviation of accuracy across folds.
"""

import numpy as np
import matplotlib.pyplot as plt


class KNNClassifier:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k=5):
        """
        Parameters
        ----------
        k : int
            Number of neighbors.
        """
        raise NotImplementedError

    def fit(self, X, y):
        """
        Store training data.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predict class labels for new data.

        Parameters
        ----------
        X : np.ndarray
            Test features of shape (n_samples, n_features).

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels.
        """
        raise NotImplementedError


class LogisticRegression:
    """Binary logistic regression trained with gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for gradient descent.
        n_iterations : int
            Number of gradient descent iterations.
        """
        raise NotImplementedError

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        raise NotImplementedError

    def fit(self, X, y):
        """
        Train the model using gradient descent on the negative log-likelihood.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Binary training labels (0 or 1) of shape (n_samples,).

        Returns
        -------
        loss_history : list of float
            Loss at each iteration (for monitoring convergence).
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).

        Returns
        -------
        proba : np.ndarray
            Predicted probability of class 1 for each sample.
        """
        raise NotImplementedError

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).
        threshold : float
            Decision threshold.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels (0 or 1).
        """
        raise NotImplementedError


def plot_decision_boundary(classifier, X, y, title="Decision Boundary"):
    """
    Plot the decision boundary of a classifier on 2D data.

    Parameters
    ----------
    classifier : object
        Fitted classifier with a predict method.
    X : np.ndarray
        Feature matrix of shape (n_samples, 2).
    y : np.ndarray
        Labels.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    raise NotImplementedError


def k_fold_cross_validation(classifier_class, X, y, k=5, **classifier_kwargs):
    """
    Perform k-fold cross-validation.

    Parameters
    ----------
    classifier_class : type
        Class of the classifier (e.g., KNNClassifier).
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Labels of shape (n_samples,).
    k : int
        Number of folds.
    **classifier_kwargs
        Keyword arguments passed to the classifier constructor.

    Returns
    -------
    mean_accuracy : float
        Mean accuracy across folds.
    std_accuracy : float
        Standard deviation of accuracy across folds.
    fold_accuracies : list of float
        Accuracy for each fold.
    """
    raise NotImplementedError


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic 2D dataset
    n = 200
    X0 = np.random.randn(n // 2, 2) + np.array([1, 1])
    X1 = np.random.randn(n // 2, 2) + np.array([-1, -1])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    print("=== KNN Classifier ===")
    knn = KNNClassifier(k=5)
    knn.fit(X, y)
    preds = knn.predict(X)
    print(f"Training accuracy: {np.mean(preds == y):.4f}")

    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(learning_rate=0.1, n_iterations=500)
    loss_hist = lr.fit(X, y)
    preds_lr = lr.predict(X)
    print(f"Training accuracy: {np.mean(preds_lr == y):.4f}")
    print(f"Final loss: {loss_hist[-1]:.4f}")

    print("\n=== Decision Boundaries ===")
    fig_knn = plot_decision_boundary(knn, X, y, title="KNN (k=5)")
    fig_lr = plot_decision_boundary(lr, X, y, title="Logistic Regression")

    print("\n=== Cross-Validation ===")
    mean_acc, std_acc, folds = k_fold_cross_validation(KNNClassifier, X, y, k=5, k_neighbors=5)
    print(f"KNN CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    plt.show()
