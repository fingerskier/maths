"""
Perceptron and Multi-Layer Perceptron
=====================================

Implement neural network building blocks from scratch, from a single perceptron
to a multi-layer network with backpropagation.

Tasks
-----
1. Single Perceptron: Implement a single-layer perceptron for binary classification.
   Use the step activation function and the perceptron learning rule. Train on
   linearly separable data and report convergence.

2. Two-Layer MLP with Backpropagation: Implement a 2-layer neural network
   (one hidden layer) with sigmoid activations. Derive and implement the
   backpropagation algorithm for computing gradients. Train using gradient descent.

3. XOR Problem: Train the MLP on the XOR dataset, which is not linearly separable.
   Demonstrate that a single perceptron fails but the MLP succeeds. Report the
   learned weights and final loss.

4. Decision Boundary Visualization: For the XOR problem and a 2D classification
   problem, visualize the decision boundary learned by the MLP. Show how the
   hidden layer creates a nonlinear boundary.
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """Single-layer perceptron for binary classification."""

    def __init__(self, learning_rate=0.1, max_epochs=1000):
        """
        Parameters
        ----------
        learning_rate : float
            Learning rate for weight updates.
        max_epochs : int
            Maximum number of passes over the training data.
        """
        raise NotImplementedError

    def fit(self, X, y):
        """
        Train the perceptron using the perceptron learning rule.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Binary labels (0 or 1) of shape (n_samples,).

        Returns
        -------
        n_epochs : int
            Number of epochs until convergence (or max_epochs).
        errors_per_epoch : list of int
            Number of misclassifications in each epoch.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).

        Returns
        -------
        predictions : np.ndarray
            Predicted labels (0 or 1).
        """
        raise NotImplementedError


class MLP:
    """Two-layer neural network (one hidden layer) with backpropagation."""

    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1):
        """
        Parameters
        ----------
        n_input : int
            Number of input features.
        n_hidden : int
            Number of hidden neurons.
        n_output : int
            Number of output neurons.
        learning_rate : float
            Learning rate for gradient descent.
        """
        raise NotImplementedError

    def sigmoid(self, z):
        """Sigmoid activation function."""
        raise NotImplementedError

    def sigmoid_derivative(self, a):
        """Derivative of sigmoid given the activation output a = sigmoid(z)."""
        raise NotImplementedError

    def forward(self, X):
        """
        Forward pass through the network.

        Parameters
        ----------
        X : np.ndarray
            Input of shape (n_samples, n_input).

        Returns
        -------
        output : np.ndarray
            Network output of shape (n_samples, n_output).
        """
        raise NotImplementedError

    def backward(self, X, y, output):
        """
        Backward pass: compute gradients and update weights.

        Parameters
        ----------
        X : np.ndarray
            Input of shape (n_samples, n_input).
        y : np.ndarray
            Target of shape (n_samples, n_output).
        output : np.ndarray
            Network output from forward pass.
        """
        raise NotImplementedError

    def fit(self, X, y, n_epochs=10000):
        """
        Train the network.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_input).
        y : np.ndarray
            Targets of shape (n_samples, n_output).
        n_epochs : int
            Number of training epochs.

        Returns
        -------
        loss_history : list of float
            Mean squared error at each epoch.
        """
        raise NotImplementedError

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (for binary classification).

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_input).
        threshold : float
            Decision threshold.

        Returns
        -------
        predictions : np.ndarray
        """
        raise NotImplementedError


def xor_experiment():
    """
    Demonstrate that a perceptron fails on XOR but an MLP succeeds.

    Returns
    -------
    perceptron_accuracy : float
        Perceptron accuracy on XOR.
    mlp_accuracy : float
        MLP accuracy on XOR.
    mlp_loss_history : list of float
        Training loss curve for the MLP.
    """
    raise NotImplementedError


def plot_mlp_decision_boundary(mlp, X, y, title="MLP Decision Boundary"):
    """
    Visualize the decision boundary of a trained MLP on 2D data.

    Parameters
    ----------
    mlp : MLP
        Trained MLP instance.
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


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Single Perceptron (AND gate) ===")
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    p = Perceptron(learning_rate=0.1)
    n_epochs, errors = p.fit(X_and, y_and)
    print(f"Converged in {n_epochs} epochs")
    print(f"Predictions: {p.predict(X_and)}")

    print("\n=== XOR Experiment ===")
    p_acc, m_acc, loss_hist = xor_experiment()
    print(f"Perceptron accuracy on XOR: {p_acc:.2f}")
    print(f"MLP accuracy on XOR: {m_acc:.2f}")
    print(f"MLP final loss: {loss_hist[-1]:.6f}")

    print("\n=== MLP Decision Boundary ===")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    mlp = MLP(n_input=2, n_hidden=4, n_output=1, learning_rate=1.0)
    mlp.fit(X_xor, y_xor, n_epochs=10000)
    fig = plot_mlp_decision_boundary(mlp, X_xor, y_xor.ravel())
    plt.show()
