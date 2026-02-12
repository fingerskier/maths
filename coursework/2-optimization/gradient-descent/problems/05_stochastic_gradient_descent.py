"""
Problem 5: Stochastic Gradient Descent (SGD)

Minimize a sum-of-functions objective (common in machine learning):

    F(w) = (1/N) * sum_{i=1}^{N} f_i(w)

where each f_i(w) = (w^T x_i - y_i)^2 is a squared loss for data point i.

This is linear regression: find w that minimizes mean squared error.

Tasks:
  (a) Implement batch gradient descent (uses all N data points per step)
  (b) Implement SGD (uses one random data point per step)
  (c) Implement mini-batch SGD (uses a subset of B data points per step)
  (d) Compare convergence behavior: batch vs SGD vs mini-batch
      (track objective value over effective passes through the data)
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=200, d=5, seed=42):
    """Generate a synthetic linear regression dataset."""
    rng = np.random.RandomState(seed)
    w_true = rng.randn(d)
    X = rng.randn(n, d)
    y = X @ w_true + 0.1 * rng.randn(n)
    return X, y, w_true


def mse_loss(w, X, y):
    """Mean squared error: (1/N) ||Xw - y||^2."""
    residuals = X @ w - y
    return np.mean(residuals ** 2)


def mse_gradient(w, X, y):
    """Gradient of MSE: (2/N) X^T (Xw - y)."""
    residuals = X @ w - y
    return (2 / len(y)) * X.T @ residuals


def batch_gd(X, y, w0=None, alpha=0.01, max_epochs=100):
    """
    Full batch gradient descent.
    Return (w_opt, loss_history) where loss_history has one entry per epoch.
    """
    if w0 is None:
        w0 = np.zeros(X.shape[1])
    # TODO
    raise NotImplementedError


def sgd(X, y, w0=None, alpha=0.01, max_epochs=100, seed=0):
    """
    Stochastic gradient descent (one sample at a time).

    At each step, pick a random index i and compute:
        w <- w - alpha * grad f_i(w)

    Return (w_opt, loss_history) where loss_history records the full
    loss after each epoch (full pass through the data).
    """
    if w0 is None:
        w0 = np.zeros(X.shape[1])
    # TODO
    raise NotImplementedError


def mini_batch_sgd(X, y, w0=None, alpha=0.01, batch_size=32,
                    max_epochs=100, seed=0):
    """
    Mini-batch SGD.
    Return (w_opt, loss_history) with one loss entry per epoch.
    """
    if w0 is None:
        w0 = np.zeros(X.shape[1])
    # TODO
    raise NotImplementedError


def compare_methods():
    """Compare batch GD, SGD, and mini-batch SGD."""
    X, y, w_true = generate_data()
    print(f"True weights: {w_true}")

    methods = [
        ("Batch GD", batch_gd),
        ("SGD", sgd),
        ("Mini-batch SGD (B=32)", lambda X, y, **kw: mini_batch_sgd(X, y, batch_size=32, **kw)),
    ]

    plt.figure(figsize=(10, 6))
    for name, method in methods:
        try:
            w_opt, losses = method(X, y)
            plt.semilogy(losses, label=f"{name}")
            print(f"{name}: final loss = {losses[-1]:.6f}, "
                  f"w_err = {np.linalg.norm(w_opt - w_true):.6f}")
        except NotImplementedError:
            print(f"TODO: implement {name}")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Batch GD vs SGD vs Mini-batch SGD")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_methods()
