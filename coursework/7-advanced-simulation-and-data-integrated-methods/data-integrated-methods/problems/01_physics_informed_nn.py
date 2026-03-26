"""
Problem 1: Physics-Informed Neural Network (PINN)

Solve the ODE  dy/dx = -y,  y(0) = 1  using a simple neural network
that incorporates the physics (differential equation) into its loss function.

The analytical solution is y(x) = exp(-x).

Tasks:
  (a) Implement a simple 2-layer neural network (input -> hidden -> output)
      with tanh activation, using only numpy
  (b) Define the total loss = data_loss + lambda * physics_loss, where:
      - data_loss penalizes deviation from known boundary condition y(0) = 1
      - physics_loss penalizes deviation from dy/dx + y = 0 at collocation points
  (c) Train using gradient descent and compare the learned solution
      with the analytical solution y = exp(-x)
  (d) Experiment with different values of lambda (physics loss weight)
      and observe the effect on solution quality
"""

import numpy as np


def tanh(x):
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh."""
    return 1.0 - np.tanh(x) ** 2


def init_network(n_input, n_hidden, n_output, seed=42):
    """
    Initialize a 2-layer network with random weights.

    Returns:
        dict with keys 'W1', 'b1', 'W2', 'b2'
    """
    # TODO
    raise NotImplementedError


def forward(params, x):
    """
    Forward pass through the network.

    Parameters:
        params: dict with W1, b1, W2, b2
        x: input array of shape (N, 1)

    Returns:
        y_pred: output of shape (N, 1)
        cache: dict with intermediate values for backprop
    """
    # TODO
    raise NotImplementedError


def compute_dydx(params, x, cache):
    """
    Compute dy/dx using the chain rule through the network.

    For y = W2 @ tanh(W1 @ x + b1) + b2:
        dy/dx = W2 @ diag(tanh'(W1 @ x + b1)) @ W1

    Parameters:
        params: network parameters
        x: input points (N, 1)
        cache: from forward pass

    Returns:
        dydx: shape (N, 1)
    """
    # TODO
    raise NotImplementedError


def physics_loss(params, x_collocation):
    """
    Compute physics loss: mean of (dy/dx + y)^2 at collocation points.

    The ODE is dy/dx = -y, so the residual is dy/dx + y.
    """
    # TODO
    raise NotImplementedError


def data_loss(params, x_data, y_data):
    """
    Compute data loss: mean squared error at known data points.
    """
    # TODO
    raise NotImplementedError


def total_loss(params, x_collocation, x_data, y_data, lam=1.0):
    """
    Total loss = data_loss + lam * physics_loss.
    """
    # TODO
    raise NotImplementedError


def train_pinn(n_hidden=20, n_collocation=50, lam=1.0, lr=0.01,
               n_epochs=5000, seed=42):
    """
    Train the PINN to solve dy/dx = -y, y(0) = 1.

    Parameters:
        n_hidden: number of hidden units
        n_collocation: number of interior collocation points
        lam: weight of physics loss
        lr: learning rate
        n_epochs: number of training epochs
        seed: random seed

    Returns:
        params: trained network parameters
        losses: list of total loss per epoch
    """
    # TODO
    raise NotImplementedError


def compare_with_analytical(params, x_test=None):
    """
    Compare PINN prediction with y = exp(-x) on test points.

    Returns:
        x_test, y_pred, y_exact, max_error
    """
    if x_test is None:
        x_test = np.linspace(0, 2, 100).reshape(-1, 1)
    y_exact = np.exp(-x_test)
    y_pred, _ = forward(params, x_test)
    max_error = np.max(np.abs(y_pred - y_exact))
    return x_test, y_pred, y_exact, max_error


if __name__ == "__main__":
    try:
        params, losses = train_pinn(lam=1.0)
        x, y_pred, y_exact, err = compare_with_analytical(params)
        print(f"Max error vs analytical solution: {err:.6f}")
        print(f"Final loss: {losses[-1]:.6f}")
    except NotImplementedError:
        print("TODO: implement PINN functions")
