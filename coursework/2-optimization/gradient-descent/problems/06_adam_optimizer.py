"""
Problem 6: Adam and Adaptive Learning Rate Methods

Implement adaptive learning rate optimizers and compare them on
the Rosenbrock function:

    f(x, y) = (1 - x)^2 + 100*(y - x^2)^2

Methods:
  - AdaGrad: accumulates squared gradients, shrinks step for frequent features
  - RMSProp: exponential moving average of squared gradients
  - Adam: combines momentum with RMSProp + bias correction

Tasks:
  (a) Implement AdaGrad
  (b) Implement RMSProp
  (c) Implement Adam (with bias correction)
  (d) Compare all methods + vanilla SGD on the Rosenbrock function
"""

import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dy = 200 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


def adagrad(grad_func, x0, alpha=0.5, eps=1e-8, max_iter=5000):
    """
    AdaGrad optimizer.

    G_t = G_{t-1} + g_t^2  (element-wise)
    x_{t+1} = x_t - (alpha / sqrt(G_t + eps)) * g_t

    Return (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def rmsprop(grad_func, x0, alpha=0.01, beta=0.9, eps=1e-8, max_iter=5000):
    """
    RMSProp optimizer.

    v_t = beta * v_{t-1} + (1 - beta) * g_t^2
    x_{t+1} = x_t - (alpha / sqrt(v_t + eps)) * g_t

    Return (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def adam(grad_func, x0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
         max_iter=5000):
    """
    Adam optimizer.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (first moment)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (second moment)
    m_hat = m_t / (1 - beta1^t)                         (bias correction)
    v_hat = v_t / (1 - beta2^t)
    x_{t+1} = x_t - alpha * m_hat / (sqrt(v_hat) + eps)

    Return (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def compare_optimizers(x0=np.array([-1.0, 1.0])):
    """Compare all optimizers on the Rosenbrock function."""
    optimizers = [
        ("AdaGrad", lambda x: adagrad(grad_rosenbrock, x)),
        ("RMSProp", lambda x: rmsprop(grad_rosenbrock, x)),
        ("Adam", lambda x: adam(grad_rosenbrock, x)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Contour
    xr = np.linspace(-2, 2, 200)
    yr = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(xr, yr)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 30), cmap="viridis")

    for name, opt in optimizers:
        try:
            x_opt, path, f_vals = opt(x0.copy())
            path = np.array(path)
            ax1.plot(path[:, 0], path[:, 1], ".-", markersize=1, label=name)
            ax2.semilogy(f_vals, label=f"{name} ({len(f_vals)} iters)")
        except NotImplementedError:
            print(f"TODO: implement {name}")

    ax1.plot(1, 1, "r*", markersize=15)
    ax1.set_title("Optimization Paths")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)")
    ax2.set_title("Convergence Comparison")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_optimizers()
