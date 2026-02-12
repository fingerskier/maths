"""
Problem 3: Conjugate Gradient Method

Solve the linear system Ax = b (equivalently, minimize f(x) = (1/2)x^T A x - b^T x)
using the conjugate gradient method.

    A = [[4, 1],    b = [1, 2]
         [1, 3]]

Also apply nonlinear CG (Fletcher-Reeves) to minimize:
    f(x, y) = 10*x^2 + y^2

Tasks:
  (a) Implement the linear CG algorithm for Ax = b
  (b) Verify it converges in at most n steps (where n = dimension)
  (c) Implement nonlinear CG (Fletcher-Reeves formula for beta)
  (d) Compare linear CG, nonlinear CG, and steepest descent
"""

import numpy as np


A = np.array([[4.0, 1.0],
              [1.0, 3.0]])
b = np.array([1.0, 2.0])


def quadratic_f(x):
    """f(x) = (1/2) x^T A x - b^T x."""
    return 0.5 * x @ A @ x - b @ x


def quadratic_grad(x):
    """grad f = A x - b."""
    return A @ x - b


def linear_cg(A, b, x0=None, tol=1e-12, max_iter=None):
    """
    Conjugate gradient method for Ax = b.

    Parameters:
        A: SPD matrix (n x n)
        b: RHS vector (n,)
        x0: initial guess
        tol: tolerance on ||r||
        max_iter: defaults to n

    Returns:
        (x_opt, residuals, path) where
        - residuals is a list of ||r_k|| at each iteration
        - path is a list of iterates
    """
    if x0 is None:
        x0 = np.zeros_like(b)
    if max_iter is None:
        max_iter = len(b)
    # TODO
    raise NotImplementedError


def elliptic_f(x):
    """An ill-conditioned quadratic: f(x,y) = 10*x^2 + y^2."""
    return 10 * x[0] ** 2 + x[1] ** 2


def elliptic_grad(x):
    """Gradient of the elliptic function."""
    return np.array([20 * x[0], 2 * x[1]])


def fletcher_reeves_cg(f_func, grad_func, x0, tol=1e-8, max_iter=1000):
    """
    Nonlinear CG with Fletcher-Reeves formula.

    beta_k = ||g_{k+1}||^2 / ||g_k||^2
    d_{k+1} = -g_{k+1} + beta_k * d_k

    Use a line search for the step size.
    Returns (x_opt, path, f_values).
    """
    # TODO
    raise NotImplementedError


def compare_methods(x0=np.array([5.0, 5.0])):
    """Compare steepest descent, linear CG, and nonlinear CG."""
    import matplotlib.pyplot as plt

    # Steepest descent on quadratic
    x = x0.copy()
    sd_vals = [quadratic_f(x)]
    for _ in range(50):
        g = quadratic_grad(x)
        alpha = g @ g / (g @ A @ g)  # exact step for quadratic
        x = x - alpha * g
        sd_vals.append(quadratic_f(x))

    plt.figure(figsize=(10, 6))
    plt.semilogy(sd_vals, "r-", label="Steepest Descent")

    try:
        _, residuals, cg_path = linear_cg(A, b, x0.copy())
        cg_vals = [quadratic_f(p) for p in cg_path]
        plt.semilogy(cg_vals, "b-o", label="Linear CG")
    except NotImplementedError:
        print("TODO: implement linear_cg()")

    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title("Steepest Descent vs Conjugate Gradient")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("True solution:", np.linalg.solve(A, b))
    try:
        x_opt, residuals, path = linear_cg(A, b)
        print(f"CG solution: {x_opt}")
        print(f"Converged in {len(path) - 1} iterations")
        print(f"Residual norms: {residuals}")
    except NotImplementedError:
        print("TODO: implement linear_cg()")

    try:
        x_opt, path, f_vals = fletcher_reeves_cg(elliptic_f, elliptic_grad,
                                                  np.array([5.0, 5.0]))
        print(f"\nFletcher-Reeves CG on elliptic: x = {x_opt}, f = {f_vals[-1]:.6f}")
    except NotImplementedError:
        print("TODO: implement fletcher_reeves_cg()")

    compare_methods()
