"""
Problem 3: 2D Laplace Equation (Steady-State Heat)

Solve Laplace's equation on a unit square:
    d^2u/dx^2 + d^2u/dy^2 = 0,  (x, y) in [0,1] x [0,1]

Boundary conditions:
    u(x, 0) = sin(pi * x)   (bottom)
    u(x, 1) = 0              (top)
    u(0, y) = 0              (left)
    u(1, y) = 0              (right)

Tasks:
  (a) Derive the exact (separation of variables) solution
  (b) Implement the iterative Jacobi method
  (c) Implement the Gauss-Seidel method and compare convergence speed
  (d) Plot the solution as a contour plot or surface plot
"""

import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x, y):
    """
    Exact solution via separation of variables.
    Hint: u(x,y) = sin(pi*x) * sinh(pi*(1-y)) / sinh(pi)
    """
    # TODO
    raise NotImplementedError


def jacobi_iteration(nx=50, ny=50, tol=1e-6, max_iter=50000):
    """
    Solve using Jacobi iteration.
    Returns: x, y meshgrid arrays and solution u.
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    u = np.zeros((ny, nx))
    # bottom BC
    u[0, :] = np.sin(np.pi * x)

    for iteration in range(max_iter):
        u_old = u.copy()
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u[j, i] = 0.25 * (
                    u_old[j, i + 1] + u_old[j, i - 1]
                    + u_old[j + 1, i] + u_old[j - 1, i]
                )
        # check convergence
        if np.max(np.abs(u - u_old)) < tol:
            print(f"Jacobi converged in {iteration + 1} iterations")
            break
    else:
        print(f"Jacobi did not converge in {max_iter} iterations")

    X, Y = np.meshgrid(x, y)
    return X, Y, u


def gauss_seidel(nx=50, ny=50, tol=1e-6, max_iter=50000):
    """
    Solve using Gauss-Seidel iteration (uses updated values immediately).
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    u = np.zeros((ny, nx))
    u[0, :] = np.sin(np.pi * x)

    for iteration in range(max_iter):
        max_diff = 0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                old = u[j, i]
                u[j, i] = 0.25 * (
                    u[j, i + 1] + u[j, i - 1] + u[j + 1, i] + u[j - 1, i]
                )
                max_diff = max(max_diff, abs(u[j, i] - old))
        if max_diff < tol:
            print(f"Gauss-Seidel converged in {iteration + 1} iterations")
            break
    else:
        print(f"Gauss-Seidel did not converge in {max_iter} iterations")

    X, Y = np.meshgrid(x, y)
    return X, Y, u


def plot_solution():
    X, Y, u = gauss_seidel()
    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(X, Y, u, levels=30, cmap="hot")
    fig.colorbar(cp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Laplace Equation: Gauss-Seidel Solution")
    plt.show()


if __name__ == "__main__":
    plot_solution()
