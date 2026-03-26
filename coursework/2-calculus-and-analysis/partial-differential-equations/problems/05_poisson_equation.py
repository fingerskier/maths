"""
Problem 5: 2D Poisson Equation with Direct Solver

Solve Poisson's equation on the unit square:
    d^2u/dx^2 + d^2u/dy^2 = f(x, y)

where f(x, y) = -2*pi^2 * sin(pi*x) * sin(pi*y)

Boundary conditions: u = 0 on all boundaries.

The exact solution is: u(x, y) = sin(pi*x) * sin(pi*y)

Tasks:
  (a) Discretize using the standard 5-point stencil
  (b) Assemble the linear system Au = b
  (c) Solve using scipy.sparse.linalg
  (d) Compute and plot the error vs. grid refinement (convergence study)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_term(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def solve_poisson(n):
    """
    Solve the 2D Poisson equation on an n x n interior grid.
    Returns: X, Y meshgrid and solution u on the full (n+2) x (n+2) grid.
    """
    h = 1.0 / (n + 1)
    N = n * n  # number of interior unknowns

    # build the sparse matrix for the 5-point Laplacian
    main_diag = -4 * np.ones(N)
    off_diag_1 = np.ones(N - 1)
    # zero out connections that cross row boundaries
    for i in range(1, n):
        off_diag_1[i * n - 1] = 0
    off_diag_n = np.ones(N - n)

    A = (sparse.diags([main_diag, off_diag_1, off_diag_1, off_diag_n, off_diag_n],
                       [0, 1, -1, n, -n], format="csr") / h**2)

    # build the RHS
    x_int = np.linspace(h, 1 - h, n)
    y_int = np.linspace(h, 1 - h, n)
    X_int, Y_int = np.meshgrid(x_int, y_int)
    b = source_term(X_int, Y_int).ravel()

    # solve
    u_int = spsolve(A, b)
    u_int = u_int.reshape((n, n))

    # embed in full grid with BCs
    u_full = np.zeros((n + 2, n + 2))
    u_full[1:-1, 1:-1] = u_int

    x_full = np.linspace(0, 1, n + 2)
    y_full = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(x_full, y_full)

    return X, Y, u_full


def convergence_study():
    """Measure L-infinity error for several grid sizes."""
    ns = [10, 20, 40, 80]
    errors = []
    hs = []

    for n in ns:
        X, Y, u_num = solve_poisson(n)
        u_ex = exact_solution(X, Y)
        err = np.max(np.abs(u_num - u_ex))
        errors.append(err)
        hs.append(1.0 / (n + 1))
        print(f"n={n:3d}, h={hs[-1]:.4f}, max error={err:.2e}")

    # plot convergence
    plt.figure()
    plt.loglog(hs, errors, "bo-", label="Numerical error")
    plt.loglog(hs, [h**2 for h in hs], "r--", label="O(h^2) reference")
    plt.xlabel("h")
    plt.ylabel("Max error")
    plt.title("Poisson Equation Convergence Study")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    convergence_study()

    X, Y, u = solve_poisson(40)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    c1 = axes[0].contourf(X, Y, u, levels=30, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title("Numerical Solution")

    c2 = axes[1].contourf(X, Y, exact_solution(X, Y), levels=30, cmap="viridis")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title("Exact Solution")
    plt.tight_layout()
    plt.show()
