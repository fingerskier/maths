"""
Problem 5: Phase Portraits and Stability

Analyze the 2D linear system:
    dx/dt = -x + 2y
    dy/dt =  x - 2y

Tasks:
  (a) Write the system in matrix form: d/dt [x, y]^T = A [x, y]^T
  (b) Find the eigenvalues and eigenvectors of A
  (c) Classify the equilibrium at the origin (stable/unstable node, saddle, spiral, etc.)
  (d) Plot the phase portrait with several trajectories
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


A = np.array([[-1, 2],
              [1, -2]])


def eigenanalysis():
    """Return eigenvalues and eigenvectors of A."""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def classify_equilibrium(eigenvalues):
    """
    Return a string classifying the equilibrium based on eigenvalues.
    Examples: "stable node", "unstable node", "saddle", "stable spiral", etc.
    """
    # TODO: implement classification logic
    raise NotImplementedError


def system(t, state):
    return A @ state


def plot_phase_portrait():
    """Plot the phase portrait with multiple initial conditions."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # plot several trajectories
    initial_conditions = [
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (2, 2), (-2, -2), (1, -1), (-1, 1),
    ]
    for x0, y0 in initial_conditions:
        sol = solve_ivp(system, [0, 10], [x0, y0],
                        t_eval=np.linspace(0, 10, 500))
        ax.plot(sol.y[0], sol.y[1], "b-", alpha=0.6)
        ax.annotate("", xy=(sol.y[0][1], sol.y[1][1]),
                     xytext=(sol.y[0][0], sol.y[1][0]),
                     arrowprops=dict(arrowstyle="->", color="blue"))

    # vector field
    xs = np.linspace(-3, 3, 15)
    ys = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(xs, ys)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    ax.quiver(X, Y, U, V, alpha=0.3)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phase Portrait: dx/dt = -x+2y, dy/dt = x-2y")
    ax.set_aspect("equal")
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    evals, evecs = eigenanalysis()
    print("Eigenvalues:", evals)
    print("Eigenvectors:\n", evecs)
    try:
        print("Classification:", classify_equilibrium(evals))
    except NotImplementedError:
        print("TODO: implement classify_equilibrium()")
    plot_phase_portrait()
