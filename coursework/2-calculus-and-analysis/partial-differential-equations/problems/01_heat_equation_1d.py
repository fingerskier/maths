"""
Problem 1: 1D Heat Equation — Finite Difference Method

Solve the 1D heat equation:
    du/dt = k * d^2u/dx^2,  x in [0, L], t > 0

Boundary conditions: u(0, t) = 0, u(L, t) = 0
Initial condition: u(x, 0) = sin(pi * x / L)

Parameters: L = 1, k = 0.01

Tasks:
  (a) Derive the exact (Fourier series) solution for this specific IC/BC
  (b) Implement the explicit Forward-Time Central-Space (FTCS) scheme
  (c) Investigate the stability condition (CFL): k*dt/dx^2 < 0.5
  (d) Compare numerical solution with the exact solution at several times
"""

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
K = 0.01


def exact_solution(x, t):
    """
    Exact solution for the heat equation with u(x,0) = sin(pi*x/L).
    Hint: u(x,t) = sin(pi*x/L) * exp(-k*(pi/L)^2 * t)
    """
    # TODO
    raise NotImplementedError


def ftcs_heat_equation(nx=50, dt=0.0005, t_end=1.0):
    """
    Solve using Forward-Time Central-Space (explicit) finite difference.

    Returns: x array, t array, u[time_step, space_index] 2D array
    """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    nt = int(t_end / dt)

    # stability check
    r = K * dt / dx**2
    print(f"CFL number r = {r:.4f} (must be < 0.5 for stability)")

    u = np.sin(np.pi * x / L)  # initial condition
    u[0] = u[-1] = 0  # boundary conditions

    history = [u.copy()]
    for n in range(nt):
        u_new = u.copy()
        for i in range(1, nx - 1):
            u_new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
        u_new[0] = 0
        u_new[-1] = 0
        u = u_new
        if (n + 1) % (nt // 5) == 0:
            history.append(u.copy())

    return x, history


def plot_solution():
    x, history = ftcs_heat_equation()
    for i, u in enumerate(history):
        plt.plot(x, u, label=f"snapshot {i}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("1D Heat Equation (FTCS)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_solution()
