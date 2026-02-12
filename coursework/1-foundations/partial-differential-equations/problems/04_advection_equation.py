"""
Problem 4: 1D Advection Equation — Upwind Scheme

Solve the linear advection equation:
    du/dt + a * du/dx = 0,  x in [0, 2], t > 0

with periodic boundary conditions and initial condition:
    u(x, 0) = exp(-100*(x - 0.5)^2)  (a Gaussian pulse)

Parameters: a = 1.0 (wave speed)

Tasks:
  (a) Implement the first-order upwind scheme
  (b) Implement the Lax-Wendroff scheme (second-order)
  (c) Compare numerical diffusion and dispersion between the two schemes
  (d) Verify that the exact solution is u(x, t) = u(x - a*t, 0)
"""

import numpy as np
import matplotlib.pyplot as plt

A = 1.0  # wave speed
L = 2.0


def initial_condition(x):
    return np.exp(-100 * (x - 0.5) ** 2)


def exact_solution(x, t):
    """Exact solution: shift initial condition by a*t with periodic wrapping."""
    return initial_condition((x - A * t) % L)


def upwind_scheme(nx=200, cfl=0.8, t_end=1.0):
    """First-order upwind scheme for a > 0."""
    dx = L / nx
    dt = cfl * dx / abs(A)
    nt = int(t_end / dt)
    x = np.linspace(0, L, nx, endpoint=False)

    u = initial_condition(x)
    r = A * dt / dx

    for _ in range(nt):
        u_new = u.copy()
        for i in range(nx):
            u_new[i] = u[i] - r * (u[i] - u[(i - 1) % nx])
        u = u_new

    return x, u, nt * dt


def lax_wendroff(nx=200, cfl=0.8, t_end=1.0):
    """Second-order Lax-Wendroff scheme."""
    dx = L / nx
    dt = cfl * dx / abs(A)
    nt = int(t_end / dt)
    x = np.linspace(0, L, nx, endpoint=False)

    u = initial_condition(x)
    r = A * dt / dx

    for _ in range(nt):
        u_new = u.copy()
        for i in range(nx):
            u_new[i] = (u[i]
                        - 0.5 * r * (u[(i + 1) % nx] - u[(i - 1) % nx])
                        + 0.5 * r**2 * (u[(i + 1) % nx] - 2 * u[i] + u[(i - 1) % nx]))
        u = u_new

    return x, u, nt * dt


def compare_schemes():
    x_up, u_up, t_up = upwind_scheme()
    x_lw, u_lw, t_lw = lax_wendroff()
    u_exact = exact_solution(x_up, t_up)

    plt.figure(figsize=(10, 6))
    plt.plot(x_up, u_exact, "k-", label="Exact", linewidth=2)
    plt.plot(x_up, u_up, "b--", label="Upwind", linewidth=1.5)
    plt.plot(x_lw, u_lw, "r--", label="Lax-Wendroff", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Advection Equation at t = {t_up:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Upwind L2 error:       {np.linalg.norm(u_up - u_exact):.6f}")
    print(f"Lax-Wendroff L2 error: {np.linalg.norm(u_lw - u_exact):.6f}")


if __name__ == "__main__":
    compare_schemes()
