"""
Problem 2: 1D Wave Equation

Solve the wave equation:
    d^2u/dt^2 = c^2 * d^2u/dx^2,  x in [0, L], t > 0

Boundary conditions: u(0, t) = u(L, t) = 0  (fixed ends)
Initial conditions: u(x, 0) = sin(pi*x/L),  du/dt(x, 0) = 0

Parameters: c = 1.0, L = 1.0

Tasks:
  (a) Write the exact (d'Alembert / Fourier) solution
  (b) Implement the finite difference scheme (second-order in both x and t)
  (c) Verify the CFL condition: c*dt/dx <= 1
  (d) Animate or plot the solution at several time snapshots
"""

import numpy as np
import matplotlib.pyplot as plt

C = 1.0
L = 1.0


def exact_solution(x, t):
    """
    Exact solution: u(x,t) = sin(pi*x/L)*cos(c*pi*t/L)
    """
    # TODO
    raise NotImplementedError


def finite_difference_wave(nx=100, dt=0.005, t_end=2.0):
    """
    Solve using the standard explicit finite difference scheme:
    u^{n+1}_j = 2*u^n_j - u^{n-1}_j + r^2*(u^n_{j+1} - 2*u^n_j + u^n_{j-1})
    where r = c*dt/dx
    """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    r = C * dt / dx
    print(f"CFL number: {r:.4f} (must be <= 1)")

    nt = int(t_end / dt)

    # initial conditions
    u_prev = np.sin(np.pi * x / L)
    # For du/dt(x,0) = 0, use: u^1_j = u^0_j + 0.5*r^2*(u^0_{j+1} - 2*u^0_j + u^0_{j-1})
    u_curr = np.copy(u_prev)
    for j in range(1, nx - 1):
        u_curr[j] = u_prev[j] + 0.5 * r**2 * (
            u_prev[j + 1] - 2 * u_prev[j] + u_prev[j - 1]
        )
    u_curr[0] = u_curr[-1] = 0

    snapshots = [(0, u_prev.copy())]
    save_interval = max(1, nt // 8)

    for n in range(1, nt):
        u_next = np.zeros(nx)
        for j in range(1, nx - 1):
            u_next[j] = (2 * u_curr[j] - u_prev[j]
                         + r**2 * (u_curr[j + 1] - 2 * u_curr[j] + u_curr[j - 1]))
        u_next[0] = u_next[-1] = 0
        u_prev = u_curr
        u_curr = u_next
        if (n + 1) % save_interval == 0:
            snapshots.append(((n + 1) * dt, u_curr.copy()))

    return x, snapshots


def plot_snapshots():
    x, snapshots = finite_difference_wave()
    for t_val, u in snapshots:
        plt.plot(x, u, label=f"t={t_val:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("1D Wave Equation")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_snapshots()
