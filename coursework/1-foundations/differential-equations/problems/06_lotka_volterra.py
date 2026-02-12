"""
Problem 6: Nonlinear Systems — Lotka-Volterra Predator-Prey Model

The Lotka-Volterra equations model predator-prey dynamics:
    dx/dt = alpha*x - beta*x*y    (prey)
    dy/dt = delta*x*y - gamma*y   (predator)

With parameters: alpha=1.1, beta=0.4, delta=0.1, gamma=0.4

Tasks:
  (a) Find all equilibrium points (set dx/dt = dy/dt = 0)
  (b) Linearize the system at each equilibrium and classify stability
  (c) Numerically integrate from (x0, y0) = (10, 5) for t in [0, 100]
  (d) Plot x(t) and y(t) vs. time, and the phase portrait (x vs. y)
  (e) Verify that the Lotka-Volterra conserved quantity is constant along trajectories
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

ALPHA, BETA, DELTA, GAMMA = 1.1, 0.4, 0.1, 0.4


def equilibrium_points():
    """Return list of equilibrium points [(x1, y1), (x2, y2), ...]."""
    # TODO: solve alpha*x - beta*x*y = 0 and delta*x*y - gamma*y = 0
    raise NotImplementedError


def jacobian(x, y):
    """Return the Jacobian matrix of the system evaluated at (x, y)."""
    # TODO
    raise NotImplementedError


def conserved_quantity(x, y):
    """
    The Lotka-Volterra conserved quantity (first integral):
    H(x, y) = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)
    """
    return DELTA * x - GAMMA * np.log(x) + BETA * y - ALPHA * np.log(y)


def lotka_volterra(t, state):
    x, y = state
    dxdt = ALPHA * x - BETA * x * y
    dydt = DELTA * x * y - GAMMA * y
    return [dxdt, dydt]


def simulate(x0=10, y0=5, t_end=100, n=5000):
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(lotka_volterra, [0, t_end], [x0, y0],
                    t_eval=t_eval, method="RK45", rtol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


def plot_results():
    t, x, y = simulate()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(t, x, "b-", label="Prey (x)")
    axes[0].plot(t, y, "r-", label="Predator (y)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Population")
    axes[0].legend()
    axes[0].set_title("Population vs Time")

    axes[1].plot(x, y, "g-")
    axes[1].set_xlabel("Prey (x)")
    axes[1].set_ylabel("Predator (y)")
    axes[1].set_title("Phase Portrait")

    H = conserved_quantity(x, y)
    axes[2].plot(t, H)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("H(x, y)")
    axes[2].set_title("Conserved Quantity")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        print("Equilibria:", equilibrium_points())
    except NotImplementedError:
        print("TODO: implement equilibrium_points()")
    plot_results()
