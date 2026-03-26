"""
Problem 4: Nonhomogeneous ODE — Method of Undetermined Coefficients

Solve:  y'' + y = cos(2x),  y(0) = 0, y'(0) = 1

Tasks:
  (a) Find the complementary (homogeneous) solution y_c
  (b) Guess a particular solution y_p and determine its coefficients
  (c) Combine y = y_c + y_p and apply initial conditions
  (d) Verify numerically
"""

import numpy as np
from scipy.integrate import solve_ivp


def complementary_solution(x):
    """Return y_c(x) — the general solution to y'' + y = 0 (with two free constants)."""
    # y_c = C1*cos(x) + C2*sin(x)
    # TODO: return with undetermined C1, C2 or implement after applying IC
    raise NotImplementedError


def particular_solution(x):
    """Return y_p(x) — a particular solution to y'' + y = cos(2x)."""
    # Guess y_p = A*cos(2x) + B*sin(2x), substitute to find A, B.
    # TODO
    raise NotImplementedError


def full_solution(x):
    """Return y(x) = y_c(x) + y_p(x) satisfying the initial conditions."""
    # TODO
    raise NotImplementedError


def numerical_solution(x_end=10.0, n=500):
    def ode(t, state):
        y, yp = state
        return [yp, np.cos(2 * t) - y]

    t_eval = np.linspace(0, x_end, n)
    sol = solve_ivp(ode, [0, x_end], [0.0, 1.0], t_eval=t_eval)
    return sol.t, sol.y[0]


if __name__ == "__main__":
    t, y_num = numerical_solution()
    try:
        y_exact = full_solution(t)
        print(f"Max error: {np.max(np.abs(y_exact - y_num)):.2e}")
    except NotImplementedError:
        print("TODO: implement full_solution()")
    print(f"Numerical y(pi) = {y_num[np.argmin(np.abs(t - np.pi))]:.6f}")
