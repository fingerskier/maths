"""
Problem 3: Second-Order ODE with Constant Coefficients

Solve:  y'' - 3y' + 2y = 0,  y(0) = 1, y'(0) = 0

Tasks:
  (a) Write the characteristic equation and find its roots
  (b) Write the general solution in terms of exponentials
  (c) Apply initial conditions to find constants
  (d) Convert to a first-order system and solve numerically
"""

import numpy as np
from scipy.integrate import solve_ivp


def characteristic_roots():
    """
    Return the roots of the characteristic equation r^2 - 3r + 2 = 0.
    """
    # TODO: solve r^2 - 3r + 2 = 0
    raise NotImplementedError


def analytic_solution(t):
    """Return y(t) for the IVP."""
    # TODO: y = C1*e^(r1*t) + C2*e^(r2*t), apply IC
    raise NotImplementedError


def to_first_order_system(t, state):
    """
    Convert y'' - 3y' + 2y = 0 to a first-order system:
      let u0 = y, u1 = y'
      u0' = u1
      u1' = 3*u1 - 2*u0
    """
    u0, u1 = state
    return [u1, 3 * u1 - 2 * u0]


def numerical_solution(t_end=3.0, n=300):
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(to_first_order_system, [0, t_end], [1.0, 0.0], t_eval=t_eval)
    return sol.t, sol.y[0]


if __name__ == "__main__":
    try:
        r1, r2 = characteristic_roots()
        print(f"Characteristic roots: {r1}, {r2}")
    except NotImplementedError:
        print("TODO: implement characteristic_roots()")

    t, y_num = numerical_solution()
    try:
        y_exact = analytic_solution(t)
        print(f"Max error: {np.max(np.abs(y_exact - y_num)):.2e}")
    except NotImplementedError:
        print("TODO: implement analytic_solution()")
    print(f"Numerical y(3) = {y_num[-1]:.6f}")
