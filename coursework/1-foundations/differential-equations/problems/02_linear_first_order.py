"""
Problem 2: First-Order Linear ODE with Integrating Factor

Solve:  dy/dx + 2*y = e^(-x),  y(0) = 3

Tasks:
  (a) Find the integrating factor mu(x)
  (b) Multiply through and integrate to find the general solution
  (c) Apply the initial condition to find the particular solution
  (d) Verify by substituting back into the ODE
  (e) Compare with a numerical solution
"""

import numpy as np
from scipy.integrate import solve_ivp


def integrating_factor(x):
    """Return mu(x) for the ODE y' + 2y = e^(-x)."""
    # TODO: mu(x) = exp(integral of 2 dx)
    raise NotImplementedError


def analytic_solution(x):
    """Return the particular solution y(x) with y(0) = 3."""
    # TODO: solve using integrating factor method
    raise NotImplementedError


def verify_solution(x):
    """Check that y' + 2*y == e^(-x) for the analytic solution."""
    h = 1e-8
    y = analytic_solution(x)
    y_prime = (analytic_solution(x + h) - analytic_solution(x - h)) / (2 * h)
    lhs = y_prime + 2 * y
    rhs = np.exp(-x)
    return np.abs(lhs - rhs)


def numerical_solution(x_end=3.0, n=300):
    """Solve numerically for comparison."""
    def ode(t, y):
        return np.exp(-t) - 2 * y[0]

    t_eval = np.linspace(0, x_end, n)
    sol = solve_ivp(ode, [0, x_end], [3.0], t_eval=t_eval)
    return sol.t, sol.y[0]


if __name__ == "__main__":
    t, y_num = numerical_solution()
    try:
        y_exact = analytic_solution(t)
        max_err = np.max(np.abs(y_exact - y_num))
        print(f"Max error between analytic and numerical: {max_err:.2e}")
        print(f"Residual at x=1: {verify_solution(1.0):.2e}")
    except NotImplementedError:
        print("TODO: implement analytic_solution()")
