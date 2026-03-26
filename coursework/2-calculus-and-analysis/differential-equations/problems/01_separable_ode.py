"""
Problem 1: Separable ODEs

Solve the separable ODE:  dy/dx = x * y^2,  y(0) = 1

Tasks:
  (a) Separate variables and integrate both sides to find y(x) analytically
  (b) Implement the analytic solution
  (c) Solve numerically using scipy and compare with the analytic result
  (d) Plot both solutions on the same axes
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def analytic_solution(x):
    """
    Return y(x) for the IVP dy/dx = x*y^2, y(0) = 1.
    Hint: separation gives -1/y = x^2/2 + C.
    """
    # TODO
    raise NotImplementedError


def numerical_solution(x_span=(0, 1.3), n_points=200):
    """Solve the ODE numerically using solve_ivp."""
    def ode(t, y):
        return t * y[0] ** 2

    t_eval = np.linspace(*x_span, n_points)
    sol = solve_ivp(ode, x_span, [1.0], t_eval=t_eval, method="RK45")
    return sol.t, sol.y[0]


def plot_comparison():
    t_num, y_num = numerical_solution()
    try:
        y_exact = analytic_solution(t_num)
        plt.plot(t_num, y_exact, "b-", label="Analytic")
    except NotImplementedError:
        pass
    plt.plot(t_num, y_num, "r--", label="Numerical (RK45)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Separable ODE: dy/dx = x*y^2")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_comparison()
