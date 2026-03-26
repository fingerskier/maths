"""
ODE Numerical Methods
=====================

Implement numerical methods for solving ordinary differential equations (ODEs)
of the form y'(t) = f(t, y), y(t0) = y0.

Tasks
-----
1. Euler's Method: Implement the forward Euler method with fixed step size h.
   Solve y' = -2y, y(0) = 1 and compare with the exact solution y = e^{-2t}.

2. Improved Euler (Heun's Method): Implement the second-order predictor-corrector
   method. Compare accuracy with basic Euler on the same problem.

3. Classical Runge-Kutta (RK4): Implement the standard 4th-order Runge-Kutta
   method. Demonstrate the superior accuracy compared to lower-order methods.

4. Adaptive Step Size Control: Implement an adaptive RK4-5 method (or embedded
   pair) that adjusts the step size to maintain a local error below a given
   tolerance. Return the solution along with the step sizes used.

5. Stiff Equation Demonstration: Solve the stiff ODE y' = -1000(y - cos(t)) -
   sin(t), y(0) = 1 using Euler and RK4. Show that explicit methods require
   very small step sizes, and compare the solutions and step counts.
"""

import numpy as np
import matplotlib.pyplot as plt


def euler_method(f, t_span, y0, h):
    """
    Solve an ODE using the forward Euler method.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    t_span : tuple
        (t_start, t_end).
    y0 : float or np.ndarray
        Initial condition.
    h : float
        Step size.

    Returns
    -------
    t : np.ndarray
        Time points.
    y : np.ndarray
        Solution values at each time point.
    """
    raise NotImplementedError


def heuns_method(f, t_span, y0, h):
    """
    Solve an ODE using Heun's method (improved Euler / RK2).

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    t_span : tuple
        (t_start, t_end).
    y0 : float or np.ndarray
        Initial condition.
    h : float
        Step size.

    Returns
    -------
    t : np.ndarray
        Time points.
    y : np.ndarray
        Solution values at each time point.
    """
    raise NotImplementedError


def rk4_method(f, t_span, y0, h):
    """
    Solve an ODE using the classical 4th-order Runge-Kutta method.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    t_span : tuple
        (t_start, t_end).
    y0 : float or np.ndarray
        Initial condition.
    h : float
        Step size.

    Returns
    -------
    t : np.ndarray
        Time points.
    y : np.ndarray
        Solution values at each time point.
    """
    raise NotImplementedError


def adaptive_rk45(f, t_span, y0, tol=1e-6, h_init=0.1, h_min=1e-10, h_max=1.0):
    """
    Solve an ODE with adaptive step size control using an RK4-5 embedded pair.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    t_span : tuple
        (t_start, t_end).
    y0 : float or np.ndarray
        Initial condition.
    tol : float
        Local error tolerance.
    h_init : float
        Initial step size.
    h_min, h_max : float
        Minimum and maximum allowed step sizes.

    Returns
    -------
    t : np.ndarray
        Adaptive time points.
    y : np.ndarray
        Solution values at each time point.
    h_used : np.ndarray
        Step sizes used at each step.
    """
    raise NotImplementedError


def stiff_equation_demo():
    """
    Demonstrate the difficulty of stiff equations for explicit methods.

    Solve y' = -1000(y - cos(t)) - sin(t), y(0) = 1 on [0, 1].
    Compare Euler and RK4 with various step sizes and plot results.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure showing solutions and step size effects.
    results : dict
        Dictionary with method names mapping to (t, y, n_steps) tuples.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Test ODE: y' = -2y, y(0) = 1, exact: y = exp(-2t)
    f = lambda t, y: -2 * y
    exact = lambda t: np.exp(-2 * t)
    t_span = (0.0, 5.0)
    y0 = 1.0

    print("=== Euler's Method ===")
    t_e, y_e = euler_method(f, t_span, y0, h=0.1)
    print(f"Final error: {abs(y_e[-1] - exact(t_e[-1])):.6e}")

    print("\n=== Heun's Method ===")
    t_h, y_h = heuns_method(f, t_span, y0, h=0.1)
    print(f"Final error: {abs(y_h[-1] - exact(t_h[-1])):.6e}")

    print("\n=== RK4 Method ===")
    t_r, y_r = rk4_method(f, t_span, y0, h=0.1)
    print(f"Final error: {abs(y_r[-1] - exact(t_r[-1])):.6e}")

    print("\n=== Adaptive RK45 ===")
    t_a, y_a, h_used = adaptive_rk45(f, t_span, y0, tol=1e-8)
    print(f"Steps used: {len(t_a)}, Final error: {abs(y_a[-1] - exact(t_a[-1])):.6e}")

    print("\n=== Stiff Equation Demo ===")
    fig, results = stiff_equation_demo()
    for method, (t, y, n) in results.items():
        print(f"{method}: {n} steps")
    plt.show()
