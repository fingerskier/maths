"""
Problem 5: Lagrange Multipliers

Find the maximum and minimum values of f(x, y) = x^2 * y
subject to the constraint g(x, y) = x^2 + y^2 - 3 = 0.

Tasks:
  (a) Set up the Lagrange conditions: grad(f) = lambda * grad(g)
  (b) Solve the system of equations to find all critical points
  (c) Determine which points give the maximum and minimum values
  (d) Verify numerically by parameterizing the constraint curve
"""

import numpy as np


def find_critical_points():
    """
    Return a list of critical points [(x, y, lambda), ...] satisfying:
      2*x*y = lambda * 2*x
      x^2   = lambda * 2*y
      x^2 + y^2 = 3
    """
    # TODO: solve the system analytically, return list of tuples
    raise NotImplementedError


def classify_critical_points(points):
    """Given critical points, return (max_val, max_point, min_val, min_point)."""
    f = lambda x, y: x**2 * y
    values = [(f(x, y), (x, y)) for x, y, _ in points]
    max_val, max_pt = max(values, key=lambda t: t[0])
    min_val, min_pt = min(values, key=lambda t: t[0])
    return max_val, max_pt, min_val, min_pt


def verify_numerically(n=10000):
    """Parameterize the constraint circle and find extrema."""
    theta = np.linspace(0, 2 * np.pi, n)
    x = np.sqrt(3) * np.cos(theta)
    y = np.sqrt(3) * np.sin(theta)
    f_vals = x**2 * y
    i_max = np.argmax(f_vals)
    i_min = np.argmin(f_vals)
    return {
        "max": (f_vals[i_max], (x[i_max], y[i_max])),
        "min": (f_vals[i_min], (x[i_min], y[i_min])),
    }


if __name__ == "__main__":
    print("Numerical verification:", verify_numerically())
    try:
        pts = find_critical_points()
        print("Critical points:", pts)
        print("Classification:", classify_critical_points(pts))
    except NotImplementedError:
        print("TODO: implement find_critical_points()")
