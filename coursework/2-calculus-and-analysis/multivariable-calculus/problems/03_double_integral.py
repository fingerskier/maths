"""
Problem 3: Double Integration

Evaluate the double integral of f(x, y) = x^2 + y^2 over the region D,
where D is the disk x^2 + y^2 <= 4.

Tasks:
  (a) Set up and evaluate the integral in Cartesian coordinates
  (b) Convert to polar coordinates and evaluate
  (c) Verify numerically using scipy.integrate.dblquad

Hint: In polar coordinates, x = r*cos(theta), y = r*sin(theta), dA = r dr d(theta).
"""

import numpy as np


def integrate_cartesian():
    """Evaluate the double integral in Cartesian coordinates (return exact value)."""
    # TODO: return the analytic result
    raise NotImplementedError


def integrate_polar():
    """Evaluate using polar coordinate substitution (return exact value)."""
    # TODO: integral of (r^2) * r dr d(theta) from r=0..2, theta=0..2*pi
    raise NotImplementedError


def integrate_numerical():
    """Verify using scipy numerical integration."""
    from scipy import integrate

    def integrand(y, x):
        return x**2 + y**2

    result, error = integrate.dblquad(
        integrand,
        -2, 2,
        lambda x: -np.sqrt(4 - x**2),
        lambda x: np.sqrt(4 - x**2),
    )
    return result


if __name__ == "__main__":
    print("Cartesian result:", integrate_cartesian())
    print("Polar result:", integrate_polar())
    print("Numerical result:", integrate_numerical())
