"""
Problem 6: Line Integrals and Green's Theorem

Let F(x, y) = (y^2, 2*x*y + 3) be a vector field.

Tasks:
  (a) Compute the line integral of F along the curve C: r(t) = (t, t^2)
      for t in [0, 1] directly via parameterization
  (b) Determine if F is conservative (check if dF1/dy == dF2/dx)
  (c) If conservative, find the potential function phi such that F = grad(phi)
  (d) Verify using the Fundamental Theorem of Line Integrals
"""

import numpy as np
from scipy import integrate


def line_integral_parameterized():
    """
    Evaluate integral_C F . dr using the parameterization r(t) = (t, t^2).

    F . dr = F1*dx + F2*dy = y^2*dx + (2xy+3)*dy
    With x=t, y=t^2: dx=dt, dy=2t*dt
    """
    def integrand(t):
        x, y = t, t**2
        dx_dt, dy_dt = 1, 2 * t
        return y**2 * dx_dt + (2 * x * y + 3) * dy_dt

    result, _ = integrate.quad(integrand, 0, 1)
    return result


def is_conservative():
    """Check if dF1/dy == dF2/dx. Return True/False."""
    # F1 = y^2, F2 = 2*x*y + 3
    # TODO: compute partial derivatives and compare
    raise NotImplementedError


def potential_function(x, y):
    """Return phi(x, y) such that grad(phi) = F, if F is conservative."""
    # TODO: find phi by integrating F1 w.r.t. x, then matching with F2
    raise NotImplementedError


def verify_ftli():
    """
    Fundamental Theorem of Line Integrals:
    integral_C F . dr = phi(r(1)) - phi(r(0))
    r(0) = (0, 0), r(1) = (1, 1)
    """
    return potential_function(1, 1) - potential_function(0, 0)


if __name__ == "__main__":
    print("Line integral (parameterized):", line_integral_parameterized())
    try:
        print("Is conservative?", is_conservative())
        print("phi(1,1) - phi(0,0) =", verify_ftli())
    except NotImplementedError:
        print("TODO: implement is_conservative() and potential_function()")
