"""
Problem 7: Divergence Theorem and Stokes' Theorem

Part A - Divergence Theorem:
  Let F(x, y, z) = (x^3, y^3, z^3).
  Verify the Divergence Theorem for the unit sphere x^2 + y^2 + z^2 = 1:
      integral_V div(F) dV  ==  integral_S F . n dS

Part B - Stokes' Theorem:
  Let G(x, y, z) = (-y, x, z^2).
  Verify Stokes' Theorem for the hemisphere z = sqrt(1 - x^2 - y^2), z >= 0:
      integral_S curl(G) . n dS  ==  integral_C G . dr
  where C is the unit circle in the xy-plane.
"""

import numpy as np
from scipy import integrate


# --- Part A: Divergence Theorem ---

def divergence_F(x, y, z):
    """Compute div(F) = dF1/dx + dF2/dy + dF3/dz for F = (x^3, y^3, z^3)."""
    # TODO
    raise NotImplementedError


def volume_integral():
    """Evaluate integral of div(F) over the unit ball using spherical coords."""
    # TODO: set up triple integral in spherical coordinates
    raise NotImplementedError


def surface_flux_numerical():
    """Numerically compute the surface integral of F . n over the unit sphere."""
    # parameterize: x=sin(phi)cos(theta), y=sin(phi)sin(theta), z=cos(phi)
    def integrand(phi, theta):
        sp, cp = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)
        x, y, z = sp * ct, sp * st, cp
        # outward normal for unit sphere is (x, y, z), |n| = sin(phi)
        F_dot_n = x**4 + y**4 + z**4  # (x^3*x + y^3*y + z^3*z)
        return F_dot_n * sp  # include the sin(phi) from surface element

    result, _ = integrate.dblquad(integrand, 0, 2 * np.pi, 0, np.pi)
    return result


# --- Part B: Stokes' Theorem ---

def curl_G():
    """
    Compute curl(G) for G = (-y, x, z^2).
    Return as a symbolic description or implement evaluation function.
    """
    # curl(G) = (dG3/dy - dG2/dz, dG1/dz - dG3/dx, dG2/dx - dG1/dy)
    # TODO
    raise NotImplementedError


def line_integral_circle():
    """
    Evaluate integral_C G . dr around the unit circle in the xy-plane.
    Parameterize: r(t) = (cos(t), sin(t), 0), t in [0, 2*pi].
    """
    def integrand(t):
        x, y, z = np.cos(t), np.sin(t), 0
        dx, dy, dz = -np.sin(t), np.cos(t), 0
        return (-y) * dx + x * dy + z**2 * dz

    result, _ = integrate.quad(integrand, 0, 2 * np.pi)
    return result


if __name__ == "__main__":
    print("=== Divergence Theorem ===")
    print("Surface flux (numerical):", surface_flux_numerical())
    try:
        print("Volume integral:", volume_integral())
    except NotImplementedError:
        print("TODO: implement volume_integral()")

    print("\n=== Stokes' Theorem ===")
    print("Line integral around C:", line_integral_circle())
    try:
        print("curl(G):", curl_G())
    except NotImplementedError:
        print("TODO: implement curl_G()")
