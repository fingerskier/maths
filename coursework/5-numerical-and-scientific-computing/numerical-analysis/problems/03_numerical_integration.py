"""
Numerical Integration
=====================

Implement classical quadrature rules and error analysis techniques for
approximating definite integrals.

Tasks
-----
1. Trapezoidal Rule: Implement the composite trapezoidal rule for approximating
   the integral of f over [a, b] using n subintervals. Return the approximate
   integral value.

2. Simpson's Rule: Implement the composite Simpson's 1/3 rule for approximating
   the integral of f over [a, b] using n subintervals (n must be even).

3. Gaussian Quadrature: Implement 2-point and 3-point Gauss-Legendre quadrature
   on [a, b]. Use the known nodes and weights for the reference interval [-1, 1]
   and transform to [a, b].

4. Richardson Extrapolation: Given a quadrature rule applied with step sizes h
   and h/2, use Richardson extrapolation to obtain a higher-order estimate. Apply
   this to the trapezoidal rule to derive Romberg integration.

5. Error Analysis: For a known integral (e.g., integral of sin(x) from 0 to pi = 2),
   compute the error of the trapezoidal and Simpson's rules as a function of n.
   Verify the theoretical convergence rates O(h^2) and O(h^4) respectively.
   Return arrays of n values and corresponding errors.
"""

import numpy as np


def trapezoidal_rule(f, a, b, n):
    """
    Composite trapezoidal rule.

    Parameters
    ----------
    f : callable
        Integrand.
    a, b : float
        Integration limits.
    n : int
        Number of subintervals.

    Returns
    -------
    integral : float
        Approximate integral value.
    """
    raise NotImplementedError


def simpsons_rule(f, a, b, n):
    """
    Composite Simpson's 1/3 rule.

    Parameters
    ----------
    f : callable
        Integrand.
    a, b : float
        Integration limits.
    n : int
        Number of subintervals (must be even).

    Returns
    -------
    integral : float
        Approximate integral value.
    """
    raise NotImplementedError


def gauss_legendre_2pt(f, a, b):
    """
    2-point Gauss-Legendre quadrature on [a, b].

    Parameters
    ----------
    f : callable
        Integrand.
    a, b : float
        Integration limits.

    Returns
    -------
    integral : float
        Approximate integral value.
    """
    raise NotImplementedError


def gauss_legendre_3pt(f, a, b):
    """
    3-point Gauss-Legendre quadrature on [a, b].

    Parameters
    ----------
    f : callable
        Integrand.
    a, b : float
        Integration limits.

    Returns
    -------
    integral : float
        Approximate integral value.
    """
    raise NotImplementedError


def richardson_extrapolation(f, a, b, n):
    """
    Apply Richardson extrapolation to the trapezoidal rule.

    Compute T(h) and T(h/2), then combine for a higher-order estimate.

    Parameters
    ----------
    f : callable
        Integrand.
    a, b : float
        Integration limits.
    n : int
        Initial number of subintervals.

    Returns
    -------
    improved_estimate : float
        Richardson-extrapolated integral estimate.
    """
    raise NotImplementedError


def error_analysis(exact_value=2.0):
    """
    Analyse convergence rates of trapezoidal and Simpson's rules.

    Integrate sin(x) from 0 to pi (exact value = 2) for various n values.

    Parameters
    ----------
    exact_value : float
        Exact value of the integral.

    Returns
    -------
    n_values : np.ndarray
        Array of subinterval counts used.
    trap_errors : np.ndarray
        Absolute errors from the trapezoidal rule.
    simp_errors : np.ndarray
        Absolute errors from Simpson's rule.
    """
    raise NotImplementedError


if __name__ == "__main__":
    f = np.sin
    a, b = 0.0, np.pi
    exact = 2.0

    print("=== Trapezoidal Rule ===")
    for n in [10, 100, 1000]:
        val = trapezoidal_rule(f, a, b, n)
        print(f"n={n:5d}: integral={val:.12f}, error={abs(val - exact):.2e}")

    print("\n=== Simpson's Rule ===")
    for n in [10, 100, 1000]:
        val = simpsons_rule(f, a, b, n)
        print(f"n={n:5d}: integral={val:.12f}, error={abs(val - exact):.2e}")

    print("\n=== Gaussian Quadrature ===")
    val2 = gauss_legendre_2pt(f, a, b)
    val3 = gauss_legendre_3pt(f, a, b)
    print(f"2-point: {val2:.12f}, error={abs(val2 - exact):.2e}")
    print(f"3-point: {val3:.12f}, error={abs(val3 - exact):.2e}")

    print("\n=== Richardson Extrapolation ===")
    val = richardson_extrapolation(f, a, b, 10)
    print(f"Extrapolated: {val:.12f}, error={abs(val - exact):.2e}")

    print("\n=== Error Analysis ===")
    n_vals, trap_err, simp_err = error_analysis()
    print(f"Trapezoidal rate: ~ h^{np.polyfit(np.log(1.0/n_vals), np.log(trap_err), 1)[0]:.2f}")
    print(f"Simpson's rate:   ~ h^{np.polyfit(np.log(1.0/n_vals), np.log(simp_err), 1)[0]:.2f}")
