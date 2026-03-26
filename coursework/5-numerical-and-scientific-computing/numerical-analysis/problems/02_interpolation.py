"""
Interpolation Methods
=====================

Implement polynomial and spline interpolation techniques and explore their
properties, including the Runge phenomenon.

Tasks
-----
1. Lagrange Interpolation: Given n+1 data points (x_i, y_i), construct the
   Lagrange interpolating polynomial and evaluate it at arbitrary points.
   Return the interpolated values.

2. Newton Divided Differences: Build the Newton form of the interpolating
   polynomial using a divided difference table. Return the divided difference
   coefficients and evaluate the polynomial at given points.

3. Cubic Spline Interpolation: Implement natural cubic spline interpolation
   (second derivatives zero at endpoints). Given data points, compute spline
   coefficients and evaluate the spline at arbitrary points.

4. Runge Phenomenon Demonstration: Interpolate f(x) = 1/(1 + 25x^2) on [-1, 1]
   using equispaced nodes for n = 5, 10, 15, 20. Plot the interpolants against
   the true function to demonstrate oscillation at the boundaries. Then repeat
   with Chebyshev nodes to show the improvement.
"""

import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x_data, y_data, x_eval):
    """
    Evaluate the Lagrange interpolating polynomial at given points.

    Parameters
    ----------
    x_data : array_like
        Interpolation nodes (n+1 points).
    y_data : array_like
        Function values at the nodes.
    x_eval : array_like
        Points at which to evaluate the interpolant.

    Returns
    -------
    y_eval : np.ndarray
        Interpolated values at x_eval.
    """
    raise NotImplementedError


def newton_divided_differences(x_data, y_data):
    """
    Compute the Newton divided difference coefficients.

    Parameters
    ----------
    x_data : array_like
        Interpolation nodes.
    y_data : array_like
        Function values at the nodes.

    Returns
    -------
    coeffs : np.ndarray
        Divided difference coefficients (top row of the table).
    """
    raise NotImplementedError


def newton_interpolation_eval(x_data, coeffs, x_eval):
    """
    Evaluate the Newton interpolating polynomial at given points.

    Parameters
    ----------
    x_data : array_like
        Interpolation nodes.
    coeffs : np.ndarray
        Divided difference coefficients from newton_divided_differences.
    x_eval : array_like
        Points at which to evaluate the interpolant.

    Returns
    -------
    y_eval : np.ndarray
        Interpolated values at x_eval.
    """
    raise NotImplementedError


def cubic_spline_natural(x_data, y_data):
    """
    Compute natural cubic spline coefficients.

    Parameters
    ----------
    x_data : array_like
        Sorted interpolation nodes.
    y_data : array_like
        Function values at the nodes.

    Returns
    -------
    coeffs : list of tuples
        List of (a_i, b_i, c_i, d_i) for each spline segment, where
        S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3.
    """
    raise NotImplementedError


def cubic_spline_eval(x_data, coeffs, x_eval):
    """
    Evaluate a cubic spline at given points.

    Parameters
    ----------
    x_data : array_like
        Sorted interpolation nodes.
    coeffs : list of tuples
        Spline coefficients from cubic_spline_natural.
    x_eval : array_like
        Points at which to evaluate the spline.

    Returns
    -------
    y_eval : np.ndarray
        Spline values at x_eval.
    """
    raise NotImplementedError


def runge_phenomenon_demo():
    """
    Demonstrate the Runge phenomenon and the improvement with Chebyshev nodes.

    Interpolate f(x) = 1/(1 + 25x^2) on [-1, 1] using equispaced and Chebyshev
    nodes for n = 5, 10, 15, 20. Plot results.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with subplots comparing equispaced vs Chebyshev interpolation.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Test data
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.sin(x_data)
    x_eval = np.linspace(0, 4, 50)

    print("=== Lagrange Interpolation ===")
    y_lag = lagrange_interpolation(x_data, y_data, x_eval)
    print(f"Interpolated {len(x_eval)} points from {len(x_data)} nodes")

    print("\n=== Newton Divided Differences ===")
    coeffs = newton_divided_differences(x_data, y_data)
    print(f"Coefficients: {coeffs}")
    y_newt = newton_interpolation_eval(x_data, coeffs, x_eval)
    print(f"Max difference from Lagrange: {np.max(np.abs(y_lag - y_newt)):.2e}")

    print("\n=== Cubic Spline ===")
    spline_coeffs = cubic_spline_natural(x_data, y_data)
    y_spline = cubic_spline_eval(x_data, spline_coeffs, x_eval)
    print(f"Spline max error vs sin: {np.max(np.abs(y_spline - np.sin(x_eval))):.2e}")

    print("\n=== Runge Phenomenon ===")
    fig = runge_phenomenon_demo()
    plt.show()
