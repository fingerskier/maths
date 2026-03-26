"""
Problem: Regression Basics

Implement linear regression from scratch using ordinary least squares (OLS).

Tasks:
  (a) Implement simple linear regression (one predictor) via closed-form
      OLS formulas: slope and intercept
  (b) Compute R-squared (coefficient of determination)
  (c) Perform residual analysis: compute residuals, check for patterns
  (d) Implement multiple linear regression via the normal equations:
      beta = (X^T X)^{-1} X^T y
"""

import numpy as np
import matplotlib.pyplot as plt


def simple_linear_regression(x, y):
    """
    Fit a simple linear regression y = a + b * x using OLS.

    Formulas:
        b = sum((x_i - x_bar)(y_i - y_bar)) / sum((x_i - x_bar)^2)
        a = y_bar - b * x_bar

    Parameters:
        x: np.ndarray, shape (n,), predictor values
        y: np.ndarray, shape (n,), response values

    Returns:
        dict with keys:
            'intercept': float, the estimated intercept (a)
            'slope': float, the estimated slope (b)
    """
    # TODO
    raise NotImplementedError


def predict_simple(x, intercept, slope):
    """
    Predict y values from a simple linear model.

    Parameters:
        x: np.ndarray, shape (n,)
        intercept: float
        slope: float

    Returns:
        np.ndarray, shape (n,): predicted y values
    """
    # TODO
    raise NotImplementedError


def r_squared(y_true, y_pred):
    """
    Compute the coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot
    where SS_res = sum((y_i - y_hat_i)^2) and SS_tot = sum((y_i - y_bar)^2)

    Parameters:
        y_true: np.ndarray, shape (n,), actual values
        y_pred: np.ndarray, shape (n,), predicted values

    Returns:
        float: R^2 value
    """
    # TODO
    raise NotImplementedError


def compute_residuals(y_true, y_pred):
    """
    Compute residuals e_i = y_i - y_hat_i.

    Parameters:
        y_true: np.ndarray, shape (n,)
        y_pred: np.ndarray, shape (n,)

    Returns:
        np.ndarray, shape (n,): residuals
    """
    # TODO
    raise NotImplementedError


def residual_analysis(x, y_true, y_pred):
    """
    Perform basic residual diagnostics.

    Parameters:
        x: np.ndarray, shape (n,), predictor values
        y_true: np.ndarray, shape (n,)
        y_pred: np.ndarray, shape (n,)

    Returns:
        dict with keys:
            'residuals': np.ndarray, shape (n,)
            'mean_residual': float (should be near 0)
            'std_residual': float
            'residual_sum_of_squares': float
    """
    # TODO
    raise NotImplementedError


def multiple_linear_regression(X, y):
    """
    Fit a multiple linear regression y = X @ beta using the normal equations.

    The design matrix X should already include a column of ones for the
    intercept if desired.

    beta = (X^T X)^{-1} X^T y

    Parameters:
        X: np.ndarray, shape (n, p), design matrix
        y: np.ndarray, shape (n,), response vector

    Returns:
        dict with keys:
            'coefficients': np.ndarray, shape (p,), estimated beta
            'predictions': np.ndarray, shape (n,), X @ beta
            'r_squared': float
    """
    # TODO
    raise NotImplementedError


def plot_regression(x, y, intercept, slope):
    """
    Plot data points and the fitted regression line.

    Parameters:
        x: np.ndarray, shape (n,)
        y: np.ndarray, shape (n,)
        intercept: float
        slope: float

    Returns:
        matplotlib.figure.Figure
    """
    # TODO
    raise NotImplementedError


def plot_residuals(y_pred, residuals):
    """
    Plot residuals vs predicted values to check for patterns.

    Parameters:
        y_pred: np.ndarray, shape (n,)
        residuals: np.ndarray, shape (n,)

    Returns:
        matplotlib.figure.Figure
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Generate data: y = 3 + 2*x + noise
    x = rng.uniform(0, 10, size=50)
    y = 3 + 2 * x + rng.normal(0, 1.5, size=50)

    try:
        model = simple_linear_regression(x, y)
        print(f"Simple LR: intercept={model['intercept']:.3f} (true 3.0), "
              f"slope={model['slope']:.3f} (true 2.0)")

        y_pred = predict_simple(x, model['intercept'], model['slope'])
        r2 = r_squared(y, y_pred)
        print(f"R-squared: {r2:.4f}")
    except NotImplementedError:
        print("TODO: implement simple_linear_regression / predict_simple / r_squared")

    try:
        y_pred = predict_simple(x, model['intercept'], model['slope'])
        analysis = residual_analysis(x, y, y_pred)
        print(f"\nResidual analysis:")
        print(f"  Mean residual: {analysis['mean_residual']:.4f}")
        print(f"  Std residual: {analysis['std_residual']:.3f}")
        print(f"  RSS: {analysis['residual_sum_of_squares']:.3f}")
    except NotImplementedError:
        print("TODO: implement residual_analysis")

    # Multiple regression: y = 1 + 2*x1 + 3*x2 + noise
    try:
        x1 = rng.uniform(0, 10, size=100)
        x2 = rng.uniform(0, 5, size=100)
        y_multi = 1 + 2 * x1 + 3 * x2 + rng.normal(0, 1.0, size=100)

        X = np.column_stack([np.ones(100), x1, x2])
        result = multiple_linear_regression(X, y_multi)
        print(f"\nMultiple LR coefficients: {result['coefficients']}")
        print(f"(Expected: [1.0, 2.0, 3.0])")
        print(f"R-squared: {result['r_squared']:.4f}")
    except NotImplementedError:
        print("TODO: implement multiple_linear_regression")
