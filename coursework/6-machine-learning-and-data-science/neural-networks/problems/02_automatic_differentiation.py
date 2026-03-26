"""
Automatic Differentiation
==========================

Implement both forward-mode and reverse-mode automatic differentiation from
scratch in pure Python. No external AD libraries allowed.

Tasks
-----
1. Forward-Mode AD: Implement a DualNumber class that carries a value and its
   derivative (tangent). Overload arithmetic operations (+, -, *, /, **) and
   implement elementary functions (sin, cos, exp, log). Use this to compute
   derivatives of scalar functions in a single forward pass.

2. Reverse-Mode AD (Backpropagation): Implement a Variable class that builds a
   computational graph during the forward pass and supports a backward() method
   to compute gradients via reverse accumulation. Support the same operations as
   forward mode.

3. Composite Function Gradients: Use both forward and reverse mode to compute
   gradients of composite functions such as f(x) = sin(x^2) * exp(-x) + log(x).
   Verify that both modes produce the same gradient.

4. Numerical Verification: For each computed derivative, verify against a
   finite-difference approximation: f'(x) ~ (f(x+h) - f(x-h)) / (2h) for
   small h. Report the discrepancy.
"""

import math


class DualNumber:
    """Dual number for forward-mode automatic differentiation."""

    def __init__(self, value, derivative=0.0):
        """
        Parameters
        ----------
        value : float
            The function value.
        derivative : float
            The derivative (tangent) value.
        """
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


def dual_sin(x):
    """Sine for DualNumber."""
    raise NotImplementedError


def dual_cos(x):
    """Cosine for DualNumber."""
    raise NotImplementedError


def dual_exp(x):
    """Exponential for DualNumber."""
    raise NotImplementedError


def dual_log(x):
    """Natural logarithm for DualNumber."""
    raise NotImplementedError


def forward_mode_derivative(f, x):
    """
    Compute f'(x) using forward-mode AD.

    Parameters
    ----------
    f : callable
        Function accepting a DualNumber and returning a DualNumber.
    x : float
        Point at which to evaluate the derivative.

    Returns
    -------
    value : float
        f(x).
    derivative : float
        f'(x).
    """
    raise NotImplementedError


class Variable:
    """Variable node for reverse-mode automatic differentiation."""

    def __init__(self, value, children=(), operation=""):
        """
        Parameters
        ----------
        value : float
            The numeric value.
        children : tuple
            Tuple of (Variable, local_gradient) pairs for backpropagation.
        operation : str
            Description of the operation that created this node.
        """
        raise NotImplementedError

    def backward(self):
        """
        Compute gradients via reverse-mode AD (backpropagation).

        Sets the .grad attribute on all ancestor Variable nodes.
        """
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


def var_sin(x):
    """Sine for Variable."""
    raise NotImplementedError


def var_cos(x):
    """Cosine for Variable."""
    raise NotImplementedError


def var_exp(x):
    """Exponential for Variable."""
    raise NotImplementedError


def var_log(x):
    """Natural logarithm for Variable."""
    raise NotImplementedError


def numerical_derivative(f, x, h=1e-7):
    """
    Compute f'(x) using central finite differences.

    Parameters
    ----------
    f : callable
        Function accepting a float and returning a float.
    x : float
        Point at which to compute the derivative.
    h : float
        Step size.

    Returns
    -------
    derivative : float
        Numerical approximation of f'(x).
    """
    raise NotImplementedError


def verify_gradients(f_dual, f_var, f_numeric, x, name="f"):
    """
    Verify that forward-mode, reverse-mode, and numerical derivatives agree.

    Parameters
    ----------
    f_dual : callable
        Function using DualNumber operations.
    f_var : callable
        Function using Variable operations.
    f_numeric : callable
        Plain Python function (float -> float).
    x : float
        Point at which to verify.
    name : str
        Function name for reporting.

    Returns
    -------
    results : dict
        Dictionary with 'forward', 'reverse', 'numerical' derivative values.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Forward-Mode AD ===")
    # f(x) = x^2 * sin(x), f'(x) = 2x*sin(x) + x^2*cos(x)
    def f_dual(x):
        return x ** 2 * dual_sin(x)

    val, deriv = forward_mode_derivative(f_dual, 2.0)
    exact = 2 * 2.0 * math.sin(2.0) + 4.0 * math.cos(2.0)
    print(f"f(2) = {val:.8f}, f'(2) = {deriv:.8f} (exact: {exact:.8f})")

    print("\n=== Reverse-Mode AD ===")
    x = Variable(2.0)
    y = x ** 2 * var_sin(x)
    y.backward()
    print(f"f(2) = {y.value:.8f}, f'(2) = {x.grad:.8f}")

    print("\n=== Numerical Verification ===")
    f_num = lambda x: x**2 * math.sin(x)
    nd = numerical_derivative(f_num, 2.0)
    print(f"Numerical derivative: {nd:.8f}")

    print("\n=== Composite Function Verification ===")
    # g(x) = sin(x^2) * exp(-x) + log(x)
    def g_dual(x):
        return dual_sin(x ** 2) * dual_exp(-x) + dual_log(x)

    def g_var(x):
        return var_sin(x ** 2) * var_exp(-x) + var_log(x)

    g_num = lambda x: math.sin(x**2) * math.exp(-x) + math.log(x)

    results = verify_gradients(g_dual, g_var, g_num, 1.5, name="g")
    for mode, val in results.items():
        print(f"  {mode}: {val:.10f}")
