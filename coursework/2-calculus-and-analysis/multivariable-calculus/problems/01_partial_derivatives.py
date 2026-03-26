"""
Problem 1: Partial Derivatives

Compute the partial derivatives of a function f(x, y) = x^2*y + y^3 - 2*x*y^2.

Tasks:
  (a) Implement df/dx (partial derivative with respect to x)
  (b) Implement df/dy (partial derivative with respect to y)
  (c) Evaluate both at the point (1, 2)

Use symbolic differentiation (sympy) to verify your analytic answers.
"""


def f(x, y):
    return x**2 * y + y**3 - 2 * x * y**2


def df_dx(x, y):
    """Return the partial derivative of f with respect to x, evaluated at (x, y)."""
    # TODO: implement analytically
    raise NotImplementedError


def df_dy(x, y):
    """Return the partial derivative of f with respect to y, evaluated at (x, y)."""
    # TODO: implement analytically
    raise NotImplementedError


def verify_symbolic():
    """Use sympy to verify partial derivatives."""
    import sympy as sp

    x, y = sp.symbols("x y")
    expr = x**2 * y + y**3 - 2 * x * y**2
    print("df/dx =", sp.diff(expr, x))
    print("df/dy =", sp.diff(expr, y))
    print("df/dx at (1,2) =", sp.diff(expr, x).subs([(x, 1), (y, 2)]))
    print("df/dy at (1,2) =", sp.diff(expr, y).subs([(x, 1), (y, 2)]))


if __name__ == "__main__":
    print("df/dx(1,2) =", df_dx(1, 2))
    print("df/dy(1,2) =", df_dy(1, 2))
    print("\nSymbolic verification:")
    verify_symbolic()
