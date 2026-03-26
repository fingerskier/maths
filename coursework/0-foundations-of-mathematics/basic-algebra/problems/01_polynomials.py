"""
Problem 1: Polynomials

Polynomials are represented as coefficient lists where index i holds the
coefficient of x^i.  For example, [3, 0, -2, 1] represents 3 - 2x^2 + x^3.

Tasks:
  (a) Implement polynomial addition and multiplication
  (b) Implement polynomial evaluation using Horner's method
  (c) Find roots of quadratic and cubic polynomials (real roots)
  (d) Implement polynomial long division, returning quotient and remainder
"""

from math import sqrt, acos, cos, pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_trailing_zeros(coeffs):
    """Remove trailing zero coefficients (but keep at least [0])."""
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs


# ---------------------------------------------------------------------------
# Part (a): Polynomial addition and multiplication
# ---------------------------------------------------------------------------

def poly_add(p, q):
    """
    Add two polynomials represented as coefficient lists.

    Args:
        p: list of coefficients [a0, a1, ..., an]
        q: list of coefficients [b0, b1, ..., bm]

    Returns:
        Coefficient list of p + q with no unnecessary trailing zeros.

    Example:
        poly_add([1, 2], [3, 0, 4]) => [4, 2, 4]   # (1+2x) + (3+4x^2)
    """
    raise NotImplementedError


def poly_multiply(p, q):
    """
    Multiply two polynomials represented as coefficient lists.

    Args:
        p: list of coefficients
        q: list of coefficients

    Returns:
        Coefficient list of p * q with no unnecessary trailing zeros.

    Example:
        poly_multiply([1, 1], [1, 1]) => [1, 2, 1]   # (1+x)^2
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (b): Polynomial evaluation — Horner's method
# ---------------------------------------------------------------------------

def poly_eval(p, x):
    """
    Evaluate polynomial p at value x using Horner's method.

    Horner's method rewrites a0 + a1*x + ... + an*x^n as
        a0 + x*(a1 + x*(a2 + ... + x*an)...)
    which uses n multiplications and n additions.

    Args:
        p: list of coefficients [a0, a1, ..., an]
        x: numeric value at which to evaluate

    Returns:
        The value p(x).

    Example:
        poly_eval([1, 0, -1], 3) => 1 + 0*3 + (-1)*9 = -8
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (c): Root finding for quadratics and cubics
# ---------------------------------------------------------------------------

def quadratic_roots(a, b, c):
    """
    Find the real roots of a*x^2 + b*x + c = 0.

    Args:
        a, b, c: coefficients (a != 0)

    Returns:
        Sorted list of real roots (may have 0, 1, or 2 elements).
        For a repeated root, return it once.

    Example:
        quadratic_roots(1, -3, 2) => [1.0, 2.0]
        quadratic_roots(1, 0, 1)  => []   # no real roots
    """
    raise NotImplementedError


def cubic_roots_real(a, b, c, d):
    """
    Find the real roots of a*x^3 + b*x^2 + c*x + d = 0.

    Use the analytical method (Cardano's formula or trigonometric method).
    Return all real roots as a sorted list.

    Args:
        a, b, c, d: coefficients (a != 0)

    Returns:
        Sorted list of real roots.

    Example:
        cubic_roots_real(1, 0, 0, -8) => [2.0]   # x^3 = 8
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (d): Polynomial long division
# ---------------------------------------------------------------------------

def poly_divmod(dividend, divisor):
    """
    Perform polynomial long division.

    Compute quotient Q and remainder R such that:
        dividend = divisor * Q + R
    with deg(R) < deg(divisor).

    Args:
        dividend: coefficient list of the dividend polynomial
        divisor:  coefficient list of the divisor polynomial (non-zero)

    Returns:
        (quotient, remainder) as coefficient lists.

    Example:
        poly_divmod([1, 0, 0, 1], [1, 1])
        # (x^3 + 1) / (x + 1) => quotient = x^2 - x + 1, remainder = 0
        => ([1, -1, 1], [0])
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Part (a): Polynomial Addition & Multiplication ===")
    try:
        s = poly_add([1, 2, 3], [4, 5])
        print(f"  (1 + 2x + 3x^2) + (4 + 5x) = {s}")
        p = poly_multiply([1, 1], [1, -1])
        print(f"  (1 + x)(1 - x) = {p}")
    except NotImplementedError:
        print("TODO: implement poly_add and poly_multiply")

    print()
    print("=== Part (b): Horner's Method ===")
    try:
        val = poly_eval([2, -3, 0, 1], 2)
        print(f"  p(x) = 2 - 3x + x^3, p(2) = {val}")
    except NotImplementedError:
        print("TODO: implement poly_eval")

    print()
    print("=== Part (c): Root Finding ===")
    try:
        r2 = quadratic_roots(1, -5, 6)
        print(f"  Roots of x^2 - 5x + 6: {r2}")
        r3 = cubic_roots_real(1, -6, 11, -6)
        print(f"  Roots of x^3 - 6x^2 + 11x - 6: {r3}")
    except NotImplementedError:
        print("TODO: implement quadratic_roots and cubic_roots_real")

    print()
    print("=== Part (d): Polynomial Long Division ===")
    try:
        q, r = poly_divmod([1, 0, 0, 1], [1, 1])
        print(f"  (x^3 + 1) / (x + 1): quotient={q}, remainder={r}")
        q2, r2 = poly_divmod([1, 0, 1], [1, 1])
        print(f"  (x^2 + 1) / (x + 1): quotient={q2}, remainder={r2}")
    except NotImplementedError:
        print("TODO: implement poly_divmod")
