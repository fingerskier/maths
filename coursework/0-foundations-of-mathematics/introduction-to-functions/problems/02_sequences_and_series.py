"""
Problem 2: Sequences and Series

Tasks:
  (a) Implement arithmetic and geometric sequence generators
  (b) Compute partial sums of series
  (c) Implement recursive sequences (Fibonacci, general linear recurrence)
  (d) Test convergence of series numerically using the ratio test and
      comparison test
  (e) Compute Taylor series approximations for exp(x), sin(x), cos(x)
"""

from math import factorial, inf


# ---------------------------------------------------------------------------
# Part (a): Arithmetic and geometric sequences
# ---------------------------------------------------------------------------

def arithmetic_sequence(a, d, n):
    """
    Generate the first n terms of an arithmetic sequence.

    The sequence is a, a+d, a+2d, ...

    Args:
        a: first term
        d: common difference
        n: number of terms to generate

    Returns:
        list of n terms

    Example:
        arithmetic_sequence(2, 3, 5) => [2, 5, 8, 11, 14]
    """
    raise NotImplementedError


def geometric_sequence(a, r, n):
    """
    Generate the first n terms of a geometric sequence.

    The sequence is a, a*r, a*r^2, ...

    Args:
        a: first term (non-zero)
        r: common ratio
        n: number of terms to generate

    Returns:
        list of n terms

    Example:
        geometric_sequence(3, 2, 5) => [3, 6, 12, 24, 48]
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (b): Partial sums
# ---------------------------------------------------------------------------

def partial_sum(term_func, n):
    """
    Compute the partial sum S_n = sum_{k=0}^{n-1} term_func(k).

    Args:
        term_func: callable(k) -> number, the k-th term of the series
        n: number of terms to sum

    Returns:
        The partial sum (a number).

    Example:
        partial_sum(lambda k: 1 / 2**k, 10)  # geometric series
    """
    raise NotImplementedError


def partial_sums(term_func, n):
    """
    Compute a list of partial sums S_1, S_2, ..., S_n.

    Args:
        term_func: callable(k) -> number, the k-th term (starting at k=0)
        n: number of partial sums to compute

    Returns:
        list of n partial sums where the i-th element is
        sum_{k=0}^{i} term_func(k).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (c): Recursive sequences
# ---------------------------------------------------------------------------

def fibonacci(n):
    """
    Compute the first n Fibonacci numbers.

    F(0) = 0, F(1) = 1, F(k) = F(k-1) + F(k-2).

    Args:
        n: number of terms

    Returns:
        list of the first n Fibonacci numbers.

    Example:
        fibonacci(8) => [0, 1, 1, 2, 3, 5, 8, 13]
    """
    raise NotImplementedError


def linear_recurrence(coeffs, initial, n):
    """
    Generate n terms of a linear recurrence relation.

    The recurrence is:
        a(k) = coeffs[0]*a(k-1) + coeffs[1]*a(k-2) + ...

    Args:
        coeffs: list of coefficients [c1, c2, ...] where
                a(k) = c1*a(k-1) + c2*a(k-2) + ...
        initial: list of initial values [a(0), a(1), ...] with
                 len(initial) == len(coeffs)
        n: total number of terms to generate

    Returns:
        list of n terms.

    Example (Fibonacci):
        linear_recurrence([1, 1], [0, 1], 8)
        => [0, 1, 1, 2, 3, 5, 8, 13]
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (d): Convergence tests
# ---------------------------------------------------------------------------

def ratio_test(term_func, start=1, num_terms=50):
    """
    Apply the ratio test to the series sum term_func(k).

    Compute the limit of |a(k+1) / a(k)| as k -> infinity
    (approximated over num_terms terms starting from start).

    Args:
        term_func: callable(k) -> number
        start: starting index (default 1 to avoid division by zero issues)
        num_terms: how many consecutive ratios to compute

    Returns:
        (ratio_limit, verdict) where:
            ratio_limit: float, the estimated limit of |a(k+1)/a(k)|
            verdict: one of "converges", "diverges", or "inconclusive"
                     (converges if ratio < 1, diverges if ratio > 1,
                      inconclusive if ratio == 1)
    """
    raise NotImplementedError


def comparison_test_geometric(term_func, r, start=0, num_terms=100):
    """
    Apply the comparison test against a geometric series with ratio r.

    If |term_func(k)| <= C * r^k for some constant C and 0 < r < 1,
    then the series converges.

    This function checks whether |a(k)| / r^k remains bounded.

    Args:
        term_func: callable(k) -> number
        r: comparison ratio (0 < r < 1)
        start: starting index
        num_terms: number of terms to check

    Returns:
        (bounded, max_ratio) where:
            bounded: True if |a(k)| / r^k stays bounded
            max_ratio: the maximum value of |a(k)| / r^k observed
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (e): Taylor series approximations
# ---------------------------------------------------------------------------

def taylor_exp(x, n_terms=20):
    """
    Approximate exp(x) using the first n_terms of its Taylor series.

    exp(x) = sum_{k=0}^{n_terms-1} x^k / k!

    Args:
        x: point at which to evaluate
        n_terms: number of terms in the Taylor expansion

    Returns:
        Approximation of exp(x).
    """
    raise NotImplementedError


def taylor_sin(x, n_terms=20):
    """
    Approximate sin(x) using the first n_terms non-zero terms of its
    Taylor series.

    sin(x) = sum_{k=0}^{n_terms-1} (-1)^k * x^(2k+1) / (2k+1)!

    Args:
        x: point at which to evaluate (radians)
        n_terms: number of non-zero terms

    Returns:
        Approximation of sin(x).
    """
    raise NotImplementedError


def taylor_cos(x, n_terms=20):
    """
    Approximate cos(x) using the first n_terms non-zero terms of its
    Taylor series.

    cos(x) = sum_{k=0}^{n_terms-1} (-1)^k * x^(2k) / (2k)!

    Args:
        x: point at which to evaluate (radians)
        n_terms: number of non-zero terms

    Returns:
        Approximation of cos(x).
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Part (a): Arithmetic & Geometric Sequences ===")
    try:
        print(f"  Arithmetic(2, 3, 6): {arithmetic_sequence(2, 3, 6)}")
        print(f"  Geometric(1, 0.5, 6): {geometric_sequence(1, 0.5, 6)}")
    except NotImplementedError:
        print("TODO: implement sequence generators")

    print()
    print("=== Part (b): Partial Sums ===")
    try:
        s = partial_sum(lambda k: 1 / factorial(k), 10)
        print(f"  sum(1/k!, k=0..9) = {s:.10f}  (e ~ 2.7182818285)")
        sums = partial_sums(lambda k: 1 / 2**k, 8)
        print(f"  Partial sums of 1/2^k: {[round(x, 4) for x in sums]}")
    except NotImplementedError:
        print("TODO: implement partial_sum / partial_sums")

    print()
    print("=== Part (c): Recursive Sequences ===")
    try:
        print(f"  Fibonacci(10): {fibonacci(10)}")
        lucas = linear_recurrence([1, 1], [2, 1], 10)
        print(f"  Lucas(10):     {lucas}")
    except NotImplementedError:
        print("TODO: implement fibonacci / linear_recurrence")

    print()
    print("=== Part (d): Convergence Tests ===")
    try:
        ratio, verdict = ratio_test(lambda k: 1 / factorial(k))
        print(f"  Ratio test for 1/k!: limit = {ratio:.6f}, {verdict}")
        ratio2, verdict2 = ratio_test(lambda k: k)
        print(f"  Ratio test for k:    limit = {ratio2:.6f}, {verdict2}")
    except NotImplementedError:
        print("TODO: implement ratio_test")

    print()
    print("=== Part (e): Taylor Series ===")
    try:
        from math import exp, sin, cos, pi
        x = 1.0
        print(f"  exp({x}):  Taylor = {taylor_exp(x):.10f},  "
              f"exact = {exp(x):.10f}")
        x = pi / 4
        print(f"  sin(pi/4): Taylor = {taylor_sin(x):.10f},  "
              f"exact = {sin(x):.10f}")
        print(f"  cos(pi/4): Taylor = {taylor_cos(x):.10f},  "
              f"exact = {cos(x):.10f}")
    except NotImplementedError:
        print("TODO: implement Taylor series functions")
