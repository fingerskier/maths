"""
Root Finding Methods
====================

Implement and compare classical root-finding algorithms for nonlinear equations.

Tasks
-----
1. Bisection Method: Implement the bisection method to find a root of f(x) = 0
   on a given interval [a, b]. Return the approximate root and the number of
   iterations required to achieve a specified tolerance.

2. Newton-Raphson Method: Implement Newton's method using f and f'. Starting from
   an initial guess x0, iterate x_{n+1} = x_n - f(x_n)/f'(x_n) until convergence.
   Handle cases where f'(x_n) is near zero.

3. Secant Method: Implement the secant method, which approximates the derivative
   using two previous iterates. Starting from x0 and x1, iterate until convergence.

4. Fixed-Point Iteration: Implement fixed-point iteration x_{n+1} = g(x_n) for a
   given function g. Detect divergence and return the fixed point if convergence
   is achieved.

5. Convergence Rate Comparison: For the equation f(x) = x^3 - 2x - 5 = 0 (which
   has a root near x = 2.0946), run all four methods and return a dictionary mapping
   method names to lists of successive errors |x_n - x*|. This allows comparison
   of convergence rates (linear for bisection, superlinear for secant, quadratic
   for Newton).

Test functions:
- f1(x) = x^3 - 2x - 5,  root near 2.0946
- f2(x) = cos(x) - x,     root near 0.7391
"""

import numpy as np


def bisection(f, a, b, tol=1e-10, max_iter=1000):
    """
    Find a root of f in [a, b] using the bisection method.

    Parameters
    ----------
    f : callable
        Continuous function with f(a) and f(b) of opposite sign.
    a, b : float
        Interval endpoints.
    tol : float
        Stopping tolerance on interval width.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    root : float
        Approximate root.
    iterations : int
        Number of iterations performed.
    """
    raise NotImplementedError


def newton_raphson(f, f_prime, x0, tol=1e-10, max_iter=1000):
    """
    Find a root of f using Newton-Raphson iteration.

    Parameters
    ----------
    f : callable
        Function whose root is sought.
    f_prime : callable
        Derivative of f.
    x0 : float
        Initial guess.
    tol : float
        Stopping tolerance on |f(x)|.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    root : float
        Approximate root.
    iterations : int
        Number of iterations performed.
    """
    raise NotImplementedError


def secant_method(f, x0, x1, tol=1e-10, max_iter=1000):
    """
    Find a root of f using the secant method.

    Parameters
    ----------
    f : callable
        Function whose root is sought.
    x0, x1 : float
        Two initial guesses.
    tol : float
        Stopping tolerance on |f(x)|.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    root : float
        Approximate root.
    iterations : int
        Number of iterations performed.
    """
    raise NotImplementedError


def fixed_point_iteration(g, x0, tol=1e-10, max_iter=1000):
    """
    Find a fixed point of g(x) = x using iteration x_{n+1} = g(x_n).

    Parameters
    ----------
    g : callable
        Function for which g(x*) = x* is sought.
    x0 : float
        Initial guess.
    tol : float
        Stopping tolerance on |x_{n+1} - x_n|.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    fixed_point : float
        Approximate fixed point.
    iterations : int
        Number of iterations performed.
    converged : bool
        Whether the method converged within max_iter.
    """
    raise NotImplementedError


def convergence_rate_comparison(tol=1e-12):
    """
    Compare convergence rates of all four methods on f(x) = x^3 - 2x - 5.

    Returns
    -------
    errors : dict
        Dictionary with keys 'bisection', 'newton', 'secant', 'fixed_point',
        each mapping to a list of absolute errors |x_n - x*| at each iteration.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Test function: f(x) = x^3 - 2x - 5
    f = lambda x: x**3 - 2 * x - 5
    f_prime = lambda x: 3 * x**2 - 2

    print("=== Bisection Method ===")
    root, iters = bisection(f, 2.0, 3.0)
    print(f"Root: {root:.10f}, Iterations: {iters}")

    print("\n=== Newton-Raphson Method ===")
    root, iters = newton_raphson(f, f_prime, 2.0)
    print(f"Root: {root:.10f}, Iterations: {iters}")

    print("\n=== Secant Method ===")
    root, iters = secant_method(f, 2.0, 3.0)
    print(f"Root: {root:.10f}, Iterations: {iters}")

    print("\n=== Fixed-Point Iteration ===")
    g = lambda x: (x**3 - 5) / 2.0
    fp, iters, converged = fixed_point_iteration(g, 2.0)
    print(f"Fixed point: {fp:.10f}, Iterations: {iters}, Converged: {converged}")

    print("\n=== Convergence Rate Comparison ===")
    errors = convergence_rate_comparison()
    for method, errs in errors.items():
        print(f"{method}: {len(errs)} iterations, final error = {errs[-1]:.2e}")
