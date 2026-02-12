"""
Problem 7: Laplace Transforms for Solving ODEs

Solve using Laplace transforms:
    y'' + 4y' + 3y = e^(-t),  y(0) = 0, y'(0) = 1

Tasks:
  (a) Take the Laplace transform of both sides
  (b) Solve for Y(s) algebraically
  (c) Use partial fraction decomposition to find Y(s) in simple terms
  (d) Invert to find y(t)
  (e) Verify with sympy's inverse_laplace_transform
"""

import numpy as np


def solve_with_sympy():
    """Use sympy to solve the ODE via Laplace transforms and directly."""
    import sympy as sp

    t, s = sp.symbols("t s", positive=True)

    # Direct ODE solution for verification
    y = sp.Function("y")
    ode = sp.Eq(y(t).diff(t, 2) + 4 * y(t).diff(t) + 3 * y(t), sp.exp(-t))
    direct_sol = sp.dsolve(ode, y(t), ics={y(0): 0, y(t).diff(t).subs(t, 0): 1})
    print("Direct ODE solution:", direct_sol)

    # Laplace transform approach
    # L{y''} = s^2*Y - s*y(0) - y'(0) = s^2*Y - 1
    # L{y'}  = s*Y - y(0) = s*Y
    # L{y}   = Y
    # L{e^(-t)} = 1/(s+1)
    Y = sp.Symbol("Y")
    lhs = s**2 * Y - 1 + 4 * (s * Y) + 3 * Y
    rhs = 1 / (s + 1)
    Y_sol = sp.solve(lhs - rhs, Y)[0]
    print("\nY(s) =", Y_sol)
    print("Partial fractions:", sp.apart(Y_sol, s))

    # Inverse Laplace
    y_t = sp.inverse_laplace_transform(Y_sol, s, t)
    print("y(t) =", sp.simplify(y_t))


def analytic_solution(t):
    """Return y(t) — the solution obtained via Laplace transforms."""
    # TODO: implement the closed-form solution
    raise NotImplementedError


def numerical_check():
    """Verify by numerical integration."""
    from scipy.integrate import solve_ivp

    def ode(t, state):
        y, yp = state
        return [yp, np.exp(-t) - 4 * yp - 3 * y]

    sol = solve_ivp(ode, [0, 5], [0, 1], t_eval=np.linspace(0, 5, 200))
    return sol.t, sol.y[0]


if __name__ == "__main__":
    solve_with_sympy()
    t, y_num = numerical_check()
    try:
        y_exact = analytic_solution(t)
        print(f"\nMax error: {np.max(np.abs(y_exact - y_num)):.2e}")
    except NotImplementedError:
        print("\nTODO: implement analytic_solution()")
