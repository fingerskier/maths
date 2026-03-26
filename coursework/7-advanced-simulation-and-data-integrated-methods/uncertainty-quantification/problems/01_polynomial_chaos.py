"""
Polynomial Chaos Expansion for Uncertainty Quantification
==========================================================

Polynomial chaos expansion (PCE) provides a spectral approach to propagating
uncertainty through computational models. Instead of running expensive Monte
Carlo simulations, PCE represents the stochastic solution as a series expansion
in orthogonal polynomials of the random inputs.

Consider a simple stochastic ODE:
    dy/dt = -a * y,  y(0) = 1
where the decay rate 'a' is uncertain, modeled as a uniform random variable
on [0.5, 1.5].

Tasks
-----
1. Implement a polynomial chaos expansion using Legendre polynomials (appropriate
   for uniform input distributions). Represent the solution y(t; a) as:
       y(t; a) ≈ sum_{k=0}^{P} c_k(t) * L_k(xi)
   where xi is the standard uniform variable mapped to [-1, 1] and L_k are
   Legendre polynomials. Use Galerkin projection or collocation to find the
   coefficients c_k(t).

2. Compute the mean and variance of the output y(t) from the PCE coefficients:
       E[y] = c_0(t)
       Var[y] = sum_{k=1}^{P} c_k(t)^2 * <L_k^2>
   Evaluate at several time points and plot the evolution of mean and variance.

3. Compare the PCE-based statistics with brute-force Monte Carlo simulation
   (e.g., 10,000 samples). Assess convergence of PCE as polynomial order P
   increases and compare computational cost.

4. Perform sensitivity analysis via Sobol indices derived from the PCE
   coefficients. For this single-input problem the first-order Sobol index
   should be 1; extend to a two-parameter case (uncertain initial condition
   and decay rate) and compute first-order indices for each parameter.
"""

import numpy as np


def legendre_polynomials(order, xi):
    """
    Evaluate Legendre polynomials up to the given order at points xi in [-1, 1].

    Parameters
    ----------
    order : int
        Maximum polynomial order.
    xi : np.ndarray
        Evaluation points in [-1, 1].

    Returns
    -------
    np.ndarray
        Array of shape (order + 1, len(xi)) with polynomial values.
    """
    raise NotImplementedError


def pce_coefficients_collocation(order, n_collocation, t_eval):
    """
    Compute PCE coefficients for the stochastic ODE dy/dt = -a*y using
    stochastic collocation (pseudospectral projection).

    Parameters
    ----------
    order : int
        Polynomial chaos expansion order P.
    n_collocation : int
        Number of collocation (quadrature) points.
    t_eval : np.ndarray
        Time points at which to evaluate the solution.

    Returns
    -------
    np.ndarray
        PCE coefficients of shape (order + 1, len(t_eval)).
    """
    raise NotImplementedError


def compute_mean_variance(coefficients):
    """
    Compute mean and variance of the output from PCE coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        PCE coefficients of shape (order + 1, n_times).

    Returns
    -------
    mean : np.ndarray
        Mean at each time point.
    variance : np.ndarray
        Variance at each time point.
    """
    raise NotImplementedError


def monte_carlo_statistics(n_samples, t_eval):
    """
    Estimate mean and variance of the stochastic ODE output using Monte Carlo.

    Parameters
    ----------
    n_samples : int
        Number of random samples.
    t_eval : np.ndarray
        Time points at which to evaluate the solution.

    Returns
    -------
    mean : np.ndarray
        Estimated mean at each time point.
    variance : np.ndarray
        Estimated variance at each time point.
    """
    raise NotImplementedError


def sobol_indices_from_pce(coefficients_2d, index_sets):
    """
    Compute first-order Sobol indices from PCE coefficients for a
    two-parameter model.

    Parameters
    ----------
    coefficients_2d : np.ndarray
        PCE coefficients for the two-parameter expansion.
    index_sets : list of list of int
        Multi-index sets indicating which polynomial indices correspond
        to each input parameter.

    Returns
    -------
    S1 : list of float
        First-order Sobol indices for each parameter.
    """
    raise NotImplementedError


if __name__ == "__main__":
    t_eval = np.linspace(0, 5, 50)

    # Task 1: PCE with increasing order
    for order in [2, 4, 6]:
        coeffs = pce_coefficients_collocation(order, order + 1, t_eval)
        mean, var = compute_mean_variance(coeffs)
        print(f"PCE order {order}: mean(t=5) = {mean[-1]:.6f}, var(t=5) = {var[-1]:.6f}")

    # Task 3: Monte Carlo comparison
    mc_mean, mc_var = monte_carlo_statistics(10000, t_eval)
    print(f"Monte Carlo: mean(t=5) = {mc_mean[-1]:.6f}, var(t=5) = {mc_var[-1]:.6f}")

    # Task 4: Sobol indices (placeholder call)
    print("Sobol indices computation (two-parameter extension)...")
