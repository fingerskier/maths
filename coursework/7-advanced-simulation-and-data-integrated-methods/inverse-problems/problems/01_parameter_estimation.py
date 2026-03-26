"""
Parameter Estimation for Inverse Problems
==========================================

Inverse problems involve determining model parameters from observed data.
Unlike forward problems (given parameters, compute output), inverse problems
are often ill-posed: small perturbations in data can lead to large changes
in estimated parameters.

Consider the ODE system:
    dy/dt = -k * y,  y(0) = y0
where the true parameters are k = 0.5 and y0 = 2.0. We observe noisy
measurements y_obs(t_i) = y_true(t_i) + noise at discrete time points.

Tasks
-----
1. Solve the inverse problem: given noisy observations, estimate the
   parameters (k, y0) by minimizing the data misfit. Implement a forward
   model solver and a residual function for the parameter estimation.

2. Implement ordinary least squares (OLS) fitting by minimizing
       J(k, y0) = sum_i (y_model(t_i; k, y0) - y_obs(t_i))^2
   Use scipy.optimize.least_squares or implement Gauss-Newton iteration.

3. Implement Tikhonov regularization to stabilize the inversion:
       J_reg(m) = ||F(m) - d||^2 + alpha * ||L(m - m_prior)||^2
   where alpha is the regularization parameter, L is the regularization
   matrix (identity or derivative operator), and m_prior is a prior guess.

4. Implement the L-curve method for choosing the optimal regularization
   parameter alpha. Plot the trade-off curve (log||residual|| vs
   log||solution norm||) and identify the corner.
"""

import numpy as np
import scipy


def forward_model(t, k, y0):
    """
    Solve the forward ODE dy/dt = -k*y, y(0) = y0.

    Parameters
    ----------
    t : np.ndarray
        Time points at which to evaluate the solution.
    k : float
        Decay rate parameter.
    y0 : float
        Initial condition.

    Returns
    -------
    np.ndarray
        Solution y(t).
    """
    raise NotImplementedError


def generate_observations(t, k_true, y0_true, noise_std=0.1, seed=42):
    """
    Generate synthetic noisy observations from the true model.

    Parameters
    ----------
    t : np.ndarray
        Observation time points.
    k_true : float
        True decay rate.
    y0_true : float
        True initial condition.
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy observations.
    """
    raise NotImplementedError


def least_squares_estimate(t, y_obs, initial_guess):
    """
    Estimate parameters (k, y0) using ordinary least squares.

    Parameters
    ----------
    t : np.ndarray
        Observation time points.
    y_obs : np.ndarray
        Noisy observations.
    initial_guess : tuple
        Initial guess (k0, y0_0) for the optimization.

    Returns
    -------
    k_est : float
        Estimated decay rate.
    y0_est : float
        Estimated initial condition.
    residual_norm : float
        Norm of the residual at the solution.
    """
    raise NotImplementedError


def tikhonov_regularization(t, y_obs, alpha, m_prior, initial_guess):
    """
    Estimate parameters using Tikhonov regularization.

    Parameters
    ----------
    t : np.ndarray
        Observation time points.
    y_obs : np.ndarray
        Noisy observations.
    alpha : float
        Regularization parameter.
    m_prior : np.ndarray
        Prior parameter estimate [k_prior, y0_prior].
    initial_guess : tuple
        Initial guess for the optimization.

    Returns
    -------
    k_est : float
        Estimated decay rate.
    y0_est : float
        Estimated initial condition.
    residual_norm : float
        Data misfit norm.
    solution_norm : float
        Regularization term norm.
    """
    raise NotImplementedError


def l_curve_analysis(t, y_obs, alphas, m_prior, initial_guess):
    """
    Compute L-curve data for a range of regularization parameters and
    identify the optimal alpha at the curve's corner.

    Parameters
    ----------
    t : np.ndarray
        Observation time points.
    y_obs : np.ndarray
        Noisy observations.
    alphas : np.ndarray
        Array of regularization parameters to test.
    m_prior : np.ndarray
        Prior parameter estimate.
    initial_guess : tuple
        Initial guess for the optimization.

    Returns
    -------
    residual_norms : np.ndarray
        Data misfit norms for each alpha.
    solution_norms : np.ndarray
        Solution norms for each alpha.
    alpha_opt : float
        Optimal regularization parameter (at L-curve corner).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Setup
    k_true, y0_true = 0.5, 2.0
    t_obs = np.linspace(0, 10, 20)
    y_obs = generate_observations(t_obs, k_true, y0_true, noise_std=0.1)

    # Task 2: Least squares
    k_est, y0_est, res_norm = least_squares_estimate(t_obs, y_obs, (1.0, 1.0))
    print(f"OLS estimate: k = {k_est:.4f} (true: {k_true}), y0 = {y0_est:.4f} (true: {y0_true})")
    print(f"Residual norm: {res_norm:.6f}")

    # Task 3: Tikhonov regularization
    m_prior = np.array([1.0, 1.0])
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        k_r, y0_r, rn, sn = tikhonov_regularization(
            t_obs, y_obs, alpha, m_prior, (1.0, 1.0)
        )
        print(f"Tikhonov (alpha={alpha}): k = {k_r:.4f}, y0 = {y0_r:.4f}")

    # Task 4: L-curve analysis
    alphas = np.logspace(-4, 2, 50)
    res_norms, sol_norms, alpha_opt = l_curve_analysis(
        t_obs, y_obs, alphas, m_prior, (1.0, 1.0)
    )
    print(f"Optimal regularization parameter: alpha = {alpha_opt:.6f}")
