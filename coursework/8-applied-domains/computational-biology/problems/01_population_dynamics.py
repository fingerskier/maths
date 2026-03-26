"""
Population Dynamics Models
===========================

Mathematical models of population dynamics are fundamental in computational
biology. This problem explores several classic models with increasing
complexity.

Tasks
-----
1. Implement the SIR (Susceptible-Infected-Recovered) epidemic model:
       dS/dt = -beta * S * I / N
       dI/dt = beta * S * I / N - gamma * I
       dR/dt = gamma * I
   Simulate an outbreak and identify the peak infection time. Compute R0 and
   the final epidemic size.

2. Implement the Lotka-Volterra predator-prey model with harvesting:
       dx/dt = alpha * x - beta * x * y - h_x
       dy/dt = delta * x * y - gamma * y - h_y
   where h_x, h_y are constant harvesting rates. Analyze how harvesting
   changes the equilibrium and can lead to population collapse.

3. Implement an age-structured population model using the Leslie matrix:
       n(t+1) = L * n(t)
   where n is the age-distribution vector and L is the Leslie matrix with
   fertilities on the first row and survival probabilities on the sub-diagonal.
   Compute the dominant eigenvalue (growth rate) and stable age distribution.

4. Perform parameter sensitivity analysis on the SIR model: how do changes
   in beta and gamma affect peak infection and final epidemic size?

5. Construct a bifurcation diagram for the Lotka-Volterra model: vary the
   harvesting rate and plot the long-term population levels, identifying the
   critical harvesting rate at which a population goes extinct.
"""

import numpy as np
import matplotlib.pyplot as plt


def sir_model(t, y, beta, gamma, N):
    """
    Right-hand side of the SIR ODE system.

    Parameters
    ----------
    t : float
        Current time.
    y : np.ndarray
        State vector [S, I, R].
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    N : float
        Total population.

    Returns
    -------
    np.ndarray
        Derivatives [dS/dt, dI/dt, dR/dt].
    """
    raise NotImplementedError


def simulate_sir(beta, gamma, N, I0, t_span, num_points=1000):
    """
    Simulate the SIR epidemic model.

    Parameters
    ----------
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    N : float
        Total population.
    I0 : float
        Initial number of infected individuals.
    t_span : tuple
        (t_start, t_end) time interval.
    num_points : int
        Number of output time points.

    Returns
    -------
    t : np.ndarray
        Time points.
    S : np.ndarray
        Susceptible population over time.
    I : np.ndarray
        Infected population over time.
    R : np.ndarray
        Recovered population over time.
    """
    raise NotImplementedError


def lotka_volterra_harvesting(t, y, alpha, beta, delta, gamma, h_x=0.0, h_y=0.0):
    """
    Lotka-Volterra predator-prey model with constant harvesting.

    Parameters
    ----------
    t : float
        Current time.
    y : np.ndarray
        State vector [prey, predator].
    alpha : float
        Prey growth rate.
    beta : float
        Predation rate.
    delta : float
        Predator growth efficiency.
    gamma : float
        Predator death rate.
    h_x : float
        Prey harvesting rate.
    h_y : float
        Predator harvesting rate.

    Returns
    -------
    np.ndarray
        Derivatives [dx/dt, dy/dt].
    """
    raise NotImplementedError


def simulate_lotka_volterra(alpha, beta, delta, gamma, x0, y0, t_span,
                            h_x=0.0, h_y=0.0, num_points=2000):
    """
    Simulate the Lotka-Volterra model with optional harvesting.

    Parameters
    ----------
    alpha, beta, delta, gamma : float
        Model parameters.
    x0 : float
        Initial prey population.
    y0 : float
        Initial predator population.
    t_span : tuple
        Time interval.
    h_x : float
        Prey harvesting rate.
    h_y : float
        Predator harvesting rate.
    num_points : int
        Number of output time points.

    Returns
    -------
    t : np.ndarray
        Time points.
    prey : np.ndarray
        Prey population.
    predator : np.ndarray
        Predator population.
    """
    raise NotImplementedError


def leslie_matrix_model(L, n0, num_steps):
    """
    Simulate age-structured population dynamics using Leslie matrix.

    Parameters
    ----------
    L : np.ndarray
        Leslie matrix of shape (k, k).
    n0 : np.ndarray
        Initial age distribution of shape (k,).
    num_steps : int
        Number of time steps.

    Returns
    -------
    populations : np.ndarray
        Population vectors at each step, shape (num_steps + 1, k).
    total_pop : np.ndarray
        Total population at each step, shape (num_steps + 1,).
    growth_rate : float
        Dominant eigenvalue of L (asymptotic growth rate).
    stable_dist : np.ndarray
        Stable age distribution (normalized dominant eigenvector).
    """
    raise NotImplementedError


def sir_sensitivity(beta_range, gamma_range, N, I0, t_span):
    """
    Sensitivity analysis: peak infection and final size as functions of beta, gamma.

    Parameters
    ----------
    beta_range : np.ndarray
        Array of beta values to test.
    gamma_range : np.ndarray
        Array of gamma values to test.
    N : float
        Total population.
    I0 : float
        Initial infected.
    t_span : tuple
        Time interval.

    Returns
    -------
    peak_infections : np.ndarray
        Peak infected count for each (beta, gamma), shape (len(beta_range), len(gamma_range)).
    final_sizes : np.ndarray
        Final epidemic size for each (beta, gamma).
    """
    raise NotImplementedError


def bifurcation_diagram(alpha, beta, delta, gamma, x0, y0,
                        h_range, t_span, t_transient=100):
    """
    Compute bifurcation diagram: long-term prey population vs harvesting rate.

    Parameters
    ----------
    alpha, beta, delta, gamma : float
        Lotka-Volterra parameters.
    x0, y0 : float
        Initial conditions.
    h_range : np.ndarray
        Array of harvesting rates to test.
    t_span : tuple
        Total simulation time.
    t_transient : float
        Time to discard as transient before recording long-term behavior.

    Returns
    -------
    h_values : np.ndarray
        Harvesting rates.
    prey_long_term : list of np.ndarray
        Long-term prey values (local maxima/minima) for each h.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Task 1: SIR model
    N = 10000
    beta, gamma = 0.3, 0.1
    R0 = beta / gamma
    print(f"SIR Model: R0 = {R0:.2f}")

    t, S, I, R = simulate_sir(beta, gamma, N, I0=10, t_span=(0, 200))
    peak_idx = np.argmax(I)
    print(f"Peak infection: {I[peak_idx]:.0f} at t = {t[peak_idx]:.1f}")
    print(f"Final epidemic size: {R[-1]:.0f} ({100 * R[-1] / N:.1f}%)")

    # Task 2: Lotka-Volterra with harvesting
    alpha, beta_lv, delta, gamma_lv = 1.0, 0.1, 0.02, 0.5
    t_lv, prey, pred = simulate_lotka_volterra(
        alpha, beta_lv, delta, gamma_lv, x0=40, y0=9, t_span=(0, 100)
    )
    print(f"\nLotka-Volterra (no harvesting): prey range [{prey.min():.1f}, {prey.max():.1f}]")

    t_lv, prey_h, pred_h = simulate_lotka_volterra(
        alpha, beta_lv, delta, gamma_lv, x0=40, y0=9, t_span=(0, 100), h_x=2.0
    )
    print(f"With harvesting h_x=2.0: prey range [{prey_h.min():.1f}, {prey_h.max():.1f}]")

    # Task 3: Leslie matrix
    L = np.array([
        [0.0, 1.5, 0.8, 0.0],
        [0.6, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
    ])
    n0 = np.array([100, 50, 30, 10])
    populations, total_pop, growth, stable = leslie_matrix_model(L, n0, num_steps=50)
    print(f"\nLeslie model: growth rate = {growth:.4f}")
    print(f"Stable age distribution: {stable}")

    # Task 4: SIR sensitivity
    beta_range = np.linspace(0.1, 0.5, 20)
    gamma_range = np.linspace(0.05, 0.3, 20)
    peaks, finals = sir_sensitivity(beta_range, gamma_range, N, I0=10, t_span=(0, 300))
    print(f"\nSensitivity: peak infection range [{peaks.min():.0f}, {peaks.max():.0f}]")

    # Task 5: Bifurcation diagram
    h_range = np.linspace(0, 5, 100)
    h_vals, prey_lt = bifurcation_diagram(
        alpha, beta_lv, delta, gamma_lv, 40, 9, h_range, t_span=(0, 300)
    )
    print(f"\nBifurcation diagram computed for {len(h_range)} harvesting rates.")
