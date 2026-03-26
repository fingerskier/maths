"""
1D Shallow Water Equations (Saint-Venant)
==========================================

The shallow water equations model free-surface flow where the horizontal
length scale is much larger than the depth. The 1D conservative form:

    dU/dt + dF(U)/dx = 0

where U = [h, h*u]^T is the vector of conserved variables (water height h
and momentum h*u), and F(U) = [h*u, h*u^2 + 0.5*g*h^2]^T is the flux.

Tasks
-----
1. Implement the 1D shallow water equations using a finite volume method.
   Discretize the domain into cells and evolve cell averages in time.

2. Use the Lax-Friedrichs numerical flux:
       F_{i+1/2} = 0.5 * (F(U_L) + F(U_R)) - 0.5 * alpha * (U_R - U_L)
   where alpha = max|eigenvalue| = max(|u| + sqrt(g*h)) is the maximum
   wave speed, providing numerical dissipation for stability.

3. Simulate a dam break problem: initially h = h_L for x < x_dam, h = h_R
   for x > x_dam, u = 0 everywhere. Compare the numerical solution with the
   Ritter analytical solution for the dam break on a dry bed.

4. Analyze wave propagation: measure the speed of the shock wave and
   rarefaction wave, and compare with theoretical predictions. Study the
   effect of grid resolution on solution quality.
"""

import numpy as np
import matplotlib.pyplot as plt


def initial_condition_dam_break(x, x_dam=5.0, h_left=2.0, h_right=0.5):
    """
    Set up dam break initial conditions.

    Parameters
    ----------
    x : np.ndarray
        Cell center coordinates.
    x_dam : float
        Position of the dam.
    h_left : float
        Water height on the left of the dam.
    h_right : float
        Water height on the right of the dam.

    Returns
    -------
    h : np.ndarray
        Initial water height.
    hu : np.ndarray
        Initial momentum (zero everywhere).
    """
    raise NotImplementedError


def compute_flux(h, hu, g=9.81):
    """
    Compute the physical flux F(U) for the shallow water equations.

    Parameters
    ----------
    h : np.ndarray
        Water height.
    hu : np.ndarray
        Momentum.
    g : float
        Gravitational acceleration.

    Returns
    -------
    f_h : np.ndarray
        Mass flux (= hu).
    f_hu : np.ndarray
        Momentum flux (= hu^2/h + 0.5*g*h^2).
    """
    raise NotImplementedError


def lax_friedrichs_flux(h_L, hu_L, h_R, hu_R, g=9.81):
    """
    Compute Lax-Friedrichs numerical flux at a cell interface.

    Parameters
    ----------
    h_L : float or np.ndarray
        Water height on the left side.
    hu_L : float or np.ndarray
        Momentum on the left side.
    h_R : float or np.ndarray
        Water height on the right side.
    hu_R : float or np.ndarray
        Momentum on the right side.
    g : float
        Gravitational acceleration.

    Returns
    -------
    F_h : float or np.ndarray
        Numerical mass flux.
    F_hu : float or np.ndarray
        Numerical momentum flux.
    """
    raise NotImplementedError


def compute_max_wavespeed(h, hu, g=9.81):
    """
    Compute the maximum wave speed for CFL condition.

    Parameters
    ----------
    h : np.ndarray
        Water height.
    hu : np.ndarray
        Momentum.
    g : float
        Gravitational acceleration.

    Returns
    -------
    float
        Maximum wave speed max(|u| + sqrt(g*h)).
    """
    raise NotImplementedError


def evolve_step(h, hu, dx, dt, g=9.81):
    """
    Advance the solution by one time step using finite volume + Lax-Friedrichs.

    Parameters
    ----------
    h : np.ndarray
        Current water height in each cell.
    hu : np.ndarray
        Current momentum in each cell.
    dx : float
        Cell width.
    dt : float
        Time step.
    g : float
        Gravitational acceleration.

    Returns
    -------
    h_new : np.ndarray
        Updated water height.
    hu_new : np.ndarray
        Updated momentum.
    """
    raise NotImplementedError


def solve_shallow_water(x, h0, hu0, t_final, cfl=0.5, g=9.81):
    """
    Solve the 1D shallow water equations to a given final time.

    Parameters
    ----------
    x : np.ndarray
        Cell center coordinates.
    h0 : np.ndarray
        Initial water height.
    hu0 : np.ndarray
        Initial momentum.
    t_final : float
        Final simulation time.
    cfl : float
        CFL number for time step control.
    g : float
        Gravitational acceleration.

    Returns
    -------
    h : np.ndarray
        Water height at t_final.
    hu : np.ndarray
        Momentum at t_final.
    """
    raise NotImplementedError


def ritter_solution(x, t, x_dam=5.0, h_left=2.0, g=9.81):
    """
    Analytical Ritter solution for dam break on dry bed (h_right = 0).

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    t : float
        Time.
    x_dam : float
        Initial dam position.
    h_left : float
        Initial water height on the left.
    g : float
        Gravitational acceleration.

    Returns
    -------
    h_exact : np.ndarray
        Exact water height.
    u_exact : np.ndarray
        Exact velocity.
    """
    raise NotImplementedError


def plot_solution(x, h, hu, t, h_exact=None):
    """
    Plot water height and velocity profiles.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    h : np.ndarray
        Numerical water height.
    hu : np.ndarray
        Numerical momentum.
    t : float
        Current time.
    h_exact : np.ndarray or None
        Exact solution for comparison.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Task 1 & 2: Set up and solve dam break
    nx = 200
    x_min, x_max = 0.0, 10.0
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

    h0, hu0 = initial_condition_dam_break(x)
    t_final = 0.5
    h, hu = solve_shallow_water(x, h0, hu0, t_final)

    # Task 3: Compare with Ritter solution (dry bed case)
    h0_dry, hu0_dry = initial_condition_dam_break(x, h_right=0.0)
    h_dry, hu_dry = solve_shallow_water(x, h0_dry, hu0_dry, t_final)
    h_exact, u_exact = ritter_solution(x, t_final)

    rmse = np.sqrt(np.mean((h_dry - h_exact) ** 2))
    print(f"RMSE vs Ritter solution (nx={nx}): {rmse:.6f}")

    plot_solution(x, h_dry, hu_dry, t_final, h_exact)
    plt.savefig("dam_break.png", dpi=150, bbox_inches="tight")

    # Task 4: Grid resolution study
    print("\nGrid convergence study:")
    for nx_test in [50, 100, 200, 400, 800]:
        dx_t = (x_max - x_min) / nx_test
        x_t = np.linspace(x_min + dx_t / 2, x_max - dx_t / 2, nx_test)
        h0_t, hu0_t = initial_condition_dam_break(x_t, h_right=0.0)
        h_t, _ = solve_shallow_water(x_t, h0_t, hu0_t, t_final)
        h_ex_t, _ = ritter_solution(x_t, t_final)
        rmse_t = np.sqrt(np.mean((h_t - h_ex_t) ** 2))
        print(f"  nx={nx_test:4d}: RMSE = {rmse_t:.6f}")
