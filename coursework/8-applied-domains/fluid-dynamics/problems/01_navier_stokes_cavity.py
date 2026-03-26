"""
2D Lid-Driven Cavity Flow (Navier-Stokes)
==========================================

The lid-driven cavity is a benchmark problem in computational fluid dynamics.
A square cavity [0,1]x[0,1] has three stationary walls (no-slip) and a top
lid moving at constant velocity u = 1 in the x-direction.

The incompressible Navier-Stokes equations in 2D:
    du/dt + u du/dx + v du/dy = -1/rho dp/dx + nu (d^2u/dx^2 + d^2u/dy^2)
    dv/dt + u dv/dx + v dv/dy = -1/rho dp/dy + nu (d^2v/dx^2 + d^2v/dy^2)
    du/dx + dv/dy = 0   (continuity)

where nu = 1/Re is the kinematic viscosity.

Tasks
-----
1. Implement the 2D lid-driven cavity flow using a finite difference method.
   Use a staggered grid or collocated grid with appropriate pressure-velocity
   coupling (e.g., a simplified SIMPLE-like pressure correction or the
   stream function-vorticity formulation).

2. Solve the steady-state problem iteratively. Implement the pressure Poisson
   equation and iterate until velocity and pressure fields converge.

3. Visualize the velocity field (quiver plot) and streamlines. Compare the
   center of the primary vortex location with reference data.

4. Study the effect of Reynolds number (Re = 100, 400, 1000). Show how the
   flow pattern changes: at low Re, a single symmetric vortex; at higher Re,
   corner vortices appear and the primary vortex shifts.
"""

import numpy as np
import matplotlib.pyplot as plt


def initialize_fields(nx, ny):
    """
    Initialize velocity and pressure fields on the computational grid.

    Parameters
    ----------
    nx : int
        Number of grid points in x-direction.
    ny : int
        Number of grid points in y-direction.

    Returns
    -------
    u : np.ndarray
        x-velocity field of shape (ny, nx).
    v : np.ndarray
        y-velocity field of shape (ny, nx).
    p : np.ndarray
        Pressure field of shape (ny, nx).
    """
    raise NotImplementedError


def apply_boundary_conditions(u, v, p, u_lid=1.0):
    """
    Apply boundary conditions for the lid-driven cavity.

    Parameters
    ----------
    u : np.ndarray
        x-velocity field.
    v : np.ndarray
        y-velocity field.
    p : np.ndarray
        Pressure field.
    u_lid : float
        Lid velocity (top wall, x-direction).

    Returns
    -------
    u, v, p : np.ndarray
        Fields with boundary conditions applied.
    """
    raise NotImplementedError


def solve_pressure_poisson(p, u, v, dx, dy, dt, rho, nit=50):
    """
    Solve the pressure Poisson equation for pressure correction.

    Parameters
    ----------
    p : np.ndarray
        Current pressure field.
    u : np.ndarray
        x-velocity field.
    v : np.ndarray
        y-velocity field.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step.
    rho : float
        Fluid density.
    nit : int
        Number of pseudo-time iterations for pressure.

    Returns
    -------
    p : np.ndarray
        Updated pressure field.
    """
    raise NotImplementedError


def velocity_update(u, v, p, dx, dy, dt, rho, nu):
    """
    Update velocity field using momentum equations with pressure gradient.

    Parameters
    ----------
    u : np.ndarray
        Current x-velocity.
    v : np.ndarray
        Current y-velocity.
    p : np.ndarray
        Pressure field.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step.
    rho : float
        Fluid density.
    nu : float
        Kinematic viscosity (1/Re).

    Returns
    -------
    u_new : np.ndarray
        Updated x-velocity.
    v_new : np.ndarray
        Updated y-velocity.
    """
    raise NotImplementedError


def solve_cavity_flow(nx, ny, Re, nt=5000, dt=0.001):
    """
    Solve the lid-driven cavity flow problem.

    Parameters
    ----------
    nx : int
        Grid points in x.
    ny : int
        Grid points in y.
    Re : float
        Reynolds number.
    nt : int
        Number of time steps.
    dt : float
        Time step size.

    Returns
    -------
    u : np.ndarray
        Converged x-velocity field.
    v : np.ndarray
        Converged y-velocity field.
    p : np.ndarray
        Converged pressure field.
    x : np.ndarray
        x-coordinates of grid.
    y : np.ndarray
        y-coordinates of grid.
    """
    raise NotImplementedError


def find_vortex_center(u, v, x, y):
    """
    Locate the center of the primary vortex (minimum stream function).

    Parameters
    ----------
    u : np.ndarray
        x-velocity field.
    v : np.ndarray
        y-velocity field.
    x : np.ndarray
        x-coordinates.
    y : np.ndarray
        y-coordinates.

    Returns
    -------
    x_center : float
        x-coordinate of vortex center.
    y_center : float
        y-coordinate of vortex center.
    """
    raise NotImplementedError


def plot_flow_field(u, v, p, x, y, Re):
    """
    Visualize velocity field (quiver plot) and streamlines.

    Parameters
    ----------
    u : np.ndarray
        x-velocity field.
    v : np.ndarray
        y-velocity field.
    p : np.ndarray
        Pressure field.
    x : np.ndarray
        x-coordinates.
    y : np.ndarray
        y-coordinates.
    Re : float
        Reynolds number (for title).
    """
    raise NotImplementedError


if __name__ == "__main__":
    nx, ny = 41, 41

    # Task 1 & 2: Solve for Re = 100
    u, v, p, x, y = solve_cavity_flow(nx, ny, Re=100, nt=5000)

    # Task 3: Visualize and find vortex center
    xc, yc = find_vortex_center(u, v, x, y)
    print(f"Re=100: Primary vortex center at ({xc:.3f}, {yc:.3f})")
    print("  Reference (Ghia et al.): approximately (0.6172, 0.7344)")

    plot_flow_field(u, v, p, x, y, Re=100)
    plt.savefig("cavity_Re100.png", dpi=150, bbox_inches="tight")

    # Task 4: Reynolds number study
    for Re in [100, 400, 1000]:
        u, v, p, x, y = solve_cavity_flow(nx, ny, Re=Re, nt=10000)
        xc, yc = find_vortex_center(u, v, x, y)
        print(f"Re={Re}: Vortex center at ({xc:.3f}, {yc:.3f})")
        plot_flow_field(u, v, p, x, y, Re=Re)
        plt.savefig(f"cavity_Re{Re}.png", dpi=150, bbox_inches="tight")
