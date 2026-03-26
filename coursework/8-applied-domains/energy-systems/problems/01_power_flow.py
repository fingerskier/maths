"""
Power Flow Analysis
====================

Power flow (load flow) analysis determines the steady-state operating
condition of an electric power network: voltage magnitudes and angles at
each bus, and real/reactive power flows on each line.

Tasks
-----
1. Implement the DC power flow approximation for a small network (3-5 buses).
   DC power flow assumes:
   - All voltage magnitudes are 1.0 p.u.
   - Angle differences are small (sin(theta) ~ theta).
   - Reactive power and line resistance are neglected.
   This gives a linear system: P = B' * theta, where B' is the susceptance
   matrix.

2. Implement Newton-Raphson iteration for AC power flow on a 2-3 bus system.
   The AC power flow equations are:
       P_i = sum_j |V_i| |V_j| (G_ij cos(theta_ij) + B_ij sin(theta_ij))
       Q_i = sum_j |V_i| |V_j| (G_ij sin(theta_ij) - B_ij cos(theta_ij))
   Iterate until the power mismatch is below tolerance.

3. Compute line flows and losses from the converged power flow solution.
   Line power flow: S_ij = V_i * (V_i - V_j)* * y_ij* and losses = S_ij + S_ji.

4. Analyze the effect of load changes: increase load at one bus and observe
   how voltages, angles, and line flows change. Identify when the system
   approaches its transfer limit.
"""

import numpy as np


def build_ybus(bus_data, line_data):
    """
    Build the bus admittance matrix (Y_bus) from network data.

    Parameters
    ----------
    bus_data : dict
        Bus information with keys 'num_buses', 'types' (list: 'slack', 'PV', 'PQ'),
        'P_spec', 'Q_spec', 'V_spec'.
    line_data : list of dict
        Each dict has 'from', 'to', 'r' (resistance), 'x' (reactance).

    Returns
    -------
    Y_bus : np.ndarray
        Complex admittance matrix of shape (n, n).
    """
    raise NotImplementedError


def dc_power_flow(B_prime, P_injected, slack_bus=0):
    """
    Solve DC power flow: P = B' * theta.

    Parameters
    ----------
    B_prime : np.ndarray
        Susceptance matrix (imaginary part of Y_bus, negated) with slack
        bus row/column removed.
    P_injected : np.ndarray
        Net power injection at each non-slack bus.
    slack_bus : int
        Index of the slack bus (angle = 0).

    Returns
    -------
    theta : np.ndarray
        Voltage angles at all buses (slack bus angle = 0).
    """
    raise NotImplementedError


def compute_power_mismatch(V, theta, Y_bus, P_spec, Q_spec, pq_buses, pv_buses):
    """
    Compute active and reactive power mismatches for Newton-Raphson.

    Parameters
    ----------
    V : np.ndarray
        Voltage magnitudes at all buses.
    theta : np.ndarray
        Voltage angles at all buses.
    Y_bus : np.ndarray
        Bus admittance matrix.
    P_spec : np.ndarray
        Specified active power at each bus.
    Q_spec : np.ndarray
        Specified reactive power at PQ buses.
    pq_buses : list of int
        Indices of PQ buses.
    pv_buses : list of int
        Indices of PV buses.

    Returns
    -------
    mismatch : np.ndarray
        Vector of [dP; dQ] mismatches.
    """
    raise NotImplementedError


def build_jacobian(V, theta, Y_bus, pq_buses, pv_buses):
    """
    Build the Jacobian matrix for Newton-Raphson power flow.

    Parameters
    ----------
    V : np.ndarray
        Voltage magnitudes.
    theta : np.ndarray
        Voltage angles.
    Y_bus : np.ndarray
        Bus admittance matrix.
    pq_buses : list of int
        PQ bus indices.
    pv_buses : list of int
        PV bus indices.

    Returns
    -------
    J : np.ndarray
        Jacobian matrix partitioned as [[J1, J2], [J3, J4]].
    """
    raise NotImplementedError


def newton_raphson_power_flow(Y_bus, bus_data, tol=1e-6, max_iter=20):
    """
    Solve AC power flow using Newton-Raphson method.

    Parameters
    ----------
    Y_bus : np.ndarray
        Bus admittance matrix.
    bus_data : dict
        Bus information including types, specified P, Q, V.
    tol : float
        Convergence tolerance on mismatch norm.
    max_iter : int
        Maximum number of NR iterations.

    Returns
    -------
    V : np.ndarray
        Converged voltage magnitudes.
    theta : np.ndarray
        Converged voltage angles (radians).
    converged : bool
        Whether the method converged.
    num_iter : int
        Number of iterations performed.
    """
    raise NotImplementedError


def compute_line_flows(V, theta, line_data):
    """
    Compute power flows and losses on each transmission line.

    Parameters
    ----------
    V : np.ndarray
        Voltage magnitudes at each bus.
    theta : np.ndarray
        Voltage angles at each bus.
    line_data : list of dict
        Line parameters.

    Returns
    -------
    flows : list of dict
        Each dict has 'from', 'to', 'P_flow', 'Q_flow', 'P_loss', 'Q_loss'.
    """
    raise NotImplementedError


def load_sensitivity_analysis(Y_bus, bus_data, line_data, load_bus,
                              load_factors):
    """
    Analyze effect of varying load at one bus.

    Parameters
    ----------
    Y_bus : np.ndarray
        Bus admittance matrix.
    bus_data : dict
        Base case bus data.
    line_data : list of dict
        Line parameters.
    load_bus : int
        Index of the bus where load is varied.
    load_factors : np.ndarray
        Multipliers for the load (e.g., [0.5, 1.0, 1.5, 2.0]).

    Returns
    -------
    results : list of dict
        For each load factor: voltage magnitudes, angles, line flows, convergence.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Define a simple 3-bus system
    line_data = [
        {"from": 0, "to": 1, "r": 0.02, "x": 0.06},
        {"from": 0, "to": 2, "r": 0.08, "x": 0.24},
        {"from": 1, "to": 2, "r": 0.06, "x": 0.18},
    ]

    bus_data = {
        "num_buses": 3,
        "types": ["slack", "PV", "PQ"],
        "P_spec": np.array([0.0, 0.5, -1.0]),   # generation (+), load (-)
        "Q_spec": np.array([0.0, 0.0, -0.5]),
        "V_spec": np.array([1.05, 1.02, 1.0]),   # specified voltage magnitudes
    }

    Y_bus = build_ybus(bus_data, line_data)
    print("Y_bus matrix:")
    print(Y_bus)

    # Task 1: DC power flow
    B_prime = -np.imag(Y_bus)
    B_reduced = B_prime[1:, 1:]
    P_inj = bus_data["P_spec"][1:]
    theta_dc = dc_power_flow(B_reduced, P_inj)
    theta_full = np.concatenate([[0.0], theta_dc])
    print(f"\nDC Power Flow angles (deg): {np.degrees(theta_full)}")

    # Task 2: AC power flow (Newton-Raphson)
    V, theta, converged, n_iter = newton_raphson_power_flow(Y_bus, bus_data)
    print(f"\nAC Power Flow ({'converged' if converged else 'NOT converged'} "
          f"in {n_iter} iterations):")
    print(f"  Voltages (p.u.): {V}")
    print(f"  Angles (deg):    {np.degrees(theta)}")

    # Task 3: Line flows
    flows = compute_line_flows(V, theta, line_data)
    print("\nLine flows:")
    for f in flows:
        print(f"  {f['from']}->{f['to']}: P={f['P_flow']:.4f}, "
              f"Q={f['Q_flow']:.4f}, Loss={f['P_loss']:.4f}")

    # Task 4: Load sensitivity
    load_factors = np.linspace(0.5, 2.5, 9)
    results = load_sensitivity_analysis(Y_bus, bus_data, line_data,
                                        load_bus=2, load_factors=load_factors)
    print("\nLoad sensitivity at bus 2:")
    for lf, res in zip(load_factors, results):
        status = "OK" if res["convergence"] else "FAIL"
        print(f"  Load factor {lf:.1f}: V2={res.get('V', [0, 0, 0])[2]:.4f} [{status}]")
