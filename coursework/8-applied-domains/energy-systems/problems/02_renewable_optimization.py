"""
Renewable Energy System Optimization
======================================

Optimizing the mix of renewable energy sources (solar, wind) and energy
storage to reliably meet electricity demand is a key challenge in energy
systems planning. This problem formulates and solves a simplified version
of this optimization.

Tasks
-----
1. Optimize the mix of solar panels, wind turbines, and battery storage to
   meet a 24-hour demand profile at minimum cost. Decision variables:
   - solar_capacity (MW): installed solar PV capacity
   - wind_capacity (MW): installed wind capacity
   - battery_capacity (MWh): installed battery storage capacity

2. Implement a Linear Programming (LP) formulation for economic dispatch:
   Given installed capacities, minimize operating cost over 24 hours subject
   to demand satisfaction at each hour. Use a simple LP solver (e.g., the
   revised simplex method or use numpy to solve the KKT conditions).

3. Simulate battery charge/discharge over 24 hours:
   - Battery charges when renewable generation exceeds demand.
   - Battery discharges when demand exceeds generation.
   - Enforce battery constraints: 0 <= SOC <= capacity, max charge/discharge rate.

4. Compute the Levelized Cost of Energy (LCOE) for each configuration:
       LCOE = (Total annualized cost) / (Total annual energy served)
   Compare LCOE for different mixes (solar-heavy, wind-heavy, balanced).
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_demand_profile(peak_demand=100.0):
    """
    Generate a typical 24-hour electricity demand profile.

    Parameters
    ----------
    peak_demand : float
        Peak demand in MW.

    Returns
    -------
    hours : np.ndarray
        Hour indices (0 to 23).
    demand : np.ndarray
        Demand at each hour in MW.
    """
    raise NotImplementedError


def solar_generation_profile(capacity, hours):
    """
    Generate hourly solar power output based on a typical daily irradiance curve.

    Parameters
    ----------
    capacity : float
        Installed solar capacity in MW (peak).
    hours : np.ndarray
        Hour indices (0 to 23).

    Returns
    -------
    generation : np.ndarray
        Solar power output at each hour in MW.
    """
    raise NotImplementedError


def wind_generation_profile(capacity, hours, seed=42):
    """
    Generate hourly wind power output based on a stochastic wind model.

    Parameters
    ----------
    capacity : float
        Installed wind capacity in MW.
    hours : np.ndarray
        Hour indices (0 to 23).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    generation : np.ndarray
        Wind power output at each hour in MW.
    """
    raise NotImplementedError


def simulate_battery(generation, demand, battery_capacity, max_rate=None,
                     initial_soc=0.5):
    """
    Simulate battery charge/discharge over 24 hours.

    Parameters
    ----------
    generation : np.ndarray
        Total renewable generation at each hour (MW).
    demand : np.ndarray
        Demand at each hour (MW).
    battery_capacity : float
        Battery energy capacity (MWh).
    max_rate : float or None
        Maximum charge/discharge rate (MW). If None, defaults to battery_capacity / 4.
    initial_soc : float
        Initial state of charge as fraction of capacity (0 to 1).

    Returns
    -------
    soc : np.ndarray
        State of charge at each hour (MWh), shape (25,) including initial.
    charge : np.ndarray
        Charging power at each hour (MW), shape (24,).
    discharge : np.ndarray
        Discharging power at each hour (MW), shape (24,).
    unmet_demand : np.ndarray
        Unmet demand at each hour (MW), shape (24,).
    curtailment : np.ndarray
        Curtailed generation at each hour (MW), shape (24,).
    """
    raise NotImplementedError


def economic_dispatch_lp(solar_gen, wind_gen, demand, battery_capacity,
                         max_rate, cost_solar=0.0, cost_wind=0.0,
                         cost_battery_cycle=5.0, cost_unmet=200.0):
    """
    Solve economic dispatch as a linear program for 24 hours.

    Minimize total cost = sum over hours of:
        cost_battery_cycle * discharge[t] + cost_unmet * unmet[t]

    Subject to:
        solar_gen[t] + wind_gen[t] + discharge[t] - charge[t] + unmet[t] >= demand[t]
        0 <= soc[t] <= battery_capacity
        0 <= charge[t] <= max_rate
        0 <= discharge[t] <= max_rate

    Parameters
    ----------
    solar_gen : np.ndarray
        Solar generation profile (MW), shape (24,).
    wind_gen : np.ndarray
        Wind generation profile (MW), shape (24,).
    demand : np.ndarray
        Demand profile (MW), shape (24,).
    battery_capacity : float
        Battery capacity (MWh).
    max_rate : float
        Max charge/discharge rate (MW).
    cost_solar : float
        Marginal cost of solar ($/MWh).
    cost_wind : float
        Marginal cost of wind ($/MWh).
    cost_battery_cycle : float
        Cost per MWh of battery cycling ($/MWh).
    cost_unmet : float
        Penalty for unmet demand ($/MWh).

    Returns
    -------
    total_cost : float
        Optimal total operating cost ($).
    dispatch : dict
        Optimal dispatch with keys 'charge', 'discharge', 'unmet', 'soc'.
    """
    raise NotImplementedError


def compute_lcoe(solar_capacity, wind_capacity, battery_capacity,
                 annual_generation, annual_demand_served,
                 solar_capex=1000, wind_capex=1500, battery_capex=300,
                 lifetime=25, discount_rate=0.05):
    """
    Compute Levelized Cost of Energy.

    LCOE = (Annualized capital cost + annual O&M) / Annual energy served

    Parameters
    ----------
    solar_capacity : float
        Installed solar (MW).
    wind_capacity : float
        Installed wind (MW).
    battery_capacity : float
        Installed battery (MWh).
    annual_generation : float
        Total annual generation (MWh).
    annual_demand_served : float
        Total annual demand served (MWh).
    solar_capex : float
        Solar capital cost ($/kW).
    wind_capex : float
        Wind capital cost ($/kW).
    battery_capex : float
        Battery capital cost ($/kWh).
    lifetime : int
        Project lifetime (years).
    discount_rate : float
        Discount rate for annualization.

    Returns
    -------
    lcoe : float
        Levelized cost of energy ($/MWh).
    """
    raise NotImplementedError


def optimize_capacity_mix(demand, budget=None, target_reliability=0.95):
    """
    Find the optimal solar/wind/battery capacity mix.

    Parameters
    ----------
    demand : np.ndarray
        24-hour demand profile (MW).
    budget : float or None
        Total capital budget constraint ($).
    target_reliability : float
        Fraction of demand that must be met (0 to 1).

    Returns
    -------
    optimal : dict
        Dictionary with 'solar_capacity', 'wind_capacity', 'battery_capacity',
        'lcoe', 'reliability'.
    """
    raise NotImplementedError


def plot_dispatch(hours, demand, solar_gen, wind_gen, soc, unmet):
    """
    Plot the 24-hour dispatch including generation, demand, and battery SOC.

    Parameters
    ----------
    hours : np.ndarray
        Hour indices.
    demand : np.ndarray
        Demand profile.
    solar_gen : np.ndarray
        Solar generation.
    wind_gen : np.ndarray
        Wind generation.
    soc : np.ndarray
        Battery state of charge.
    unmet : np.ndarray
        Unmet demand.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Generate profiles
    hours, demand = generate_demand_profile(peak_demand=100.0)
    solar_gen = solar_generation_profile(capacity=80.0, hours=hours)
    wind_gen = wind_generation_profile(capacity=60.0, hours=hours)

    total_gen = solar_gen + wind_gen
    print(f"Peak demand: {demand.max():.1f} MW")
    print(f"Total daily demand: {demand.sum():.1f} MWh")
    print(f"Total solar generation: {solar_gen.sum():.1f} MWh")
    print(f"Total wind generation: {wind_gen.sum():.1f} MWh")

    # Task 3: Battery simulation
    battery_cap = 100.0  # MWh
    soc, charge, discharge, unmet, curtail = simulate_battery(
        total_gen, demand, battery_cap
    )
    print(f"\nBattery simulation (capacity={battery_cap} MWh):")
    print(f"  Total unmet demand: {unmet.sum():.1f} MWh")
    print(f"  Total curtailment: {curtail.sum():.1f} MWh")
    print(f"  Reliability: {1 - unmet.sum() / demand.sum():.1%}")

    # Task 2: Economic dispatch
    total_cost, dispatch = economic_dispatch_lp(
        solar_gen, wind_gen, demand, battery_cap, max_rate=25.0
    )
    print(f"\nEconomic dispatch total cost: ${total_cost:.2f}")

    # Task 4: LCOE comparison
    print("\nLCOE Comparison:")
    configs = [
        ("Solar-heavy", 120, 30, 80),
        ("Wind-heavy", 30, 120, 80),
        ("Balanced", 70, 70, 100),
    ]
    for name, s_cap, w_cap, b_cap in configs:
        s_gen = solar_generation_profile(s_cap, hours)
        w_gen = wind_generation_profile(w_cap, hours)
        total = s_gen + w_gen
        _, _, _, um, _ = simulate_battery(total, demand, b_cap)
        served = demand.sum() - um.sum()
        annual_served = served * 365
        annual_gen = total.sum() * 365
        lcoe = compute_lcoe(s_cap, w_cap, b_cap, annual_gen, annual_served)
        reliability = 1 - um.sum() / demand.sum()
        print(f"  {name}: LCOE=${lcoe:.2f}/MWh, Reliability={reliability:.1%}")

    # Plot
    plot_dispatch(hours, demand, solar_gen, wind_gen, soc, unmet)
    plt.savefig("renewable_dispatch.png", dpi=150, bbox_inches="tight")
    print("\nDispatch plot saved.")
