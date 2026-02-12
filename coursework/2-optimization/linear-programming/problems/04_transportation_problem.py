"""
Problem 4: The Transportation Problem

A company has 3 warehouses (supply) and 4 stores (demand).
Shipping costs per unit, supplies, and demands are given below.

    Cost matrix:
              Store1  Store2  Store3  Store4
    Warehouse1:  8      6      10      9
    Warehouse2:  9      12      7      5
    Warehouse3:  4      8       11     6

    Supply:  [35, 50, 40]
    Demand:  [30, 25, 35, 35]

    Note: total supply (125) == total demand (125), so the problem is balanced.

Tasks:
  (a) Formulate as a linear program (decision variables x_ij = units shipped)
  (b) Find an initial basic feasible solution using the Northwest Corner method
  (c) Solve optimally using scipy.optimize.linprog
  (d) Report the optimal shipping plan and total cost
"""

import numpy as np


COST = np.array([
    [8, 6, 10, 9],
    [9, 12, 7, 5],
    [4, 8, 11, 6],
])
SUPPLY = np.array([35, 50, 40])
DEMAND = np.array([30, 25, 35, 35])


def northwest_corner(supply, demand):
    """
    Return an initial BFS allocation matrix using the Northwest Corner method.
    The result should be a 2D array of shape (len(supply), len(demand)).
    """
    # TODO
    raise NotImplementedError


def compute_cost(allocation, cost):
    """Return the total shipping cost for a given allocation."""
    return np.sum(allocation * cost)


def solve_transportation(cost, supply, demand):
    """
    Solve the transportation problem as an LP using scipy.optimize.linprog.
    Return (optimal_cost, allocation_matrix).

    Hint: flatten x_ij into a 1D vector. Equality constraints enforce
    row sums == supply and column sums == demand.
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    try:
        nw = northwest_corner(SUPPLY.copy(), DEMAND.copy())
        print("Northwest Corner allocation:\n", nw)
        print("NW Corner cost:", compute_cost(nw, COST))
    except NotImplementedError:
        print("TODO: implement northwest_corner()")

    try:
        opt_cost, opt_alloc = solve_transportation(COST, SUPPLY, DEMAND)
        print(f"\nOptimal cost: {opt_cost}")
        print("Optimal allocation:\n", opt_alloc)
    except NotImplementedError:
        print("TODO: implement solve_transportation()")
