"""
Problem 5: The Diet Problem

A classic LP: find the minimum-cost diet that meets nutritional requirements.

Foods and their data (per serving):
    Food       Cost($)  Calories  Protein(g)  Fat(g)  Calcium(mg)
    Oatmeal     0.30      110        4          2        25
    Chicken     2.50      205       32          5        12
    Eggs        0.80      160       13         10        55
    Milk        0.60      160        8          6       300
    Beans       0.40      260       14          1        80
    Spinach     1.10       40        5          0       100

Minimum daily requirements:
    Calories >= 2000
    Protein  >= 55 g
    Fat      >= 30 g  (and Fat <= 60 g)
    Calcium  >= 800 mg

Tasks:
  (a) Formulate the LP: minimize cost subject to nutritional constraints
  (b) Solve using scipy.optimize.linprog
  (c) Interpret the solution: how many servings of each food?
  (d) Determine which constraints are active (binding) at the optimum
"""

import numpy as np


FOODS = ["Oatmeal", "Chicken", "Eggs", "Milk", "Beans", "Spinach"]
COST = np.array([0.30, 2.50, 0.80, 0.60, 0.40, 1.10])

# Rows: Calories, Protein, Fat, Calcium
NUTRIENTS = np.array([
    [110, 205, 160, 160, 260, 40],    # Calories
    [4, 32, 13, 8, 14, 5],            # Protein
    [2, 5, 10, 6, 1, 0],              # Fat
    [25, 12, 55, 300, 80, 100],        # Calcium
])

MIN_REQUIREMENTS = np.array([2000, 55, 30, 800])
MAX_FAT = 60


def formulate_and_solve():
    """
    Solve the diet problem LP.
    Return (min_cost, servings) where servings is an array of length 6.

    Constraints:
      - NUTRIENTS @ x >= MIN_REQUIREMENTS (note: linprog uses <=, so negate)
      - Fat total <= 60
      - x >= 0
    """
    # TODO
    raise NotImplementedError


def report_solution(servings):
    """Print a human-readable report of the diet."""
    print("\nOptimal Diet Plan:")
    print("-" * 40)
    for food, s in zip(FOODS, servings):
        if s > 1e-6:
            print(f"  {food:12s}: {s:.2f} servings")
    print("-" * 40)
    totals = NUTRIENTS @ servings
    labels = ["Calories", "Protein (g)", "Fat (g)", "Calcium (mg)"]
    print("Nutritional totals:")
    for label, val, req in zip(labels, totals, MIN_REQUIREMENTS):
        status = "BINDING" if abs(val - req) < 1e-4 else ""
        print(f"  {label:15s}: {val:8.1f}  (min: {req}) {status}")


if __name__ == "__main__":
    try:
        cost, servings = formulate_and_solve()
        print(f"Minimum cost: ${cost:.2f}")
        report_solution(servings)
    except NotImplementedError:
        print("TODO: implement formulate_and_solve()")
