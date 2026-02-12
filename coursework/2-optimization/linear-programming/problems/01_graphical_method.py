"""
Problem 1: Graphical Method for Linear Programming

Solve the following two-variable LP graphically:

    Maximize   z = 3x + 5y
    subject to:
        x + 2y <= 12
        2x + y <= 12
        x >= 0, y >= 0

Tasks:
  (a) Identify the corner points of the feasible region
  (b) Evaluate the objective function at each corner point
  (c) Determine the optimal solution and optimal value
  (d) Plot the feasible region and the optimal point
"""

import numpy as np
import matplotlib.pyplot as plt


def find_corner_points():
    """
    Return the corner points of the feasible region as a list of (x, y) tuples.
    Include all vertices of the feasible polygon formed by:
        x + 2y <= 12,  2x + y <= 12,  x >= 0,  y >= 0
    """
    # TODO: find intersections of constraint boundaries
    raise NotImplementedError


def evaluate_objective(points):
    """
    Evaluate z = 3x + 5y at each point.
    Return a list of (x, y, z) tuples.
    """
    return [(x, y, 3 * x + 5 * y) for x, y in points]


def find_optimal(points):
    """
    Return (x_opt, y_opt, z_opt) that maximizes z = 3x + 5y
    among the given corner points.
    """
    evaluated = evaluate_objective(points)
    return max(evaluated, key=lambda t: t[2])


def plot_feasible_region(corner_points=None):
    """Plot the feasible region, constraints, and optimal point."""
    x = np.linspace(0, 8, 400)

    # Constraint boundaries
    y1 = (12 - x) / 2      # x + 2y = 12
    y2 = 12 - 2 * x        # 2x + y = 12

    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, "b-", label=r"$x + 2y = 12$")
    plt.plot(x, y2, "r-", label=r"$2x + y = 12$")

    # Shade feasible region
    y_upper = np.minimum(y1, y2)
    y_upper = np.maximum(y_upper, 0)
    plt.fill_between(x, 0, y_upper, where=(y_upper >= 0) & (x >= 0),
                     alpha=0.2, color="green", label="Feasible region")

    if corner_points is not None:
        cx, cy = zip(*corner_points)
        plt.plot(cx, cy, "ko", markersize=6)
        opt = find_optimal(corner_points)
        plt.plot(opt[0], opt[1], "r*", markersize=15, label=f"Optimal ({opt[0]}, {opt[1]})")

    plt.xlim(-0.5, 8)
    plt.ylim(-0.5, 8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Graphical LP Solution")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        pts = find_corner_points()
        print("Corner points:", pts)
        print("Objective values:", evaluate_objective(pts))
        print("Optimal solution:", find_optimal(pts))
        plot_feasible_region(pts)
    except NotImplementedError:
        print("TODO: implement find_corner_points()")
        plot_feasible_region()
