"""
Problem 4: Jacobian and Change of Variables

Consider the transformation from (u, v) to (x, y):
    x = u^2 - v^2
    y = 2*u*v

Tasks:
  (a) Compute the Jacobian matrix of the transformation
  (b) Compute the Jacobian determinant |d(x,y)/d(u,v)|
  (c) Use the change of variables to evaluate the integral of 1 over the
      image region when (u, v) ranges over the unit disk u^2 + v^2 <= 1
"""

import numpy as np


def jacobian_matrix(u, v):
    """Return the 2x2 Jacobian matrix d(x,y)/d(u,v) as a numpy array."""
    # TODO: compute [[dx/du, dx/dv], [dy/du, dy/dv]]
    raise NotImplementedError


def jacobian_determinant(u, v):
    """Return the absolute value of the Jacobian determinant."""
    J = jacobian_matrix(u, v)
    return abs(np.linalg.det(J))


def transformed_area():
    """
    Compute the area of the image of the unit disk under the transformation
    using the change of variables formula:
        Area = integral over unit disk of |det(J)| du dv

    Return the analytic result.
    """
    # TODO: evaluate the integral analytically
    raise NotImplementedError


def transformed_area_numerical():
    """Verify numerically using Monte Carlo integration."""
    rng = np.random.default_rng(42)
    n = 1_000_000
    # sample uniformly from the square [-1,1]x[-1,1]
    uv = rng.uniform(-1, 1, size=(n, 2))
    # keep only points inside the unit disk
    mask = uv[:, 0] ** 2 + uv[:, 1] ** 2 <= 1
    uv = uv[mask]
    det_vals = np.array([jacobian_determinant(u, v) for u, v in uv])
    # area of unit disk is pi, fraction of square that's in disk is pi/4
    area_disk = np.pi
    return area_disk * np.mean(det_vals)


if __name__ == "__main__":
    print("Jacobian at (1,1):\n", jacobian_matrix(1, 1))
    print("|det J| at (1,1):", jacobian_determinant(1, 1))
    print("Analytic area:", transformed_area())
    print("Numerical area:", transformed_area_numerical())
