"""
Problem 2: Gradient and Directional Derivative

Given f(x, y, z) = x*y*z + sin(x + y), compute:

Tasks:
  (a) The gradient vector grad(f) at point (pi/4, pi/4, 1)
  (b) The directional derivative of f at that point in the direction
      of the unit vector u = (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
  (c) The direction of steepest ascent at that point
"""

import numpy as np


def gradient_f(x, y, z):
    """Return the gradient of f as a numpy array [df/dx, df/dy, df/dz]."""
    # TODO: compute analytically
    raise NotImplementedError


def directional_derivative(x, y, z, u):
    """Return the directional derivative of f at (x,y,z) in direction u."""
    # TODO: dot product of gradient with unit vector
    raise NotImplementedError


def steepest_ascent_direction(x, y, z):
    """Return the unit vector in the direction of steepest ascent."""
    grad = gradient_f(x, y, z)
    return grad / np.linalg.norm(grad)


if __name__ == "__main__":
    p = (np.pi / 4, np.pi / 4, 1)
    u = np.array([1, 1, 1]) / np.sqrt(3)

    print("Gradient at p:", gradient_f(*p))
    print("Directional derivative:", directional_derivative(*p, u))
    print("Steepest ascent direction:", steepest_ascent_direction(*p))
