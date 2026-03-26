"""
Problem: Random Walks

Simulate and analyze properties of random walks in one and two dimensions.

Tasks:
  (a) Simulate 1D symmetric random walks (+1 or -1 with equal probability)
  (b) Simulate 2D random walks on a lattice
  (c) Compute the return probability (probability of returning to origin)
  (d) Estimate the first passage time distribution to a target position
  (e) Demonstrate the connection to diffusion: the scaled random walk
      converges to Brownian motion
"""

import numpy as np
import matplotlib.pyplot as plt


def random_walk_1d(n_steps, rng=None):
    """
    Simulate a 1D symmetric random walk starting at the origin.

    At each step, move +1 or -1 with equal probability.

    Parameters:
        n_steps: int, number of steps
        rng: np.random.Generator or None

    Returns:
        np.ndarray, shape (n_steps + 1,): positions [X_0, X_1, ..., X_{n_steps}]
            where X_0 = 0
    """
    # TODO
    raise NotImplementedError


def random_walk_2d(n_steps, rng=None):
    """
    Simulate a 2D lattice random walk starting at the origin.

    At each step, move in one of four directions (up, down, left, right)
    with equal probability.

    Parameters:
        n_steps: int, number of steps
        rng: np.random.Generator or None

    Returns:
        np.ndarray, shape (n_steps + 1, 2): positions [(x_0, y_0), ..., (x_n, y_n)]
            where (x_0, y_0) = (0, 0)
    """
    # TODO
    raise NotImplementedError


def estimate_return_probability(n_steps, n_trials, dimension=1, rng=None):
    """
    Estimate the probability that a random walk returns to the origin
    within n_steps.

    Parameters:
        n_steps: int, maximum number of steps
        n_trials: int, number of independent walks to simulate
        dimension: int, 1 or 2
        rng: np.random.Generator or None

    Returns:
        float: estimated return probability
    """
    # TODO
    raise NotImplementedError


def first_passage_time(target, n_trials, max_steps=10000, rng=None):
    """
    Estimate the distribution of first passage times for a 1D random walk
    to reach the target position.

    Parameters:
        target: int, target position (e.g., +5 or -5)
        n_trials: int, number of walks to simulate
        max_steps: int, maximum steps before declaring no passage
        rng: np.random.Generator or None

    Returns:
        dict with keys:
            'times': list[int], first passage times for walks that reached target
            'mean': float, average first passage time (among those that reached)
            'fraction_reached': float, fraction of walks that reached target
    """
    # TODO
    raise NotImplementedError


def scaled_random_walk(n_steps, dt=None):
    """
    Construct a scaled random walk that approximates Brownian motion.

    If dt = 1/n_steps, then W(t) = sqrt(dt) * S(t/dt) where S is the
    cumulative sum of +1/-1 steps.

    Parameters:
        n_steps: int, number of steps
        dt: float or None, time increment (defaults to 1/n_steps)

    Returns:
        t: np.ndarray, shape (n_steps + 1,), time points in [0, 1]
        W: np.ndarray, shape (n_steps + 1,), scaled walk values
    """
    # TODO
    raise NotImplementedError


def plot_random_walks(n_steps=1000, n_walks=5):
    """
    Plot several 1D random walk trajectories on the same axes.

    Parameters:
        n_steps: int
        n_walks: int, number of walks to plot

    Returns:
        matplotlib.figure.Figure
    """
    # TODO
    raise NotImplementedError


def plot_2d_walk(n_steps=1000):
    """
    Plot a single 2D random walk trajectory.

    Parameters:
        n_steps: int

    Returns:
        matplotlib.figure.Figure
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    try:
        walk = random_walk_1d(100)
        print(f"1D walk (first 20 positions): {walk[:20]}")
    except NotImplementedError:
        print("TODO: implement random_walk_1d")

    try:
        walk2d = random_walk_2d(100)
        print(f"\n2D walk (first 10 positions):\n{walk2d[:10]}")
    except NotImplementedError:
        print("TODO: implement random_walk_2d")

    try:
        p1d = estimate_return_probability(1000, 10000, dimension=1)
        print(f"\n1D return probability (1000 steps): {p1d:.4f}")
        p2d = estimate_return_probability(1000, 10000, dimension=2)
        print(f"2D return probability (1000 steps): {p2d:.4f}")
    except NotImplementedError:
        print("TODO: implement estimate_return_probability")

    try:
        result = first_passage_time(5, 10000)
        print(f"\nFirst passage to +5: mean={result['mean']:.1f}, "
              f"reached={result['fraction_reached']:.4f}")
    except NotImplementedError:
        print("TODO: implement first_passage_time")

    try:
        t, W = scaled_random_walk(10000)
        print(f"\nScaled walk: W(0)={W[0]:.4f}, W(1)={W[-1]:.4f}")
        print(f"Std of endpoint (should be ~1): {np.std(W[-1]):.4f}")
    except NotImplementedError:
        print("TODO: implement scaled_random_walk")
