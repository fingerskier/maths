"""
Problem: Markov Chains

Represent and analyze discrete-time Markov chains using transition matrices.

Tasks:
  (a) Represent a Markov chain as a transition matrix and verify it is stochastic
  (b) Compute n-step transition probabilities via matrix exponentiation
  (c) Find the stationary distribution (left eigenvector for eigenvalue 1)
  (d) Classify states: identify absorbing, transient, and recurrent states
  (e) Simulate sample paths of the chain
"""

import numpy as np


def is_stochastic_matrix(P):
    """
    Check if P is a valid (row) stochastic matrix: all entries non-negative,
    each row sums to 1.

    Parameters:
        P: np.ndarray, shape (n, n)

    Returns:
        bool: True if P is a valid stochastic matrix
    """
    # TODO
    raise NotImplementedError


def n_step_transition(P, n):
    """
    Compute the n-step transition matrix P^n.

    Parameters:
        P: np.ndarray, shape (m, m), transition matrix
        n: int, number of steps

    Returns:
        np.ndarray, shape (m, m): the n-step transition matrix
    """
    # TODO
    raise NotImplementedError


def stationary_distribution(P):
    """
    Find the stationary distribution pi such that pi @ P = pi and sum(pi) = 1.

    Solve by finding the left eigenvector corresponding to eigenvalue 1,
    or by solving the linear system.

    Parameters:
        P: np.ndarray, shape (n, n), transition matrix

    Returns:
        np.ndarray, shape (n,): the stationary distribution
    """
    # TODO
    raise NotImplementedError


def classify_states(P):
    """
    Classify each state of the Markov chain.

    Parameters:
        P: np.ndarray, shape (n, n), transition matrix

    Returns:
        dict with keys:
            'absorbing': list[int], states i where P[i, i] = 1
            'recurrent': list[int], states in closed communicating classes
            'transient': list[int], all other states
    """
    # TODO
    raise NotImplementedError


def simulate_chain(P, initial_state, n_steps, rng=None):
    """
    Simulate a sample path of the Markov chain.

    Parameters:
        P: np.ndarray, shape (m, m), transition matrix
        initial_state: int, starting state index
        n_steps: int, number of transitions to simulate
        rng: np.random.Generator or None

    Returns:
        list[int]: sequence of states [X_0, X_1, ..., X_{n_steps}]
    """
    # TODO
    raise NotImplementedError


def expected_hitting_time(P, target_state):
    """
    Compute the expected hitting time to target_state from each other state.

    Solve the system: h_i = 1 + sum_j P[i,j] * h_j for i != target,
    h_target = 0.

    Parameters:
        P: np.ndarray, shape (n, n)
        target_state: int

    Returns:
        np.ndarray, shape (n,): expected hitting times from each state
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Weather model: Sunny(0), Cloudy(1), Rainy(2)
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])

    try:
        print(f"Is stochastic: {is_stochastic_matrix(P)}")
    except NotImplementedError:
        print("TODO: implement is_stochastic_matrix")

    try:
        P10 = n_step_transition(P, 10)
        print(f"\n10-step transition matrix:\n{P10}")
    except NotImplementedError:
        print("TODO: implement n_step_transition")

    try:
        pi = stationary_distribution(P)
        print(f"\nStationary distribution: {pi}")
        print(f"Verification (pi @ P): {pi @ P}")
    except NotImplementedError:
        print("TODO: implement stationary_distribution")

    try:
        classes = classify_states(P)
        print(f"\nState classification: {classes}")
    except NotImplementedError:
        print("TODO: implement classify_states")

    try:
        path = simulate_chain(P, 0, 20)
        states = {0: 'Sunny', 1: 'Cloudy', 2: 'Rainy'}
        print(f"\nSample path: {[states[s] for s in path]}")
    except NotImplementedError:
        print("TODO: implement simulate_chain")

    try:
        h = expected_hitting_time(P, 0)
        print(f"\nExpected hitting time to Sunny: {h}")
    except NotImplementedError:
        print("TODO: implement expected_hitting_time")
