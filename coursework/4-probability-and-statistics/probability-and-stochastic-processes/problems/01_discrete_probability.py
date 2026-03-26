"""
Problem: Discrete Probability

Compute probabilities from finite sample spaces using classical rules
and Bayes' theorem.

Tasks:
  (a) Compute the probability of events from explicitly defined sample spaces
  (b) Apply Bayes' theorem to compute posterior probabilities
  (c) Compute conditional probabilities P(A|B) = P(A ∩ B) / P(B)
  (d) Test whether two events are independent: P(A ∩ B) == P(A) * P(B)
"""

import numpy as np


def event_probability(sample_space, event):
    """
    Compute the probability of an event assuming equally likely outcomes.

    Parameters:
        sample_space: list, all possible outcomes
        event: list, the favorable outcomes (subset of sample_space)

    Returns:
        float: P(event) = |event| / |sample_space|
    """
    # TODO
    raise NotImplementedError


def conditional_probability(p_a_and_b, p_b):
    """
    Compute the conditional probability P(A|B) = P(A ∩ B) / P(B).

    Parameters:
        p_a_and_b: float, probability of A and B
        p_b: float, probability of B (must be > 0)

    Returns:
        float: P(A|B)
    """
    # TODO
    raise NotImplementedError


def bayes_theorem(p_b_given_a, p_a, p_b):
    """
    Compute P(A|B) using Bayes' theorem:
    P(A|B) = P(B|A) * P(A) / P(B)

    Parameters:
        p_b_given_a: float, P(B|A)
        p_a: float, P(A), prior probability
        p_b: float, P(B), marginal probability of B (must be > 0)

    Returns:
        float: P(A|B), posterior probability
    """
    # TODO
    raise NotImplementedError


def bayes_full(p_b_given_hypotheses, p_hypotheses):
    """
    Compute posterior probabilities for all hypotheses using Bayes' theorem
    with the law of total probability.

    P(H_i|B) = P(B|H_i) * P(H_i) / sum_j(P(B|H_j) * P(H_j))

    Parameters:
        p_b_given_hypotheses: np.ndarray, shape (n,), P(B|H_i) for each hypothesis
        p_hypotheses: np.ndarray, shape (n,), prior P(H_i) for each hypothesis

    Returns:
        np.ndarray, shape (n,): posterior P(H_i|B) for each hypothesis
    """
    # TODO
    raise NotImplementedError


def are_independent(p_a, p_b, p_a_and_b, tol=1e-9):
    """
    Test whether events A and B are independent.

    Two events are independent if P(A ∩ B) = P(A) * P(B).

    Parameters:
        p_a: float, P(A)
        p_b: float, P(B)
        p_a_and_b: float, P(A ∩ B)
        tol: float, tolerance for floating-point comparison

    Returns:
        bool: True if A and B are independent
    """
    # TODO
    raise NotImplementedError


def simulate_event_probability(experiment_fn, event_fn, n_trials=100000):
    """
    Estimate a probability via Monte Carlo simulation.

    Parameters:
        experiment_fn: callable returning a random outcome
        event_fn: callable taking an outcome and returning bool
        n_trials: int, number of simulation trials

    Returns:
        float: estimated probability
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Fair die sample space
    sample_space = [1, 2, 3, 4, 5, 6]
    even = [2, 4, 6]

    try:
        p = event_probability(sample_space, even)
        print(f"P(even on fair die) = {p:.4f} (expected 0.5)")
    except NotImplementedError:
        print("TODO: implement event_probability")

    try:
        # P(A|B) where P(A ∩ B) = 0.1, P(B) = 0.4
        p = conditional_probability(0.1, 0.4)
        print(f"P(A|B) = {p:.4f} (expected 0.25)")
    except NotImplementedError:
        print("TODO: implement conditional_probability")

    try:
        # Medical test: P(disease) = 0.01, P(positive|disease) = 0.95,
        # P(positive) = 0.05
        p = bayes_theorem(0.95, 0.01, 0.05)
        print(f"P(disease|positive) = {p:.4f} (expected 0.19)")
    except NotImplementedError:
        print("TODO: implement bayes_theorem")

    try:
        # Two hypotheses: fair coin (H0) vs biased coin (H1, P(heads)=0.8)
        # Observed: heads
        posteriors = bayes_full(
            p_b_given_hypotheses=np.array([0.5, 0.8]),
            p_hypotheses=np.array([0.5, 0.5])
        )
        print(f"Posterior (fair | heads) = {posteriors[0]:.4f}")
        print(f"Posterior (biased | heads) = {posteriors[1]:.4f}")
    except NotImplementedError:
        print("TODO: implement bayes_full")

    try:
        independent = are_independent(0.5, 0.3, 0.15)
        print(f"\nP(A)=0.5, P(B)=0.3, P(A∩B)=0.15 -> independent: {independent}")
        dependent = are_independent(0.5, 0.3, 0.2)
        print(f"P(A)=0.5, P(B)=0.3, P(A∩B)=0.20 -> independent: {dependent}")
    except NotImplementedError:
        print("TODO: implement are_independent")
