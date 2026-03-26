"""
Problem: Pigeonhole Principle

Explore the pigeonhole principle through simulation and proof-based examples.

Tasks:
  (a) Simulate the birthday paradox: estimate the probability that at least two
      people in a group share a birthday, using Monte Carlo trials
  (b) Verify the pigeonhole principle: given n items placed into m containers
      (n > m), prove that at least one container holds ceil(n/m) items
  (c) Demonstrate minimum collisions in a hashing scenario: when n keys are
      hashed into a table of size table_size, show the guaranteed minimum
      number of collisions
"""

import random
import math


def birthday_paradox_simulation(num_people, num_trials=10000):
    """
    Estimate the probability that at least two people in a group of
    `num_people` share a birthday (out of 365 days).

    Parameters:
        num_people: number of people in the group
        num_trials: number of Monte Carlo simulations to run

    Returns:
        float: estimated probability of at least one shared birthday
    """
    # TODO
    raise NotImplementedError


def pigeonhole_proof(n_items, m_containers):
    """
    Demonstrate the pigeonhole principle: if n_items are distributed among
    m_containers, at least one container must contain at least
    ceil(n_items / m_containers) items.

    Parameters:
        n_items: number of items to distribute
        m_containers: number of containers

    Returns:
        dict with keys:
            'min_max_load': int, the guaranteed minimum for the maximum load
            'distribution': list[int], one example distribution showing the principle
    """
    # TODO
    raise NotImplementedError


def hash_collision_demo(n_keys, table_size):
    """
    Demonstrate guaranteed collisions when hashing n_keys into a hash table
    of the given size.

    Parameters:
        n_keys: number of keys to insert
        table_size: number of slots in the hash table

    Returns:
        dict with keys:
            'min_collisions': int, minimum guaranteed number of collisions
                              (keys beyond one per occupied slot)
            'example_table': dict mapping slot -> list of keys that hashed there
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    try:
        # Birthday paradox: with 23 people, probability should be ~0.507
        prob = birthday_paradox_simulation(23, num_trials=100000)
        print(f"Birthday paradox (23 people): P(shared birthday) ~ {prob:.4f}")

        prob50 = birthday_paradox_simulation(50, num_trials=100000)
        print(f"Birthday paradox (50 people): P(shared birthday) ~ {prob50:.4f}")
    except NotImplementedError:
        print("TODO: implement birthday_paradox_simulation")

    try:
        result = pigeonhole_proof(10, 3)
        print(f"\nPigeonhole: 10 items, 3 containers -> "
              f"min max load = {result['min_max_load']}")
        print(f"Example distribution: {result['distribution']}")
    except NotImplementedError:
        print("TODO: implement pigeonhole_proof")

    try:
        result = hash_collision_demo(15, 10)
        print(f"\nHash collisions: 15 keys, 10 slots -> "
              f"min collisions = {result['min_collisions']}")
    except NotImplementedError:
        print("TODO: implement hash_collision_demo")
