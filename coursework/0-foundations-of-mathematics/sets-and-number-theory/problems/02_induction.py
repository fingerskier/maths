"""
Problem 2: Mathematical Induction

Prove and verify the following identities using mathematical induction.
For each, provide:
  - A computational verification for n = 1..1000
  - A written proof outline (as comments or docstring)

Identity 1: sum(k, k=1..n) = n*(n+1)/2
Identity 2: sum(k^2, k=1..n) = n*(n+1)*(2n+1)/6
Identity 3: sum(2^k, k=0..n) = 2^(n+1) - 1

Tasks:
  (a) Implement the left-hand side (direct summation) and right-hand side
      (closed form) for each identity
  (b) Verify they agree for n = 1 to 1000
  (c) Write the proof structure:
      - Base case (n=1)
      - Inductive step: assume P(k), prove P(k+1)
"""


def sum_of_integers_lhs(n):
    """Direct computation: 1 + 2 + ... + n."""
    # TODO
    raise NotImplementedError


def sum_of_integers_rhs(n):
    """Closed form: n*(n+1)//2."""
    # TODO
    raise NotImplementedError


def sum_of_squares_lhs(n):
    """Direct computation: 1^2 + 2^2 + ... + n^2."""
    # TODO
    raise NotImplementedError


def sum_of_squares_rhs(n):
    """Closed form: n*(n+1)*(2*n+1)//6."""
    # TODO
    raise NotImplementedError


def sum_of_powers_of_2_lhs(n):
    """Direct computation: 2^0 + 2^1 + ... + 2^n."""
    # TODO
    raise NotImplementedError


def sum_of_powers_of_2_rhs(n):
    """Closed form: 2^(n+1) - 1."""
    # TODO
    raise NotImplementedError


def verify_all(max_n=1000):
    """Verify all three identities for n = 1..max_n."""
    for n in range(1, max_n + 1):
        assert sum_of_integers_lhs(n) == sum_of_integers_rhs(n), \
            f"Identity 1 failed at n={n}"
        assert sum_of_squares_lhs(n) == sum_of_squares_rhs(n), \
            f"Identity 2 failed at n={n}"
        assert sum_of_powers_of_2_lhs(n) == sum_of_powers_of_2_rhs(n), \
            f"Identity 3 failed at n={n}"
    print(f"All three identities verified for n = 1..{max_n}")


"""
Proof Outline for Identity 1: sum(k, k=1..n) = n(n+1)/2

Base case (n=1):
    LHS = 1, RHS = 1*2/2 = 1. ✓

Inductive step:
    Assume sum(k, k=1..m) = m(m+1)/2 for some m >= 1.
    Then sum(k, k=1..m+1) = sum(k, k=1..m) + (m+1)
                           = m(m+1)/2 + (m+1)
                           = (m+1)(m/2 + 1)
                           = (m+1)(m+2)/2. ✓

TODO: Write similar proofs for Identities 2 and 3.
"""


if __name__ == "__main__":
    try:
        verify_all()
    except NotImplementedError:
        print("TODO: implement the summation functions")
