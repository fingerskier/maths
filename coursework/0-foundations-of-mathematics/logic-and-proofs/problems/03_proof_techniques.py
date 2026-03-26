"""
Problem 3: Proof Techniques

Tasks:
  (a) Implement direct proof verification for simple theorems:
      - The sum of two even numbers is even
      - The product of two odd numbers is odd
      - The sum of an even and an odd number is odd
  (b) Implement proof by contradiction (numerical verification):
      - sqrt(2) is irrational: search for coprime p, q with p^2 = 2*q^2
      - There are infinitely many primes (verify no finite set suffices)
  (c) Implement proof by induction checker:
      - Verify base case and inductive step for formulas like
        sum(1..n) = n*(n+1)/2
      - Verify base case and inductive step for sum of first n odd
        numbers = n^2
  (d) Implement proof by contrapositive examples:
      - If n^2 is even then n is even (prove via contrapositive:
        if n is odd then n^2 is odd)
"""

from math import gcd, isqrt


# ---------------------------------------------------------------------------
# Part (a): Direct proof verification
# ---------------------------------------------------------------------------

def is_even(n):
    """Return True if n is even."""
    raise NotImplementedError


def is_odd(n):
    """Return True if n is odd."""
    raise NotImplementedError


def verify_sum_of_two_evens_is_even(limit=100):
    """
    Verify by exhaustive check that for all even a, b in [0, limit),
    a + b is even.

    Returns:
        True if the property holds for all tested pairs.
    """
    raise NotImplementedError


def verify_product_of_two_odds_is_odd(limit=100):
    """
    Verify by exhaustive check that for all odd a, b in [1, limit),
    a * b is odd.

    Returns:
        True if the property holds for all tested pairs.
    """
    raise NotImplementedError


def verify_even_plus_odd_is_odd(limit=100):
    """
    Verify by exhaustive check that for all even a and odd b in [0, limit),
    a + b is odd.

    Returns:
        True if the property holds for all tested pairs.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (b): Proof by contradiction (numerical verification)
# ---------------------------------------------------------------------------

def sqrt2_irrationality_search(max_denominator=10000):
    """
    Search for coprime integers p, q with p^2 = 2*q^2.

    If sqrt(2) were rational, such p, q would exist. This function
    searches up to max_denominator and returns any counterexample found,
    or None if none exists (supporting the irrationality claim).

    Returns:
        (p, q) if found, or None if no such pair exists up to max_denominator.
    """
    raise NotImplementedError


def verify_infinitely_many_primes(n_primes=20):
    """
    Verify that no finite list of the first k primes generates all primes.

    For each k in 1..n_primes, take the first k primes p1, ..., pk,
    compute N = p1*p2*...*pk + 1, and verify that N is either prime itself
    or has a prime factor not in {p1, ..., pk}.

    Returns:
        True if the verification passes for all tested k.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (c): Proof by induction checker
# ---------------------------------------------------------------------------

def check_induction(base_n, predicate, inductive_step, max_n=100):
    """
    Verify a proof by induction up to max_n.

    Args:
        base_n: the base case value (usually 0 or 1)
        predicate: callable(n) -> bool that checks if the property holds for n
        inductive_step: callable(n) -> bool that checks: if predicate(n)
                        then predicate(n+1) (returns True if step is valid)
        max_n: verify up to this value

    Returns:
        True if base case holds and inductive step holds for all
        base_n <= n < max_n.
    """
    raise NotImplementedError


def verify_sum_formula(n):
    """
    Check that 1 + 2 + ... + n == n*(n+1)//2.

    Args:
        n: positive integer

    Returns:
        True if the formula holds.
    """
    raise NotImplementedError


def verify_sum_of_odd_numbers(n):
    """
    Check that 1 + 3 + 5 + ... + (2n-1) == n^2.

    Args:
        n: positive integer

    Returns:
        True if the formula holds.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (d): Proof by contrapositive
# ---------------------------------------------------------------------------

def verify_contrapositive_even_square(limit=200):
    """
    Prove 'if n^2 is even then n is even' via the contrapositive:
    'if n is odd then n^2 is odd'.

    Verify the contrapositive for all n in [0, limit).

    Returns:
        True if the contrapositive holds for all tested values.
    """
    raise NotImplementedError


def verify_contrapositive_divisibility(limit=200):
    """
    Prove 'if n^2 is divisible by 3 then n is divisible by 3' via
    the contrapositive: 'if n is not divisible by 3 then n^2 is not
    divisible by 3'.

    Verify for all n in [1, limit).

    Returns:
        True if the contrapositive holds for all tested values.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Part (a): Direct Proof Verification ===")
    try:
        print(f"Sum of two evens is even: {verify_sum_of_two_evens_is_even()}")
        print(f"Product of two odds is odd: {verify_product_of_two_odds_is_odd()}")
        print(f"Even + odd is odd: {verify_even_plus_odd_is_odd()}")
    except NotImplementedError:
        print("TODO: implement direct proof verifications")

    print()
    print("=== Part (b): Proof by Contradiction ===")
    try:
        result = sqrt2_irrationality_search()
        if result is None:
            print("No coprime p, q with p^2 = 2*q^2 found (supports irrationality)")
        else:
            print(f"Found counterexample: p={result[0]}, q={result[1]}")
        print(f"Infinitely many primes check: {verify_infinitely_many_primes()}")
    except NotImplementedError:
        print("TODO: implement proof by contradiction checks")

    print()
    print("=== Part (c): Proof by Induction ===")
    try:
        # Check sum formula via induction framework
        ok = check_induction(
            base_n=1,
            predicate=lambda n: sum(range(1, n + 1)) == n * (n + 1) // 2,
            inductive_step=lambda n: (
                (sum(range(1, n + 1)) == n * (n + 1) // 2)
                <= (sum(range(1, n + 2)) == (n + 1) * (n + 2) // 2)
            ),
            max_n=50,
        )
        print(f"Sum formula induction: {ok}")
        print(f"Sum formula direct check n=100: {verify_sum_formula(100)}")
        print(f"Sum of odd numbers n=50: {verify_sum_of_odd_numbers(50)}")
    except NotImplementedError:
        print("TODO: implement induction checker")

    print()
    print("=== Part (d): Proof by Contrapositive ===")
    try:
        print(f"n^2 even => n even (via contrapositive): "
              f"{verify_contrapositive_even_square()}")
        print(f"3|n^2 => 3|n (via contrapositive): "
              f"{verify_contrapositive_divisibility()}")
    except NotImplementedError:
        print("TODO: implement contrapositive verifications")
