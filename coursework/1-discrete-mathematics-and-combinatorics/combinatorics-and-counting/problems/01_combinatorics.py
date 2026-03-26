"""
Problem 5: Combinatorics and Counting

Tasks:
  (a) Implement factorial, permutations P(n,k), and combinations C(n,k) from scratch
  (b) Verify Pascal's identity: C(n,k) = C(n-1,k-1) + C(n-1,k)
  (c) Verify the Binomial Theorem: (x+y)^n = sum(C(n,k) * x^k * y^(n-k), k=0..n)
  (d) Implement Catalan numbers and verify the first 15 values
  (e) Count the number of derangements D(n) and verify the formula
      D(n) = n! * sum((-1)^k / k!, k=0..n)
"""


def factorial(n):
    """Compute n! iteratively."""
    # TODO
    raise NotImplementedError


def permutations(n, k):
    """Compute P(n, k) = n! / (n-k)!"""
    # TODO
    raise NotImplementedError


def combinations(n, k):
    """Compute C(n, k) = n! / (k! * (n-k)!)"""
    # TODO
    raise NotImplementedError


def verify_pascals_identity(max_n=20):
    """Check C(n,k) = C(n-1,k-1) + C(n-1,k) for all valid n, k."""
    for n in range(2, max_n + 1):
        for k in range(1, n):
            assert combinations(n, k) == combinations(n - 1, k - 1) + combinations(n - 1, k), \
                f"Pascal's identity failed at n={n}, k={k}"
    print(f"Pascal's identity verified for n = 2..{max_n}")


def verify_binomial_theorem(n=10, x=2, y=3):
    """Check (x+y)^n == sum(C(n,k) * x^k * y^(n-k))."""
    lhs = (x + y) ** n
    rhs = sum(combinations(n, k) * x**k * y**(n - k) for k in range(n + 1))
    assert lhs == rhs, f"Binomial theorem failed for n={n}, x={x}, y={y}"
    print(f"Binomial Theorem verified: ({x}+{y})^{n} = {lhs}")


def catalan(n):
    """Compute the n-th Catalan number: C(n) = C(2n, n) / (n+1)."""
    # TODO
    raise NotImplementedError


def verify_catalan():
    """Verify first 15 Catalan numbers against known values."""
    known = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862,
             16796, 58786, 208012, 742900, 2674440]
    for i, expected in enumerate(known):
        assert catalan(i) == expected, f"Catalan({i}) = {catalan(i)}, expected {expected}"
    print("First 15 Catalan numbers verified.")


def derangements(n):
    """
    Compute D(n), the number of derangements of n elements.
    D(n) = n! * sum((-1)^k / k!, k=0..n)
    """
    # TODO
    raise NotImplementedError


def verify_derangements():
    """Verify first several derangement numbers."""
    known = {0: 1, 1: 0, 2: 1, 3: 2, 4: 9, 5: 44, 6: 265, 7: 1854, 8: 14833}
    for n, expected in known.items():
        assert derangements(n) == expected, \
            f"D({n}) = {derangements(n)}, expected {expected}"
    print("Derangement numbers verified.")


if __name__ == "__main__":
    try:
        verify_pascals_identity()
        verify_binomial_theorem()
        verify_catalan()
        verify_derangements()
    except NotImplementedError:
        print("TODO: implement the functions")
