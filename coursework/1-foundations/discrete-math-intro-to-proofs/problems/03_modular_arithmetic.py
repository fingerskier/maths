"""
Problem 3: Modular Arithmetic and the Euclidean Algorithm

Tasks:
  (a) Implement the Extended Euclidean Algorithm:
      Given a, b, find gcd(a, b) and coefficients x, y such that a*x + b*y = gcd(a, b)
  (b) Implement modular exponentiation: compute (base^exp) mod m efficiently
  (c) Find the modular inverse of a mod m (when it exists)
  (d) Solve a system of linear congruences using the Chinese Remainder Theorem:
        x ≡ 2 (mod 3)
        x ≡ 3 (mod 5)
        x ≡ 2 (mod 7)
"""


def gcd(a, b):
    """Compute gcd(a, b) using the Euclidean algorithm."""
    # TODO
    raise NotImplementedError


def extended_gcd(a, b):
    """
    Return (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    # TODO
    raise NotImplementedError


def mod_exp(base, exp, m):
    """Compute (base^exp) % m using fast exponentiation (repeated squaring)."""
    # TODO
    raise NotImplementedError


def mod_inverse(a, m):
    """
    Return x such that (a * x) % m == 1.
    Raise ValueError if gcd(a, m) != 1 (no inverse exists).
    """
    # TODO: use extended_gcd
    raise NotImplementedError


def chinese_remainder_theorem(remainders, moduli):
    """
    Solve the system: x ≡ r_i (mod m_i) for all i.
    Returns x mod (product of all moduli).

    Assumes all moduli are pairwise coprime.
    """
    # TODO
    raise NotImplementedError


def test_extended_gcd():
    g, x, y = extended_gcd(240, 46)
    assert g == 2
    assert 240 * x + 46 * y == 2
    print(f"gcd(240, 46) = {g}, 240*{x} + 46*{y} = {g}")


def test_mod_exp():
    # 2^10 mod 1000 = 1024 mod 1000 = 24
    assert mod_exp(2, 10, 1000) == 24
    # Fermat's little theorem: 2^(p-1) mod p = 1 for prime p
    assert mod_exp(2, 12, 13) == 1
    print("mod_exp tests passed")


def test_crt():
    # x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = 23 (mod 105)
    x = chinese_remainder_theorem([2, 3, 2], [3, 5, 7])
    assert x % 3 == 2
    assert x % 5 == 3
    assert x % 7 == 2
    print(f"CRT solution: x = {x} (mod 105)")


if __name__ == "__main__":
    try:
        test_extended_gcd()
        test_mod_exp()
        test_crt()
    except NotImplementedError:
        print("TODO: implement the functions")
