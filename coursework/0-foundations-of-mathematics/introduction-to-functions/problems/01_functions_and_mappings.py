"""
Problem 1: Functions and Mappings

A function from set A to set B is represented as a dictionary mapping
each element of A to an element of B.

Tasks:
  (a) Check if a mapping (dict) is a valid function from domain A to
      codomain B (every element of A is mapped, every image is in B)
  (b) Check if a function is injective (one-to-one), surjective (onto),
      or bijective
  (c) Compute the composition of two functions (g after f)
  (d) Find the inverse of a bijection
  (e) Implement function iteration: f composed with itself n times
"""


# ---------------------------------------------------------------------------
# Part (a): Validate a function
# ---------------------------------------------------------------------------

def is_valid_function(f, domain, codomain):
    """
    Check if f is a valid function from domain to codomain.

    A valid function must:
      - Map every element in domain to exactly one value
      - Map every element to a value within codomain

    Args:
        f: dict mapping elements to elements
        domain: set of domain elements
        codomain: set of codomain elements

    Returns:
        True if f is a valid function from domain to codomain.

    Example:
        is_valid_function({1: 'a', 2: 'b'}, {1, 2}, {'a', 'b', 'c'})
        => True
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (b): Injective, surjective, bijective
# ---------------------------------------------------------------------------

def is_injective(f):
    """
    Check if function f (dict) is injective (one-to-one).

    A function is injective if no two distinct domain elements map to
    the same codomain element.

    Args:
        f: dict representing the function

    Returns:
        True if f is injective.
    """
    raise NotImplementedError


def is_surjective(f, codomain):
    """
    Check if function f (dict) is surjective (onto) with respect to codomain.

    A function is surjective if every element of codomain is mapped to
    by at least one domain element.

    Args:
        f: dict representing the function
        codomain: set of codomain elements

    Returns:
        True if f is surjective onto codomain.
    """
    raise NotImplementedError


def is_bijective(f, codomain):
    """
    Check if function f is bijective (both injective and surjective).

    Args:
        f: dict representing the function
        codomain: set of codomain elements

    Returns:
        True if f is bijective.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (c): Function composition
# ---------------------------------------------------------------------------

def compose(f, g):
    """
    Compute the composition g . f (first apply f, then g).

    For each x in the domain of f, (g . f)(x) = g(f(x)).
    The domain of f must be compatible: f(x) must be in the domain of g
    for every x.

    Args:
        f: dict representing function f
        g: dict representing function g

    Returns:
        dict representing g . f

    Example:
        compose({1: 'a', 2: 'b'}, {'a': True, 'b': False})
        => {1: True, 2: False}
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (d): Inverse of a bijection
# ---------------------------------------------------------------------------

def inverse(f):
    """
    Compute the inverse of a bijective function f.

    Args:
        f: dict representing a bijective function

    Returns:
        dict representing f^{-1}

    Raises:
        ValueError: if f is not injective (inverse does not exist)

    Example:
        inverse({1: 'a', 2: 'b', 3: 'c'}) => {'a': 1, 'b': 2, 'c': 3}
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (e): Function iteration
# ---------------------------------------------------------------------------

def iterate(f, n):
    """
    Compute the n-th iterate of f: f composed with itself n times.

    f^0 is the identity on dom(f), f^1 = f, f^2 = f . f, etc.
    Requires that f maps its domain into (a subset of) its domain.

    Args:
        f: dict representing a function from a set to itself
        n: non-negative integer

    Returns:
        dict representing f^n

    Example:
        f = {1: 2, 2: 3, 3: 1}
        iterate(f, 0) => {1: 1, 2: 2, 3: 3}
        iterate(f, 1) => {1: 2, 2: 3, 3: 1}
        iterate(f, 3) => {1: 1, 2: 2, 3: 3}   # f^3 = identity (cycle)
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Part (a): Validate Function ===")
    try:
        f = {1: 'a', 2: 'b', 3: 'a'}
        ok = is_valid_function(f, {1, 2, 3}, {'a', 'b', 'c'})
        print(f"  f = {f}, valid function: {ok}")
        bad = {1: 'a'}  # missing domain element 2
        ok2 = is_valid_function(bad, {1, 2}, {'a', 'b'})
        print(f"  f = {bad}, valid function from {{1,2}}: {ok2}")
    except NotImplementedError:
        print("TODO: implement is_valid_function")

    print()
    print("=== Part (b): Injective / Surjective / Bijective ===")
    try:
        f = {1: 'a', 2: 'b', 3: 'c'}
        print(f"  f = {f}")
        print(f"    Injective:  {is_injective(f)}")
        print(f"    Surjective: {is_surjective(f, {'a', 'b', 'c'})}")
        print(f"    Bijective:  {is_bijective(f, {'a', 'b', 'c'})}")
        g = {1: 'a', 2: 'a', 3: 'b'}
        print(f"  g = {g}")
        print(f"    Injective:  {is_injective(g)}")
    except NotImplementedError:
        print("TODO: implement injectivity / surjectivity checks")

    print()
    print("=== Part (c): Function Composition ===")
    try:
        f = {1: 'a', 2: 'b'}
        g = {'a': True, 'b': False}
        h = compose(f, g)
        print(f"  f = {f}, g = {g}")
        print(f"  g . f = {h}")
    except NotImplementedError:
        print("TODO: implement compose")

    print()
    print("=== Part (d): Inverse ===")
    try:
        f = {1: 'x', 2: 'y', 3: 'z'}
        inv = inverse(f)
        print(f"  f = {f}")
        print(f"  f^{{-1}} = {inv}")
    except NotImplementedError:
        print("TODO: implement inverse")

    print()
    print("=== Part (e): Function Iteration ===")
    try:
        f = {1: 2, 2: 3, 3: 1}
        for n in range(4):
            fn = iterate(f, n)
            print(f"  f^{n} = {fn}")
    except NotImplementedError:
        print("TODO: implement iterate")
