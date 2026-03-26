"""
Problem 2: Linear Systems and Gaussian Elimination

Solve linear systems Ax = b without using numpy.linalg.  Matrices are
represented as lists of lists (row-major order).

Tasks:
  (a) Solve 2x2 and 3x3 linear systems using Gaussian elimination
      with partial pivoting
  (b) Implement row reduction to (row) echelon form
  (c) Detect inconsistent systems (no solution) and underdetermined
      systems (infinitely many solutions)
  (d) Compute the determinant of a square matrix via cofactor expansion
"""


# ---------------------------------------------------------------------------
# Part (a): Gaussian elimination solver
# ---------------------------------------------------------------------------

def solve_2x2(A, b):
    """
    Solve a 2x2 system A x = b using Gaussian elimination.

    Args:
        A: 2x2 matrix as [[a11, a12], [a21, a22]]
        b: right-hand side as [b1, b2]

    Returns:
        Solution as [x1, x2], or None if the system is singular.

    Example:
        solve_2x2([[2, 1], [1, 3]], [5, 10]) => [1.0, 3.0]
    """
    raise NotImplementedError


def gaussian_elimination(A, b):
    """
    Solve an n x n system A x = b using Gaussian elimination with
    partial pivoting.

    Args:
        A: n x n matrix as list of lists (will not be mutated)
        b: right-hand side as list of length n

    Returns:
        Solution as list of floats, or None if the system is singular.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (b): Row reduction to echelon form
# ---------------------------------------------------------------------------

def row_echelon_form(M):
    """
    Reduce an m x n matrix M to row echelon form using partial pivoting.

    The matrix is modified in place (a copy is made internally).

    Args:
        M: m x n matrix as list of lists

    Returns:
        The row echelon form as a list of lists (floats).

    Example:
        row_echelon_form([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        # Some upper-triangular-like form
    """
    raise NotImplementedError


def reduced_row_echelon_form(M):
    """
    Reduce an m x n matrix M to reduced row echelon form (RREF).

    In RREF each pivot is 1 and is the only nonzero entry in its column.

    Args:
        M: m x n matrix as list of lists

    Returns:
        The RREF as a list of lists (floats).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (c): Detect inconsistent and underdetermined systems
# ---------------------------------------------------------------------------

def classify_system(A, b):
    """
    Classify the linear system Ax = b.

    Form the augmented matrix [A | b], reduce to echelon form, and
    determine the nature of the system.

    Args:
        A: m x n matrix as list of lists
        b: right-hand side as list of length m

    Returns:
        One of the strings:
            "unique"          - exactly one solution
            "underdetermined" - infinitely many solutions
            "inconsistent"    - no solution
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part (d): Determinant via cofactor expansion
# ---------------------------------------------------------------------------

def minor(M, row, col):
    """
    Compute the minor matrix obtained by deleting the given row and column.

    Args:
        M: n x n matrix as list of lists
        row: row index to delete
        col: column index to delete

    Returns:
        (n-1) x (n-1) matrix as list of lists.
    """
    raise NotImplementedError


def determinant(M):
    """
    Compute the determinant of a square matrix via cofactor expansion
    along the first row.

    Args:
        M: n x n matrix as list of lists

    Returns:
        The determinant (a number).

    Example:
        determinant([[1, 2], [3, 4]]) => -2
        determinant([[6, 1, 1],
                     [4, -2, 5],
                     [2, 8, 7]]) => -306
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== Part (a): Gaussian Elimination ===")
    try:
        sol = solve_2x2([[2, 1], [1, 3]], [5, 10])
        print(f"  2x2 solution: {sol}")
        sol3 = gaussian_elimination(
            [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
            [8, -11, -3],
        )
        print(f"  3x3 solution: {sol3}")
    except NotImplementedError:
        print("TODO: implement Gaussian elimination")

    print()
    print("=== Part (b): Row Echelon Form ===")
    try:
        ref = row_echelon_form([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print("  REF:")
        for row in ref:
            print(f"    {[round(x, 4) for x in row]}")
    except NotImplementedError:
        print("TODO: implement row_echelon_form")

    print()
    print("=== Part (c): System Classification ===")
    try:
        c1 = classify_system([[1, 2], [3, 4]], [5, 6])
        print(f"  [[1,2],[3,4]]x = [5,6]: {c1}")
        c2 = classify_system([[1, 2], [2, 4]], [3, 6])
        print(f"  [[1,2],[2,4]]x = [3,6]: {c2}")
        c3 = classify_system([[1, 2], [2, 4]], [3, 7])
        print(f"  [[1,2],[2,4]]x = [3,7]: {c3}")
    except NotImplementedError:
        print("TODO: implement classify_system")

    print()
    print("=== Part (d): Determinant ===")
    try:
        d2 = determinant([[1, 2], [3, 4]])
        print(f"  det([[1,2],[3,4]]) = {d2}")
        d3 = determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        print(f"  det([[6,1,1],[4,-2,5],[2,8,7]]) = {d3}")
    except NotImplementedError:
        print("TODO: implement determinant")
