"""
Sparse Matrices
===============

Implement sparse matrix storage and operations, and compare performance
with dense representations.

Tasks
-----
1. CSR Sparse Matrix Format: Implement a Compressed Sparse Row (CSR) matrix
   class that stores a sparse matrix using three arrays: data (nonzero values),
   col_indices (column indices of nonzeros), and row_ptr (pointers to the start
   of each row in data). Support construction from a dense matrix.

2. Sparse Matrix-Vector Multiplication: Implement matrix-vector multiplication
   for your CSR format. Given a CSR matrix and a dense vector, compute the
   product without expanding to dense form.

3. Conjugate Gradient Solver: Implement the conjugate gradient method for solving
   Ax = b where A is symmetric positive-definite. Accept a function (or CSR matrix)
   for the matrix-vector product. Include convergence monitoring.

4. Dense vs Sparse Performance Comparison: Create a large sparse SPD matrix
   (e.g., discretized Laplacian), solve the system using both dense and sparse
   approaches, and compare memory usage estimates and iteration counts.
"""

import numpy as np


class CSRMatrix:
    """Compressed Sparse Row matrix format."""

    def __init__(self, data, col_indices, row_ptr, shape):
        """
        Initialize a CSR matrix.

        Parameters
        ----------
        data : np.ndarray
            Nonzero values, in row-major order.
        col_indices : np.ndarray
            Column index for each entry in data.
        row_ptr : np.ndarray
            row_ptr[i] is the index into data where row i begins.
            row_ptr has length n_rows + 1.
        shape : tuple
            (n_rows, n_cols).
        """
        raise NotImplementedError

    @classmethod
    def from_dense(cls, A):
        """
        Create a CSR matrix from a dense numpy array.

        Parameters
        ----------
        A : np.ndarray
            Dense matrix of shape (m, n).

        Returns
        -------
        CSRMatrix
        """
        raise NotImplementedError

    def to_dense(self):
        """
        Convert back to a dense numpy array.

        Returns
        -------
        A : np.ndarray
        """
        raise NotImplementedError

    def matvec(self, x):
        """
        Compute the matrix-vector product A @ x.

        Parameters
        ----------
        x : np.ndarray
            Vector of length n_cols.

        Returns
        -------
        y : np.ndarray
            Result vector of length n_rows.
        """
        raise NotImplementedError

    def nnz(self):
        """Return the number of nonzero entries."""
        raise NotImplementedError


def conjugate_gradient(A_matvec, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Solve Ax = b using the conjugate gradient method.

    Parameters
    ----------
    A_matvec : callable
        Function that computes the matrix-vector product A @ x.
    b : np.ndarray
        Right-hand side vector.
    x0 : np.ndarray or None
        Initial guess (zeros if None).
    tol : float
        Convergence tolerance on residual norm.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
        Approximate solution.
    residuals : list
        List of residual norms at each iteration.
    """
    raise NotImplementedError


def create_laplacian_1d(n):
    """
    Create the 1D discrete Laplacian matrix of size n x n as a CSR matrix.

    The matrix has 2 on the diagonal and -1 on the sub/superdiagonals.

    Parameters
    ----------
    n : int
        Matrix size.

    Returns
    -------
    CSRMatrix
    """
    raise NotImplementedError


def performance_comparison(n=1000):
    """
    Compare dense vs sparse solvers for the 1D Laplacian system.

    Parameters
    ----------
    n : int
        System size.

    Returns
    -------
    results : dict
        Dictionary with keys 'dense_time', 'sparse_time', 'sparse_iters',
        'dense_memory_bytes', 'sparse_memory_bytes'.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== CSR Matrix ===")
    A_dense = np.array([[1, 0, 0, 2],
                        [0, 0, 3, 0],
                        [4, 0, 0, 5],
                        [0, 6, 0, 0]], dtype=float)
    A_csr = CSRMatrix.from_dense(A_dense)
    print(f"Shape: {A_csr.shape}, NNZ: {A_csr.nnz()}")
    print(f"Reconstruction error: {np.linalg.norm(A_csr.to_dense() - A_dense):.2e}")

    print("\n=== Sparse Matrix-Vector Product ===")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_sparse = A_csr.matvec(x)
    y_dense = A_dense @ x
    print(f"Matvec error: {np.linalg.norm(y_sparse - y_dense):.2e}")

    print("\n=== Conjugate Gradient ===")
    L = create_laplacian_1d(100)
    b = np.ones(100)
    x_sol, residuals = conjugate_gradient(L.matvec, b)
    print(f"CG converged in {len(residuals)} iterations")
    print(f"Final residual: {residuals[-1]:.2e}")

    print("\n=== Performance Comparison ===")
    results = performance_comparison(n=500)
    for key, val in results.items():
        print(f"{key}: {val}")
