"""
Problem: Exploratory Data Analysis

Compute descriptive statistics from scratch and implement basic data
visualization and outlier detection methods.

Tasks:
  (a) Compute descriptive statistics from scratch: mean, median, mode,
      standard deviation, quartiles (Q1, Q2, Q3)
  (b) Implement histogram binning (choose bin edges and count frequencies)
  (c) Detect outliers using the IQR method and z-score method
  (d) Compute the Pearson correlation coefficient between two variables
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_mean(data):
    """
    Compute the arithmetic mean from scratch (no np.mean).

    Parameters:
        data: np.ndarray, shape (n,)

    Returns:
        float: the mean
    """
    # TODO
    raise NotImplementedError


def compute_median(data):
    """
    Compute the median from scratch (no np.median).

    Parameters:
        data: np.ndarray, shape (n,)

    Returns:
        float: the median
    """
    # TODO
    raise NotImplementedError


def compute_mode(data):
    """
    Compute the mode (most frequent value) from scratch.

    For continuous data, this is most meaningful after binning. Here, assume
    data contains discrete or rounded values.

    Parameters:
        data: np.ndarray, shape (n,)

    Returns:
        The mode value (if there is a tie, return any one of the modes)
    """
    # TODO
    raise NotImplementedError


def compute_std(data, ddof=0):
    """
    Compute the standard deviation from scratch.

    Parameters:
        data: np.ndarray, shape (n,)
        ddof: int, delta degrees of freedom (0 for population, 1 for sample)

    Returns:
        float: the standard deviation
    """
    # TODO
    raise NotImplementedError


def compute_quartiles(data):
    """
    Compute Q1 (25th percentile), Q2 (50th percentile / median),
    and Q3 (75th percentile) from scratch.

    Use linear interpolation between data points.

    Parameters:
        data: np.ndarray, shape (n,)

    Returns:
        dict with keys 'Q1', 'Q2', 'Q3': float values
    """
    # TODO
    raise NotImplementedError


def histogram_binning(data, n_bins=10):
    """
    Compute histogram bin edges and counts from scratch.

    Divide the range [min, max] into n_bins equal-width bins.

    Parameters:
        data: np.ndarray, shape (n,)
        n_bins: int

    Returns:
        dict with keys:
            'bin_edges': np.ndarray, shape (n_bins + 1,)
            'counts': np.ndarray, shape (n_bins,), count per bin
    """
    # TODO
    raise NotImplementedError


def detect_outliers_iqr(data, factor=1.5):
    """
    Detect outliers using the IQR method.

    Outliers are points below Q1 - factor*IQR or above Q3 + factor*IQR.

    Parameters:
        data: np.ndarray, shape (n,)
        factor: float, multiplier for IQR (default 1.5)

    Returns:
        dict with keys:
            'lower_bound': float
            'upper_bound': float
            'outliers': np.ndarray, the outlier values
            'outlier_indices': np.ndarray, indices of outliers
    """
    # TODO
    raise NotImplementedError


def detect_outliers_zscore(data, threshold=3.0):
    """
    Detect outliers using the z-score method.

    Outliers are points with |z-score| > threshold.

    Parameters:
        data: np.ndarray, shape (n,)
        threshold: float

    Returns:
        dict with keys:
            'z_scores': np.ndarray, shape (n,)
            'outliers': np.ndarray, the outlier values
            'outlier_indices': np.ndarray, indices of outliers
    """
    # TODO
    raise NotImplementedError


def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient from scratch.

    r = sum((x_i - x_bar)(y_i - y_bar)) /
        sqrt(sum((x_i - x_bar)^2) * sum((y_i - y_bar)^2))

    Parameters:
        x: np.ndarray, shape (n,)
        y: np.ndarray, shape (n,)

    Returns:
        float: Pearson correlation coefficient in [-1, 1]
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    data = rng.normal(loc=50, scale=10, size=200)
    # Add a few outliers
    data = np.concatenate([data, np.array([120, -20, 110])])

    try:
        print(f"Mean: {compute_mean(data):.2f}")
    except NotImplementedError:
        print("TODO: implement compute_mean")

    try:
        print(f"Median: {compute_median(data):.2f}")
    except NotImplementedError:
        print("TODO: implement compute_median")

    try:
        rounded = np.round(data).astype(int)
        print(f"Mode: {compute_mode(rounded)}")
    except NotImplementedError:
        print("TODO: implement compute_mode")

    try:
        print(f"Std (population): {compute_std(data):.2f}")
        print(f"Std (sample): {compute_std(data, ddof=1):.2f}")
    except NotImplementedError:
        print("TODO: implement compute_std")

    try:
        q = compute_quartiles(data)
        print(f"Quartiles: Q1={q['Q1']:.2f}, Q2={q['Q2']:.2f}, Q3={q['Q3']:.2f}")
    except NotImplementedError:
        print("TODO: implement compute_quartiles")

    try:
        hist = histogram_binning(data, n_bins=15)
        print(f"\nHistogram: {hist['counts']}")
    except NotImplementedError:
        print("TODO: implement histogram_binning")

    try:
        result = detect_outliers_iqr(data)
        print(f"\nIQR outliers: {result['outliers']}")
        print(f"Bounds: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")
    except NotImplementedError:
        print("TODO: implement detect_outliers_iqr")

    try:
        result = detect_outliers_zscore(data)
        print(f"Z-score outliers: {result['outliers']}")
    except NotImplementedError:
        print("TODO: implement detect_outliers_zscore")

    try:
        x = rng.normal(0, 1, 100)
        y = 2 * x + rng.normal(0, 0.5, 100)
        r = pearson_correlation(x, y)
        print(f"\nPearson r (strong linear): {r:.4f}")
    except NotImplementedError:
        print("TODO: implement pearson_correlation")
