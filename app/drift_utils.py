import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """
    Stable Population Stability Index (PSI) implementation.
    - Uses equal-width bins (safer for synthetic + seasonal data)
    - Handles missing values
    - Avoids division-by-zero and infinite PSI
    - Returns a float PSI score
    """

    # Convert to pandas Series and drop NaN
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Handle degenerate case
    if expected.nunique() <= 1 or actual.nunique() <= 1:
        return 0.0   # no drift detectable

    # Determine min/max across both datasets
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    if min_val == max_val:
        return 0.0   # identical ranges = no drift

    # Create equal-width bins
    bins = np.linspace(min_val, max_val, buckets + 1)

    # Histogram counts for each bucket
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts   = np.histogram(actual,   bins=bins)[0]

    # Convert to proportions
    expected_perc = expected_counts / len(expected)
    actual_perc   = actual_counts   / len(actual)

    # Clip very small numbers to avoid log(0)
    expected_perc = np.clip(expected_perc, 0.0001, 1)
    actual_perc   = np.clip(actual_perc,   0.0001, 1)

    # Compute PSI for each bucket
    psi_values = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)

    # Return total PSI
    return float(np.sum(psi_values))