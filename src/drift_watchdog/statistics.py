"""Statistical methods for drift detection."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from typing import Tuple, Optional, Dict
import warnings


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI < 0.1: stable
    0.1 <= PSI < 0.2: slight drift
    0.2 <= PSI < 0.25: moderate drift
    PSI >= 0.25: severe drift
    
    Args:
        expected: Reference distribution
        actual: Current distribution
        bins: Number of bins for histogram
        epsilon: Small value to avoid log(0)
        
    Returns:
        PSI value
    """
    # Remove NaN values
    expected_clean = expected[~np.isnan(expected)]
    actual_clean = actual[~np.isnan(actual)]
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0
    
    # Determine bin edges from expected distribution
    min_val = min(np.min(expected_clean), np.min(actual_clean))
    max_val = max(np.max(expected_clean), np.max(actual_clean))
    
    if min_val == max_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate histograms
    expected_counts, _ = np.histogram(expected_clean, bins=bin_edges)
    actual_counts, _ = np.histogram(actual_clean, bins=bin_edges)
    
    # Convert to percentages
    expected_perc = expected_counts / len(expected_clean) + epsilon
    actual_perc = actual_counts / len(actual_clean) + epsilon
    
    # Calculate PSI
    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    
    return float(psi)


def calculate_ks_test(
    expected: np.ndarray,
    actual: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov test statistic and p-value.
    
    Args:
        expected: Reference distribution
        actual: Current distribution
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    expected_clean = expected[~np.isnan(expected)]
    actual_clean = actual[~np.isnan(actual)]
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0, 1.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, pvalue = stats.ks_2samp(expected_clean, actual_clean)
    
    return float(statistic), float(pvalue)


def calculate_jensen_shannon(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    Args:
        expected: Reference distribution
        actual: Current distribution
        bins: Number of bins for histogram
        
    Returns:
        Jensen-Shannon divergence value
    """
    expected_clean = expected[~np.isnan(expected)]
    actual_clean = actual[~np.isnan(actual)]
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0
    
    min_val = min(np.min(expected_clean), np.min(actual_clean))
    max_val = max(np.max(expected_clean), np.max(actual_clean))
    
    if min_val == max_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    expected_counts, _ = np.histogram(expected_clean, bins=bin_edges)
    actual_counts, _ = np.histogram(actual_clean, bins=bin_edges)
    
    # Convert to probabilities
    expected_prob = expected_counts / len(expected_clean)
    actual_prob = actual_counts / len(actual_clean)
    
    # Add small epsilon to avoid zeros
    expected_prob = expected_prob + 1e-10
    actual_prob = actual_prob + 1e-10
    
    # Calculate Jensen-Shannon divergence
    m = (expected_prob + actual_prob) / 2
    js = 0.5 * (stats.entropy(expected_prob, m) + stats.entropy(actual_prob, m))
    
    return float(js)


def calculate_wasserstein(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    Calculate Wasserstein (Earth Mover's) distance between distributions.
    
    Args:
        expected: Reference distribution
        actual: Current distribution
        
    Returns:
        Wasserstein distance
    """
    expected_clean = expected[~np.isnan(expected)]
    actual_clean = actual[~np.isnan(actual)]
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0
    
    w_dist = stats.wasserstein_distance(expected_clean, actual_clean)
    return float(w_dist)


def calculate_chi_squared(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> Tuple[float, float]:
    """
    Calculate Chi-squared test statistic and p-value.
    
    Args:
        expected: Reference distribution
        actual: Current distribution
        bins: Number of bins for histogram
        
    Returns:
        Tuple of (chi-squared statistic, p-value)
    """
    expected_clean = expected[~np.isnan(expected)]
    actual_clean = actual[~np.isnan(actual)]
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0, 1.0
    
    min_val = min(np.min(expected_clean), np.min(actual_clean))
    max_val = max(np.max(expected_clean), np.max(actual_clean))
    
    if min_val == max_val:
        return 0.0, 1.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    expected_counts, _ = np.histogram(expected_clean, bins=bin_edges)
    actual_counts, _ = np.histogram(actual_clean, bins=bin_edges)
    
    # Ensure no zero counts for expected
    expected_counts = np.maximum(expected_counts, 1)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, pvalue = stats.chisquare(actual_counts, expected_counts)
        return float(statistic), float(pvalue)
    except ValueError:
        # If chi-squared test fails, return default values
        return 0.0, 1.0


def calculate_feature_statistics(
    data: pd.Series,
    bins: int = 10,
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a feature.
    
    Args:
        data: Feature data
        bins: Number of bins for histogram
        
    Returns:
        Dictionary of statistics
    """
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "histogram_bins": [],
            "histogram_counts": [],
        }
    
    # Calculate histogram for better distribution preservation
    counts, bin_edges = np.histogram(clean_data, bins=bins)
    
    return {
        "mean": float(clean_data.mean()),
        "std": float(clean_data.std()),
        "min": float(clean_data.min()),
        "max": float(clean_data.max()),
        "median": float(clean_data.median()),
        "q25": float(clean_data.quantile(0.25)),
        "q75": float(clean_data.quantile(0.75)),
        "histogram_bins": bin_edges.tolist(),
        "histogram_counts": counts.tolist(),
    }
