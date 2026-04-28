"""Tests for statistical methods."""

import numpy as np
import pytest

from drift_watchdog.statistics import (
    calculate_psi,
    calculate_ks_test,
    calculate_jensen_shannon,
    calculate_wasserstein,
    calculate_chi_squared,
    calculate_feature_statistics,
)


def test_calculate_psi():
    """Test PSI calculation."""
    # Same distribution should have low PSI
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.01
    
    # Different distributions should have higher PSI
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1, 1000)
    psi = calculate_psi(data1, data2)
    assert psi > 0.1


def test_calculate_ks_test():
    """Test KS test calculation."""
    # Same distribution should have high p-value
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    stat, pvalue = calculate_ks_test(data, data)
    assert pvalue > 0.05
    
    # Different distributions should have low p-value
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1, 1000)
    stat, pvalue = calculate_ks_test(data1, data2)
    assert pvalue < 0.05


def test_calculate_jensen_shannon():
    """Test Jensen-Shannon divergence calculation."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    js = calculate_jensen_shannon(data, data)
    assert js < 0.01
    
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1, 1000)
    js = calculate_jensen_shannon(data1, data2)
    assert js > 0.05


def test_calculate_wasserstein():
    """Test Wasserstein distance calculation."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    w = calculate_wasserstein(data, data)
    assert w < 0.1
    
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1, 1000)
    w = calculate_wasserstein(data1, data2)
    assert w > 1.0


def test_calculate_chi_squared():
    """Test chi-squared test calculation."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    stat, pvalue = calculate_chi_squared(data, data)
    # Should have high p-value for similar distributions
    assert pvalue > 0.01


def test_calculate_feature_statistics():
    """Test feature statistics calculation."""
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000)
    
    import pandas as pd
    series = pd.Series(data)
    stats = calculate_feature_statistics(series)
    
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "median" in stats
    assert "q25" in stats
    assert "q75" in stats
    
    # Check reasonable values
    assert 4 < stats["mean"] < 6
    assert stats["std"] > 0
