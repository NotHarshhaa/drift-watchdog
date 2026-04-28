"""Tests for drift detector."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from drift_watchdog.detector import DriftDetector
from drift_watchdog.baseline import BaselineStore
from drift_watchdog.models import Baseline


def test_detector_no_drift():
    """Test detector with similar distributions."""
    np.random.seed(42)
    
    # Create reference data
    ref_df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(5, 2, 1000),
    })
    
    # Create baseline
    baseline = BaselineStore.create_from_dataframe(ref_df, "test-baseline")
    
    # Create similar current data
    current_df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(5, 2, 1000),
    })
    
    # Run detector
    detector = DriftDetector(baseline=baseline, psi_threshold=0.2)
    result = detector.check(current_df)
    
    # Should not detect drift
    assert not result.overall_drift
    assert result.overall_score < 0.2


def test_detector_with_drift():
    """Test detector with drifted distributions."""
    np.random.seed(42)
    
    # Create reference data
    ref_df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(5, 2, 1000),
    })
    
    # Create baseline
    baseline = BaselineStore.create_from_dataframe(ref_df, "test-baseline")
    
    # Create drifted current data
    current_df = pd.DataFrame({
        "feature1": np.random.normal(3, 1, 1000),  # Shifted mean
        "feature2": np.random.normal(5, 2, 1000),
    })
    
    # Run detector
    detector = DriftDetector(baseline=baseline, psi_threshold=0.2)
    result = detector.check(current_df)
    
    # Should detect drift
    assert result.overall_drift
    assert result.overall_score > 0.2


def test_detector_exclude_features():
    """Test detector with excluded features."""
    np.random.seed(42)
    
    # Create reference data
    ref_df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(5, 2, 1000),
        "id": range(1000),
    })
    
    # Create baseline (excluding id)
    baseline = BaselineStore.create_from_dataframe(
        ref_df,
        "test-baseline",
        exclude_features=["id"],
    )
    
    # Create drifted current data
    current_df = pd.DataFrame({
        "feature1": np.random.normal(3, 1, 1000),
        "feature2": np.random.normal(5, 2, 1000),
        "id": range(1000),
    })
    
    # Run detector
    detector = DriftDetector(baseline=baseline, psi_threshold=0.2)
    result = detector.check(current_df, exclude_features=["id"])
    
    # Should detect drift but not include id
    assert "id" not in result.features
    assert "feature1" in result.features


def test_detector_methods():
    """Test detector with different methods."""
    np.random.seed(42)
    
    ref_df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
    })
    
    baseline = BaselineStore.create_from_dataframe(ref_df, "test-baseline")
    
    current_df = pd.DataFrame({
        "feature1": np.random.normal(3, 1, 1000),
    })
    
    # Test with only PSI
    detector = DriftDetector(baseline=baseline, psi_threshold=0.2, methods=["psi"])
    result = detector.check(current_df)
    assert len(result.features) == 1
    
    # Test with multiple methods
    detector = DriftDetector(
        baseline=baseline,
        psi_threshold=0.2,
        methods=["psi", "ks_test", "jensen_shannon"],
    )
    result = detector.check(current_df)
    assert len(result.features) == 1
