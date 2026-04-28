"""Tests for baseline storage."""

import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path

from drift_watchdog.baseline import BaselineStore
from drift_watchdog.models import Baseline


def test_create_baseline_from_dataframe():
    """Test creating baseline from DataFrame."""
    np = pytest.importorskip("numpy")
    np.random.seed(42)
    
    df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 2, 100),
        "id": range(100),
    })
    
    baseline = BaselineStore.create_from_dataframe(
        df,
        "test-baseline",
        exclude_features=["id"],
    )
    
    assert baseline.name == "test-baseline"
    assert "feature1" in baseline.feature_names
    assert "feature2" in baseline.feature_names
    assert "id" not in baseline.feature_names
    assert "feature1" in baseline.statistics


def test_local_storage_save_load():
    """Test saving and loading baseline from local storage."""
    np = pytest.importorskip("numpy")
    np.random.seed(42)
    
    df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
    })
    
    baseline = BaselineStore.create_from_dataframe(df, "test-baseline")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "baseline.json"
        store = BaselineStore(str(path), storage_type="local")
        
        # Save
        store.save(baseline)
        assert path.exists()
        
        # Load
        loaded_baseline = store.load()
        assert loaded_baseline.name == baseline.name
        assert loaded_baseline.feature_names == baseline.feature_names


def test_baseline_serialization():
    """Test baseline to/from dict serialization."""
    np = pytest.importorskip("numpy")
    np.random.seed(42)
    
    df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
    })
    
    baseline = BaselineStore.create_from_dataframe(df, "test-baseline")
    
    # To dict
    data = baseline.to_dict()
    assert "name" in data
    assert "statistics" in data
    assert "feature_names" in data
    
    # From dict
    loaded = Baseline.from_dict(data)
    assert loaded.name == baseline.name
    assert loaded.feature_names == baseline.feature_names
