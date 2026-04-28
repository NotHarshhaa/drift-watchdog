"""drift-watchdog: Lightweight ML model drift detection."""

__version__ = "1.0.0"

from drift_watchdog.detector import DriftDetector
from drift_watchdog.baseline import BaselineStore
from drift_watchdog.models import DriftResult, FeatureReport

__all__ = [
    "DriftDetector",
    "BaselineStore",
    "DriftResult",
    "FeatureReport",
    "__version__",
]
