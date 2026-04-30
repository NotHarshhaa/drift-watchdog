"""drift-watchdog: Lightweight ML model drift detection."""

__version__ = "1.2.0"

from drift_watchdog.detector import DriftDetector
from drift_watchdog.baseline import BaselineStore
from drift_watchdog.models import DriftResult, FeatureReport
from drift_watchdog.concept_drift import ConceptDriftDetector, ConceptDriftResult, ConceptDriftReport
from drift_watchdog.reporting import HTMLReportGenerator
from drift_watchdog.data_quality import DataQualityChecker, DataQualityResult, DataQualityReport
from drift_watchdog.trend_analysis import DriftTrendAnalyzer, TrendAnalysisResult

__all__ = [
    "DriftDetector",
    "BaselineStore",
    "DriftResult",
    "FeatureReport",
    "ConceptDriftDetector",
    "ConceptDriftResult",
    "ConceptDriftReport",
    "HTMLReportGenerator",
    "DataQualityChecker",
    "DataQualityResult",
    "DataQualityReport",
    "DriftTrendAnalyzer",
    "TrendAnalysisResult",
    "__version__",
]
