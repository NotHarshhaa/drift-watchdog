"""drift-watchdog: Lightweight ML model drift detection."""

__version__ = "1.3.0"

from drift_watchdog.detector import DriftDetector
from drift_watchdog.baseline import BaselineStore
from drift_watchdog.models import DriftResult, FeatureReport
from drift_watchdog.concept_drift import ConceptDriftDetector, ConceptDriftResult, ConceptDriftReport
from drift_watchdog.reporting import HTMLReportGenerator
from drift_watchdog.data_quality import DataQualityChecker, DataQualityResult, DataQualityReport
from drift_watchdog.trend_analysis import DriftTrendAnalyzer, TrendAnalysisResult
from drift_watchdog.performance_tracker import PerformanceTracker, PerformanceMetrics, PerformanceTrendResult
from drift_watchdog.correlation_analysis import CorrelationAnalyzer, CorrelationAnalysisResult
from drift_watchdog.schema_validator import SchemaValidator, SchemaValidationResult
from drift_watchdog.drift_explainer import DriftExplainer, DriftExplanationResult

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
    "PerformanceTracker",
    "PerformanceMetrics",
    "PerformanceTrendResult",
    "CorrelationAnalyzer",
    "CorrelationAnalysisResult",
    "SchemaValidator",
    "SchemaValidationResult",
    "DriftExplainer",
    "DriftExplanationResult",
    "__version__",
]
