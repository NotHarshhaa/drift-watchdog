"""Data models for drift detection."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class FeatureReport:
    """Report for a single feature's drift analysis."""
    
    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    jensen_shannon: float
    wasserstein: float
    chi_squared: float
    chi_pvalue: float
    is_drift: bool
    drift_severity: str  # "none", "slight", "moderate", "severe"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "psi": self.psi,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "jensen_shannon": self.jensen_shannon,
            "wasserstein": self.wasserstein,
            "chi_squared": self.chi_squared,
            "chi_pvalue": self.chi_pvalue,
            "is_drift": self.is_drift,
            "drift_severity": self.drift_severity,
        }


@dataclass
class DriftResult:
    """Overall drift detection result."""
    
    features: Dict[str, FeatureReport]
    overall_drift: bool
    overall_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    baseline_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": {name: report.to_dict() for name, report in self.features.items()},
            "overall_drift": self.overall_drift,
            "overall_score": self.overall_score,
            "timestamp": self.timestamp.isoformat(),
            "baseline_version": self.baseline_version,
        }
    
    def alert(self) -> None:
        """Fire alert based on drift result."""
        from drift_watchdog.alerts import AlertManager
        manager = AlertManager()
        manager.send_alert(self)


@dataclass
class Baseline:
    """Baseline data for drift comparison."""
    
    name: str
    statistics: Dict[str, Any]
    feature_names: list[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "statistics": self.statistics,
            "feature_names": self.feature_names,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Baseline":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            statistics=data["statistics"],
            feature_names=data["feature_names"],
            created_at=datetime.fromisoformat(data["created_at"]),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )
