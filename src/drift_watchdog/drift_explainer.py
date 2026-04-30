"""Drift explanation generator."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from drift_watchdog.models import DriftResult, FeatureReport


@dataclass
class DriftExplanation:
    """Explanation for detected drift."""
    
    feature_name: str
    contribution_score: float
    drift_type: str  # "distribution_shift", "outlier_increase", "missing_values", "range_change"
    explanation: str
    suggested_action: str
    severity: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "contribution_score": self.contribution_score,
            "drift_type": self.drift_type,
            "explanation": self.explanation,
            "suggested_action": self.suggested_action,
            "severity": self.severity,
        }


@dataclass
class DriftExplanationResult:
    """Result of drift explanation analysis."""
    
    overall_explanation: str
    primary_drivers: List[DriftExplanation]
    secondary_drivers: List[DriftExplanation]
    summary_statistics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_explanation": self.overall_explanation,
            "primary_drivers": [d.to_dict() for d in self.primary_drivers],
            "secondary_drivers": [d.to_dict() for d in self.secondary_drivers],
            "summary_statistics": self.summary_statistics,
            "timestamp": self.timestamp.isoformat(),
        }


class DriftExplainer:
    """Generate explanations for detected drift."""
    
    def __init__(self):
        """Initialize drift explainer."""
        pass
    
    def explain(self, drift_result: DriftResult) -> DriftExplanationResult:
        """
        Generate explanations for detected drift.
        
        Args:
            drift_result: Drift detection result
            
        Returns:
            DriftExplanationResult
        """
        # Get drifted features
        drifted_features = [
            (name, report)
            for name, report in drift_result.features.items()
            if report.is_drift
        ]
        
        if not drifted_features:
            return DriftExplanationResult(
                overall_explanation="No drift detected. All features are within acceptable thresholds.",
                primary_drivers=[],
                secondary_drivers=[],
                summary_statistics={
                    "total_features": len(drift_result.features),
                    "drifted_features": 0,
                    "overall_score": drift_result.overall_score,
                },
            )
        
        # Sort by PSI score (highest first)
        drifted_features.sort(key=lambda x: x[1].psi, reverse=True)
        
        # Generate explanations for each drifted feature
        explanations = []
        for feature_name, report in drifted_features:
            explanation = self._explain_feature_drift(feature_name, report)
            explanations.append(explanation)
        
        # Split into primary and secondary drivers
        if len(explanations) > 3:
            primary_drivers = explanations[:3]
            secondary_drivers = explanations[3:]
        else:
            primary_drivers = explanations
            secondary_drivers = []
        
        # Generate overall explanation
        overall_explanation = self._generate_overall_explanation(
            drifted_features,
            drift_result.overall_score,
        )
        
        # Summary statistics
        summary_statistics = {
            "total_features": len(drift_result.features),
            "drifted_features": len(drifted_features),
            "overall_score": drift_result.overall_score,
            "max_psi": max(report.psi for _, report in drifted_features),
            "avg_psi": np.mean([report.psi for _, report in drifted_features]),
            "severe_drift_count": sum(
                1 for _, report in drifted_features
                if report.drift_severity == "severe"
            ),
        }
        
        return DriftExplanationResult(
            overall_explanation=overall_explanation,
            primary_drivers=primary_drivers,
            secondary_drivers=secondary_drivers,
            summary_statistics=summary_statistics,
        )
    
    def _explain_feature_drift(
        self,
        feature_name: str,
        report: FeatureReport,
    ) -> DriftExplanation:
        """
        Generate explanation for a single feature's drift.
        
        Args:
            feature_name: Name of the feature
            report: Feature drift report
            
        Returns:
            DriftExplanation
        """
        # Determine drift type
        drift_type = self._determine_drift_type(report)
        
        # Generate explanation
        explanation = self._generate_feature_explanation(feature_name, report, drift_type)
        
        # Generate suggested action
        suggested_action = self._generate_suggested_action(feature_name, report, drift_type)
        
        # Calculate contribution score (normalized PSI)
        contribution_score = min(report.psi / 0.5, 1.0)
        
        return DriftExplanation(
            feature_name=feature_name,
            contribution_score=contribution_score,
            drift_type=drift_type,
            explanation=explanation,
            suggested_action=suggested_action,
            severity=report.drift_severity,
        )
    
    def _determine_drift_type(self, report: FeatureReport) -> str:
        """Determine the type of drift based on statistics."""
        if report.ks_statistic > 0.5:
            return "distribution_shift"
        elif report.jensen_shannon > 0.2:
            return "distribution_shift"
        elif report.wasserstein > 1.0:
            return "range_change"
        else:
            return "distribution_shift"
    
    def _generate_feature_explanation(
        self,
        feature_name: str,
        report: FeatureReport,
        drift_type: str,
    ) -> str:
        """Generate explanation for a feature."""
        if drift_type == "distribution_shift":
            return (
                f"Feature '{feature_name}' shows significant distribution shift "
                f"(PSI={report.psi:.3f}). The current distribution differs substantially "
                f"from the baseline, indicating potential changes in data characteristics."
            )
        elif drift_type == "range_change":
            return (
                f"Feature '{feature_name}' exhibits range changes "
                f"(Wasserstein={report.wasserstein:.3f}). The scale or range "
                f"of values has shifted compared to the baseline."
            )
        else:
            return (
                f"Feature '{feature_name}' shows drift with PSI={report.psi:.3f} "
                f"and KS statistic={report.ks_statistic:.3f}."
            )
    
    def _generate_suggested_action(
        self,
        feature_name: str,
        report: FeatureReport,
        drift_type: str,
    ) -> str:
        """Generate suggested action for a drifted feature."""
        if report.drift_severity == "severe":
            return (
                f"Investigate '{feature_name}' immediately. Check upstream data pipelines "
                f"and consider retraining the model with recent data."
            )
        elif report.drift_severity == "moderate":
            return (
                f"Monitor '{feature_name}' closely. Review data sources and "
                f"prepare for potential model retraining."
            )
        else:
            return (
                f"Continue monitoring '{feature_name}'. Document the drift "
                f"pattern for future reference."
            )
    
    def _generate_overall_explanation(
        self,
        drifted_features: List[tuple],
        overall_score: float,
    ) -> str:
        """Generate overall drift explanation."""
        if not drifted_features:
            return "No significant drift detected."
        
        top_features = [name for name, _ in drifted_features[:3]]
        
        if overall_score >= 0.3:
            return (
                f"Significant drift detected (overall score: {overall_score:.3f}). "
                f"Primary contributors: {', '.join(top_features)}. "
                f"Immediate investigation and potential model retraining recommended."
            )
        elif overall_score >= 0.2:
            return (
                f"Moderate drift detected (overall score: {overall_score:.3f}). "
                f"Main affected features: {', '.join(top_features)}. "
                f"Monitor closely and prepare for intervention."
            )
        else:
            return (
                f"Slight drift detected (overall score: {overall_score:.3f}). "
                f"Affected features: {', '.join(top_features)}. "
                f"Continue monitoring and document patterns."
            )
