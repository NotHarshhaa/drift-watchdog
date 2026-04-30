"""Concept drift detection for model outputs/labels."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from drift_watchdog.statistics import (
    calculate_psi,
    calculate_ks_test,
    calculate_jensen_shannon,
    calculate_chi_squared,
)


@dataclass
class ConceptDriftReport:
    """Report for concept drift analysis."""
    
    metric_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    jensen_shannon: float
    chi_squared: float
    chi_pvalue: float
    accuracy_change: float
    is_drift: bool
    drift_severity: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "psi": self.psi,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "jensen_shannon": self.jensen_shannon,
            "chi_squared": self.chi_squared,
            "chi_pvalue": self.chi_pvalue,
            "accuracy_change": self.accuracy_change,
            "is_drift": self.is_drift,
            "drift_severity": self.drift_severity,
        }


@dataclass
class ConceptDriftResult:
    """Overall concept drift detection result."""
    
    metrics: Dict[str, ConceptDriftReport]
    overall_drift: bool
    overall_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    baseline_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": {name: report.to_dict() for name, report in self.metrics.items()},
            "overall_drift": self.overall_drift,
            "overall_score": self.overall_score,
            "timestamp": self.timestamp.isoformat(),
            "baseline_version": self.baseline_version,
        }


class ConceptDriftDetector:
    """Detect concept drift in model outputs/labels."""
    
    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
        js_threshold: float = 0.1,
        accuracy_threshold: float = 0.05,
    ):
        """
        Initialize concept drift detector.
        
        Args:
            psi_threshold: PSI threshold for drift detection
            ks_pvalue_threshold: KS test p-value threshold
            js_threshold: Jensen-Shannon divergence threshold
            accuracy_threshold: Accuracy change threshold
        """
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.js_threshold = js_threshold
        self.accuracy_threshold = accuracy_threshold
    
    def check(
        self,
        baseline_predictions: np.ndarray,
        baseline_labels: np.ndarray,
        current_predictions: np.ndarray,
        current_labels: np.ndarray,
    ) -> ConceptDriftResult:
        """
        Check for concept drift in predictions and labels.
        
        Args:
            baseline_predictions: Reference predictions
            baseline_labels: Reference labels
            current_predictions: Current predictions
            current_labels: Current labels
            
        Returns:
            ConceptDriftResult with detailed analysis
        """
        metric_reports = {}
        
        # Check prediction distribution drift
        pred_report = self._check_distribution(
            "predictions",
            baseline_predictions,
            current_predictions,
        )
        metric_reports["predictions"] = pred_report
        
        # Check label distribution drift
        label_report = self._check_distribution(
            "labels",
            baseline_labels,
            current_labels,
        )
        metric_reports["labels"] = label_report
        
        # Check accuracy change
        baseline_accuracy = self._calculate_accuracy(baseline_predictions, baseline_labels)
        current_accuracy = self._calculate_accuracy(current_predictions, current_labels)
        accuracy_change = current_accuracy - baseline_accuracy
        
        accuracy_report = self._check_accuracy_drift(
            baseline_accuracy,
            current_accuracy,
            accuracy_change,
        )
        metric_reports["accuracy"] = accuracy_report
        
        # Calculate overall drift score
        overall_score = self._calculate_overall_score(metric_reports)
        overall_drift = overall_score >= self.psi_threshold
        
        return ConceptDriftResult(
            metrics=metric_reports,
            overall_drift=overall_drift,
            overall_score=overall_score,
        )
    
    def _check_distribution(
        self,
        metric_name: str,
        baseline_values: np.ndarray,
        current_values: np.ndarray,
    ) -> ConceptDriftReport:
        """
        Check distribution drift for a metric.
        
        Args:
            metric_name: Name of the metric
            baseline_values: Reference values
            current_values: Current values
            
        Returns:
            ConceptDriftReport
        """
        # Calculate statistical measures
        psi = calculate_psi(baseline_values, current_values)
        ks_stat, ks_pvalue = calculate_ks_test(baseline_values, current_values)
        js = calculate_jensen_shannon(baseline_values, current_values)
        chi_stat, chi_pvalue = calculate_chi_squared(baseline_values, current_values)
        
        # Determine if drift is detected
        is_drift = False
        drift_severity = "none"
        
        if psi >= self.psi_threshold:
            is_drift = True
            if psi >= 0.25:
                drift_severity = "severe"
            elif psi >= 0.2:
                drift_severity = "moderate"
            else:
                drift_severity = "slight"
        
        if ks_pvalue < self.ks_pvalue_threshold:
            is_drift = True
            if drift_severity == "none":
                drift_severity = "moderate"
        
        if js >= self.js_threshold:
            is_drift = True
            if drift_severity == "none":
                drift_severity = "moderate"
        
        return ConceptDriftReport(
            metric_name=metric_name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            jensen_shannon=js,
            chi_squared=chi_stat,
            chi_pvalue=chi_pvalue,
            accuracy_change=0.0,
            is_drift=is_drift,
            drift_severity=drift_severity,
        )
    
    def _check_accuracy_drift(
        self,
        baseline_accuracy: float,
        current_accuracy: float,
        accuracy_change: float,
    ) -> ConceptDriftReport:
        """
        Check accuracy drift.
        
        Args:
            baseline_accuracy: Reference accuracy
            current_accuracy: Current accuracy
            accuracy_change: Change in accuracy
            
        Returns:
            ConceptDriftReport
        """
        is_drift = abs(accuracy_change) >= self.accuracy_threshold
        drift_severity = "none"
        
        if is_drift:
            if abs(accuracy_change) >= 0.1:
                drift_severity = "severe"
            elif abs(accuracy_change) >= 0.05:
                drift_severity = "moderate"
            else:
                drift_severity = "slight"
        
        return ConceptDriftReport(
            metric_name="accuracy",
            psi=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            jensen_shannon=0.0,
            chi_squared=0.0,
            chi_pvalue=1.0,
            accuracy_change=accuracy_change,
            is_drift=is_drift,
            drift_severity=drift_severity,
        )
    
    def _calculate_accuracy(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Calculate accuracy from predictions and labels.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Accuracy score
        """
        if len(predictions) != len(labels):
            return 0.0
        
        correct = np.sum(predictions == labels)
        return float(correct / len(predictions)) if len(predictions) > 0 else 0.0
    
    def _calculate_overall_score(
        self,
        metric_reports: Dict[str, ConceptDriftReport],
    ) -> float:
        """
        Calculate overall drift score from metric reports.
        
        Args:
            metric_reports: Dictionary of metric reports
            
        Returns:
            Overall drift score
        """
        if not metric_reports:
            return 0.0
        
        # Use mean PSI as overall score (excluding accuracy which doesn't have PSI)
        psi_values = [
            report.psi for report in metric_reports.values()
            if report.metric_name != "accuracy"
        ]
        
        return float(np.mean(psi_values)) if psi_values else 0.0
