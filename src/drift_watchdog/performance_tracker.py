"""Model performance metrics tracking."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class PerformanceMetrics:
    """Model performance metrics at a point in time."""
    
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class PerformanceTrendResult:
    """Result of performance trend analysis."""
    
    current_metrics: PerformanceMetrics
    baseline_metrics: PerformanceMetrics
    accuracy_change: float
    precision_change: float
    recall_change: float
    f1_change: float
    auc_roc_change: float
    is_degradation: bool
    degradation_severity: str
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_metrics": self.current_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "accuracy_change": self.accuracy_change,
            "precision_change": self.precision_change,
            "recall_change": self.recall_change,
            "f1_change": self.f1_change,
            "auc_roc_change": self.auc_roc_change,
            "is_degradation": self.is_degradation,
            "degradation_severity": self.degradation_severity,
            "recommendation": self.recommendation,
        }


class PerformanceTracker:
    """Track model performance metrics over time."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize performance tracker.
        
        Args:
            max_history: Maximum number of historical points to keep
        """
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC-ROC)
            
        Returns:
            PerformanceMetrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate AUC-ROC if probabilities are provided
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                auc_roc = roc_auc_score(y_true, y_proba)
            except:
                auc_roc = 0.0
        else:
            auc_roc = 0.0
        
        # Calculate confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            confusion_dict = {
                "true_positives": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
                "false_positives": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                "true_negatives": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                "false_negatives": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            }
        except:
            confusion_dict = None
        
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc_roc=float(auc_roc),
            confusion_matrix=confusion_dict,
        )
        
        self.history.append(metrics)
        return metrics
    
    def compare_to_baseline(
        self,
        baseline_metrics: PerformanceMetrics,
        degradation_threshold: float = 0.05,
    ) -> PerformanceTrendResult:
        """
        Compare current performance to baseline.
        
        Args:
            baseline_metrics: Baseline performance metrics
            degradation_threshold: Threshold for degradation detection
            
        Returns:
            PerformanceTrendResult
        """
        if not self.history:
            raise ValueError("No performance history available")
        
        current = self.history[-1]
        
        # Calculate changes
        accuracy_change = current.accuracy - baseline_metrics.accuracy
        precision_change = current.precision - baseline_metrics.precision
        recall_change = current.recall - baseline_metrics.recall
        f1_change = current.f1_score - baseline_metrics.f1_score
        auc_roc_change = current.auc_roc - baseline_metrics.auc_roc
        
        # Determine if degradation occurred
        is_degradation = (
            accuracy_change < -degradation_threshold or
            f1_change < -degradation_threshold
        )
        
        # Determine severity
        if is_degradation:
            if accuracy_change < -0.15 or f1_change < -0.15:
                severity = "severe"
            elif accuracy_change < -0.10 or f1_change < -0.10:
                severity = "moderate"
            else:
                severity = "slight"
        else:
            severity = "none"
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_degradation,
            severity,
            accuracy_change,
            f1_change,
        )
        
        return PerformanceTrendResult(
            current_metrics=current,
            baseline_metrics=baseline_metrics,
            accuracy_change=accuracy_change,
            precision_change=precision_change,
            recall_change=recall_change,
            f1_change=f1_change,
            auc_roc_change=auc_roc_change,
            is_degradation=is_degradation,
            degradation_severity=severity,
            recommendation=recommendation,
        )
    
    def _generate_recommendation(
        self,
        is_degradation: bool,
        severity: str,
        accuracy_change: float,
        f1_change: float,
    ) -> str:
        """Generate recommendation based on performance change."""
        if not is_degradation:
            if accuracy_change > 0.05 or f1_change > 0.05:
                return "EXCELLENT: Model performance has improved significantly."
            else:
                return "GOOD: Model performance is stable or slightly improved."
        
        if severity == "severe":
            return "CRITICAL: Significant performance degradation detected. Immediate retraining recommended."
        elif severity == "moderate":
            return "WARNING: Moderate performance degradation. Consider retraining soon."
        else:
            return "INFO: Slight performance degradation detected. Monitor closely."
    
    def get_history(self) -> List[PerformanceMetrics]:
        """Get the full performance history."""
        return list(self.history)
    
    def clear_history(self) -> None:
        """Clear all historical data."""
        self.history.clear()
