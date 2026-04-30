"""Feature correlation analysis for drift detection."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CorrelationChange:
    """Change in correlation between two features."""
    
    feature1: str
    feature2: str
    baseline_correlation: float
    current_correlation: float
    correlation_change: float
    is_significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature1": self.feature1,
            "feature2": self.feature2,
            "baseline_correlation": self.baseline_correlation,
            "current_correlation": self.current_correlation,
            "correlation_change": self.correlation_change,
            "is_significant": self.is_significant,
        }


@dataclass
class CorrelationAnalysisResult:
    """Result of correlation analysis."""
    
    correlation_changes: List[CorrelationChange]
    overall_correlation_drift: float
    significant_changes: int
    total_pairs: int
    is_drift_detected: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_changes": [c.to_dict() for c in self.correlation_changes],
            "overall_correlation_drift": self.overall_correlation_drift,
            "significant_changes": self.significant_changes,
            "total_pairs": self.total_pairs,
            "is_drift_detected": self.is_drift_detected,
            "timestamp": self.timestamp.isoformat(),
        }


class CorrelationAnalyzer:
    """Analyze changes in feature correlations."""
    
    def __init__(self, significance_threshold: float = 0.3):
        """
        Initialize correlation analyzer.
        
        Args:
            significance_threshold: Threshold for significant correlation change
        """
        self.significance_threshold = significance_threshold
    
    def analyze(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        method: str = "pearson",
    ) -> CorrelationAnalysisResult:
        """
        Analyze correlation changes between baseline and current data.
        
        Args:
            baseline_data: Baseline dataset
            current_data: Current dataset
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            CorrelationAnalysisResult
        """
        # Get common numeric features
        numeric_features = self._get_common_numeric_features(baseline_data, current_data)
        
        if len(numeric_features) < 2:
            return CorrelationAnalysisResult(
                correlation_changes=[],
                overall_correlation_drift=0.0,
                significant_changes=0,
                total_pairs=0,
                is_drift_detected=False,
            )
        
        # Calculate correlation matrices
        baseline_corr = baseline_data[numeric_features].corr(method=method)
        current_corr = current_data[numeric_features].corr(method=method)
        
        # Analyze correlation changes
        correlation_changes = []
        for i, feat1 in enumerate(numeric_features):
            for j, feat2 in enumerate(numeric_features):
                if i >= j:  # Avoid duplicates and self-correlations
                    continue
                
                baseline_val = baseline_corr.loc[feat1, feat2]
                current_val = current_corr.loc[feat1, feat2]
                
                # Handle NaN values
                if pd.isna(baseline_val) or pd.isna(current_val):
                    continue
                
                change = abs(current_val - baseline_val)
                is_significant = change >= self.significance_threshold
                
                correlation_changes.append(CorrelationChange(
                    feature1=feat1,
                    feature2=feat2,
                    baseline_correlation=float(baseline_val),
                    current_correlation=float(current_val),
                    correlation_change=float(change),
                    is_significant=is_significant,
                ))
        
        # Calculate overall correlation drift
        total_pairs = len(correlation_changes)
        significant_changes = sum(1 for c in correlation_changes if c.is_significant)
        
        if total_pairs > 0:
            overall_drift = np.mean([c.correlation_change for c in correlation_changes])
        else:
            overall_drift = 0.0
        
        # Determine if drift is detected
        is_drift_detected = significant_changes > (total_pairs * 0.1) if total_pairs > 0 else False
        
        return CorrelationAnalysisResult(
            correlation_changes=correlation_changes,
            overall_correlation_drift=float(overall_drift),
            significant_changes=significant_changes,
            total_pairs=total_pairs,
            is_drift_detected=is_drift_detected,
        )
    
    def _get_common_numeric_features(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> List[str]:
        """
        Get common numeric features between baseline and current data.
        
        Args:
            baseline_data: Baseline dataset
            current_data: Current dataset
            
        Returns:
            List of common numeric feature names
        """
        baseline_numeric = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
        current_numeric = current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        common_features = list(set(baseline_numeric) & set(current_numeric))
        return sorted(common_features)
    
    def get_top_changes(
        self,
        result: CorrelationAnalysisResult,
        top_n: int = 10,
    ) -> List[CorrelationChange]:
        """
        Get top correlation changes by magnitude.
        
        Args:
            result: Correlation analysis result
            top_n: Number of top changes to return
            
        Returns:
            List of top correlation changes
        """
        sorted_changes = sorted(
            result.correlation_changes,
            key=lambda x: x.correlation_change,
            reverse=True,
        )
        return sorted_changes[:top_n]
