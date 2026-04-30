"""Data quality checks for drift detection."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DataQualityReport:
    """Report for data quality analysis."""
    
    feature_name: str
    missing_count: int
    missing_percentage: float
    outlier_count: int
    outlier_percentage: float
    unique_count: int
    unique_percentage: float
    is_issue: bool
    issue_severity: str  # "none", "low", "medium", "high"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "outlier_count": self.outlier_count,
            "outlier_percentage": self.outlier_percentage,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "is_issue": self.is_issue,
            "issue_severity": self.issue_severity,
        }


@dataclass
class DataQualityResult:
    """Overall data quality check result."""
    
    features: Dict[str, DataQualityReport]
    overall_quality_score: float
    has_issues: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": {name: report.to_dict() for name, report in self.features.items()},
            "overall_quality_score": self.overall_quality_score,
            "has_issues": self.has_issues,
            "timestamp": self.timestamp.isoformat(),
        }


class DataQualityChecker:
    """Check data quality for drift detection."""
    
    def __init__(
        self,
        missing_threshold: float = 0.1,
        outlier_threshold: float = 0.05,
        outlier_method: str = "iqr",
    ):
        """
        Initialize data quality checker.
        
        Args:
            missing_threshold: Threshold for missing value percentage (0-1)
            outlier_threshold: Threshold for outlier percentage (0-1)
            outlier_method: Method for outlier detection ("iqr", "zscore")
        """
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
    
    def check(self, data: pd.DataFrame) -> DataQualityResult:
        """
        Check data quality for all features.
        
        Args:
            data: Data to check
            
        Returns:
            DataQualityResult with detailed analysis
        """
        feature_reports = {}
        
        for feature in data.columns:
            report = self._check_feature(feature, data[feature])
            feature_reports[feature] = report
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(feature_reports)
        has_issues = any(report.is_issue for report in feature_reports.values())
        
        return DataQualityResult(
            features=feature_reports,
            overall_quality_score=overall_score,
            has_issues=has_issues,
        )
    
    def _check_feature(self, feature_name: str, feature_data: pd.Series) -> DataQualityReport:
        """
        Check quality for a single feature.
        
        Args:
            feature_name: Name of the feature
            feature_data: Feature data
            
        Returns:
            DataQualityReport
        """
        total_count = len(feature_data)
        
        # Check missing values
        missing_count = feature_data.isna().sum()
        missing_percentage = missing_count / total_count if total_count > 0 else 0.0
        
        # Check outliers (only for numeric data)
        if pd.api.types.is_numeric_dtype(feature_data):
            outlier_count = self._detect_outliers(feature_data)
            outlier_percentage = outlier_count / total_count if total_count > 0 else 0.0
        else:
            outlier_count = 0
            outlier_percentage = 0.0
        
        # Check unique values
        unique_count = feature_data.nunique()
        unique_percentage = unique_count / total_count if total_count > 0 else 0.0
        
        # Determine if there are issues
        is_issue = False
        issue_severity = "none"
        
        if missing_percentage > self.missing_threshold:
            is_issue = True
            if missing_percentage > 0.3:
                issue_severity = "high"
            elif missing_percentage > 0.2:
                issue_severity = "medium"
            else:
                issue_severity = "low"
        
        if outlier_percentage > self.outlier_threshold:
            is_issue = True
            if outlier_percentage > 0.15:
                issue_severity = "high"
            elif outlier_percentage > 0.1:
                issue_severity = "medium"
            elif issue_severity == "none":
                issue_severity = "low"
        
        return DataQualityReport(
            feature_name=feature_name,
            missing_count=int(missing_count),
            missing_percentage=float(missing_percentage),
            outlier_count=int(outlier_count),
            outlier_percentage=float(outlier_percentage),
            unique_count=int(unique_count),
            unique_percentage=float(unique_percentage),
            is_issue=is_issue,
            issue_severity=issue_severity,
        )
    
    def _detect_outliers(self, data: pd.Series) -> int:
        """
        Detect outliers in numeric data.
        
        Args:
            data: Numeric feature data
            
        Returns:
            Number of outliers
        """
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return 0
        
        if self.outlier_method == "iqr":
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((clean_data < lower_bound) | (clean_data > upper_bound)).sum()
        elif self.outlier_method == "zscore":
            z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
            outliers = (z_scores > 3).sum()
        else:
            outliers = 0
        
        return int(outliers)
    
    def _calculate_overall_score(self, feature_reports: Dict[str, DataQualityReport]) -> float:
        """
        Calculate overall quality score from feature reports.
        
        Args:
            feature_reports: Dictionary of feature reports
            
        Returns:
            Overall quality score (0-1, higher is better)
        """
        if not feature_reports:
            return 1.0
        
        # Calculate score based on missing and outlier percentages
        missing_scores = [1.0 - report.missing_percentage for report in feature_reports.values()]
        outlier_scores = [1.0 - report.outlier_percentage for report in feature_reports.values()]
        
        avg_missing_score = np.mean(missing_scores)
        avg_outlier_score = np.mean(outlier_scores)
        
        # Combine scores
        overall_score = (avg_missing_score + avg_outlier_score) / 2
        return float(overall_score)
