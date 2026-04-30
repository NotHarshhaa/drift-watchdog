"""Data schema validation for drift detection."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SchemaIssue:
    """Schema validation issue."""
    
    issue_type: str  # "missing_column", "extra_column", "type_mismatch", "null_mismatch"
    feature_name: str
    expected: Any
    actual: Any
    severity: str  # "error", "warning"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type,
            "feature_name": self.feature_name,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "severity": self.severity,
        }


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""
    
    is_valid: bool
    issues: List[SchemaIssue]
    baseline_features: Set[str]
    current_features: Set[str]
    missing_features: Set[str]
    extra_features: Set[str]
    type_mismatches: Dict[str, Dict[str, str]]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "baseline_features": list(self.baseline_features),
            "current_features": list(self.current_features),
            "missing_features": list(self.missing_features),
            "extra_features": list(self.extra_features),
            "type_mismatches": self.type_mismatches,
            "timestamp": self.timestamp.isoformat(),
        }


class SchemaValidator:
    """Validate data schema consistency."""
    
    def __init__(self, strict: bool = False):
        """
        Initialize schema validator.
        
        Args:
            strict: If True, treat all mismatches as errors. If False, warnings for non-critical issues.
        """
        self.strict = strict
    
    def validate(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> SchemaValidationResult:
        """
        Validate current data schema against baseline.
        
        Args:
            baseline_data: Baseline dataset
            current_data: Current dataset
            
        Returns:
            SchemaValidationResult
        """
        issues = []
        
        # Get feature sets
        baseline_features = set(baseline_data.columns)
        current_features = set(current_data.columns)
        
        # Check for missing features
        missing_features = baseline_features - current_features
        for feature in missing_features:
            issues.append(SchemaIssue(
                issue_type="missing_column",
                feature_name=feature,
                expected=f"column '{feature}'",
                actual="not found",
                severity="error",
            ))
        
        # Check for extra features
        extra_features = current_features - baseline_features
        for feature in extra_features:
            severity = "error" if self.strict else "warning"
            issues.append(SchemaIssue(
                issue_type="extra_column",
                feature_name=feature,
                expected="not in baseline",
                actual=f"column '{feature}' found",
                severity=severity,
            ))
        
        # Check for type mismatches
        type_mismatches = {}
        common_features = baseline_features & current_features
        for feature in common_features:
            baseline_type = str(baseline_data[feature].dtype)
            current_type = str(current_data[feature].dtype)
            
            # Simplify type comparison
            baseline_simple = self._simplify_type(baseline_type)
            current_simple = self._simplify_type(current_type)
            
            if baseline_simple != current_simple:
                severity = "error" if self.strict else "warning"
                issues.append(SchemaIssue(
                    issue_type="type_mismatch",
                    feature_name=feature,
                    expected=baseline_type,
                    actual=current_type,
                    severity=severity,
                ))
                type_mismatches[feature] = {
                    "baseline": baseline_type,
                    "current": current_type,
                }
        
        # Check for null value pattern changes
        for feature in common_features:
            baseline_null_pct = baseline_data[feature].isna().mean()
            current_null_pct = current_data[feature].isna().mean()
            
            # If baseline had no nulls but current has significant nulls
            if baseline_null_pct == 0 and current_null_pct > 0:
                severity = "error" if self.strict else "warning"
                issues.append(SchemaIssue(
                    issue_type="null_mismatch",
                    feature_name=feature,
                    expected="no null values",
                    actual=f"{current_null_pct:.1%} null values",
                    severity=severity,
                ))
        
        # Determine if schema is valid
        error_count = sum(1 for issue in issues if issue.severity == "error")
        is_valid = error_count == 0
        
        return SchemaValidationResult(
            is_valid=is_valid,
            issues=issues,
            baseline_features=baseline_features,
            current_features=current_features,
            missing_features=missing_features,
            extra_features=extra_features,
            type_mismatches=type_mismatches,
        )
    
    def _simplify_type(self, dtype: str) -> str:
        """
        Simplify dtype string for comparison.
        
        Args:
            dtype: Data type string
            
        Returns:
            Simplified type string
        """
        dtype = dtype.lower()
        if "int" in dtype:
            return "int"
        elif "float" in dtype:
            return "float"
        elif "bool" in dtype:
            return "bool"
        elif "object" in dtype or "str" in dtype:
            return "str"
        elif "datetime" in dtype:
            return "datetime"
        else:
            return "other"
