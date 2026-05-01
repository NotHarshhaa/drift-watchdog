"""Drift detection engine."""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from drift_watchdog.models import Baseline, DriftResult, FeatureReport
from drift_watchdog.statistics import (
    calculate_psi,
    calculate_ks_test,
    calculate_jensen_shannon,
    calculate_wasserstein,
    calculate_chi_squared,
)


class DriftDetector:
    """Detect drift in model inputs/outputs."""
    
    def __init__(
        self,
        baseline: Baseline,
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
        js_threshold: float = 0.1,
        methods: Optional[list[str]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize drift detector.
        
        Args:
            baseline: Reference baseline
            psi_threshold: PSI threshold for drift detection
            ks_pvalue_threshold: KS test p-value threshold
            js_threshold: Jensen-Shannon divergence threshold
            methods: Detection methods to use (psi, ks_test, jensen_shannon, wasserstein, chi_squared)
            feature_importance: Dictionary mapping feature names to importance weights (0-1)
            custom_thresholds: Dictionary mapping feature names to custom PSI thresholds
        """
        self.baseline = baseline
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.js_threshold = js_threshold
        self.methods = methods or ["psi", "ks_test", "jensen_shannon"]
        self.feature_importance = feature_importance or {}
        self.custom_thresholds = custom_thresholds or {}
    
    def check(
        self,
        current_data: pd.DataFrame,
        exclude_features: Optional[list[str]] = None,
    ) -> DriftResult:
        """
        Check for drift in current data.
        
        Args:
            current_data: Current data to check
            exclude_features: Features to exclude from check
            
        Returns:
            DriftResult with detailed analysis
        """
        exclude_features = exclude_features or []
        feature_reports = {}
        
        for feature in self.baseline.feature_names:
            if feature not in current_data.columns or feature in exclude_features:
                continue
            
            if feature not in self.baseline.statistics:
                continue
            
            feature_report = self._check_feature(
                feature,
                current_data[feature].values,
            )
            feature_reports[feature] = feature_report
        
        # Calculate overall drift score
        overall_score = self._calculate_overall_score(feature_reports)
        overall_drift = overall_score >= self.psi_threshold
        
        return DriftResult(
            features=feature_reports,
            overall_drift=overall_drift,
            overall_score=overall_score,
            baseline_version=self.baseline.version,
        )
    
    def _check_feature(
        self,
        feature_name: str,
        current_values: np.ndarray,
    ) -> FeatureReport:
        """
        Check drift for a single feature.
        
        Args:
            feature_name: Name of the feature
            current_values: Current feature values
            
        Returns:
            FeatureReport with drift analysis
        """
        baseline_stats = self.baseline.statistics[feature_name]
        
        # Use histogram data from baseline if available, otherwise fall back to synthetic generation
        if "histogram_bins" in baseline_stats and "histogram_counts" in baseline_stats:
            expected_values = self._generate_from_histogram(
                baseline_stats["histogram_bins"],
                baseline_stats["histogram_counts"],
                len(current_values)
            )
        else:
            # Fallback to synthetic distribution generation for backward compatibility
            expected_values = self._generate_expected_distribution(baseline_stats, len(current_values))
        
        # Calculate all statistical measures
        psi = calculate_psi(expected_values, current_values)
        ks_stat, ks_pvalue = calculate_ks_test(expected_values, current_values)
        js = calculate_jensen_shannon(expected_values, current_values)
        wasserstein = calculate_wasserstein(expected_values, current_values)
        chi_stat, chi_pvalue = calculate_chi_squared(expected_values, current_values)
        
        # Use custom threshold if available, otherwise use default
        threshold = self.custom_thresholds.get(feature_name, self.psi_threshold)
        
        # Determine if drift is detected
        is_drift = False
        drift_severity = "none"
        
        if "psi" in self.methods and psi >= threshold:
            is_drift = True
            if psi >= 0.25:
                drift_severity = "severe"
            elif psi >= 0.2:
                drift_severity = "moderate"
            else:
                drift_severity = "slight"
        
        if "ks_test" in self.methods and ks_pvalue < self.ks_pvalue_threshold:
            is_drift = True
            if drift_severity == "none":
                drift_severity = "moderate"
        
        if "jensen_shannon" in self.methods and js >= self.js_threshold:
            is_drift = True
            if drift_severity == "none":
                drift_severity = "moderate"
        
        return FeatureReport(
            feature_name=feature_name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            jensen_shannon=js,
            wasserstein=wasserstein,
            chi_squared=chi_stat,
            chi_pvalue=chi_pvalue,
            is_drift=is_drift,
            drift_severity=drift_severity,
        )
    
    def _generate_expected_distribution(
        self,
        stats: Dict[str, float],
        n_samples: int,
    ) -> np.ndarray:
        """
        Generate synthetic distribution from baseline statistics.
        
        Args:
            stats: Baseline statistics
            n_samples: Number of samples to generate
            
        Returns:
            Synthetic expected values
        """
        # Use normal distribution as a simple approximation
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)
        
        if std == 0:
            std = 1
        
        return np.random.normal(mean, std, n_samples)
    
    def _generate_from_histogram(
        self,
        bins: list,
        counts: list,
        n_samples: int,
    ) -> np.ndarray:
        """
        Generate samples from histogram bins for better distribution preservation.
        
        Args:
            bins: Histogram bin edges
            counts: Histogram bin counts
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        if not bins or not counts or sum(counts) == 0:
            # Fallback to uniform distribution if histogram is invalid
            return np.random.uniform(0, 1, n_samples)
        
        # Normalize counts to probabilities
        total_count = sum(counts)
        probabilities = [c / total_count for c in counts]
        
        # Sample bin indices according to probabilities
        bin_indices = np.random.choice(len(counts), size=n_samples, p=probabilities)
        
        # Generate random values within each selected bin
        samples = []
        for idx in bin_indices:
            lower = bins[idx]
            upper = bins[idx + 1]
            samples.append(np.random.uniform(lower, upper))
        
        return np.array(samples)
    
    def _calculate_overall_score(self, feature_reports: Dict[str, FeatureReport]) -> float:
        """
        Calculate overall drift score from feature reports.
        
        Args:
            feature_reports: Dictionary of feature reports
            
        Returns:
            Overall drift score
        """
        if not feature_reports:
            return 0.0
        
        # Calculate weighted average using feature importance if available
        if self.feature_importance:
            total_weight = 0.0
            weighted_sum = 0.0
            
            for feature_name, report in feature_reports.items():
                weight = self.feature_importance.get(feature_name, 1.0)
                weighted_sum += report.psi * weight
                total_weight += weight
            
            return float(weighted_sum / total_weight) if total_weight > 0 else 0.0
        else:
            # Use mean PSI as overall score (default behavior)
            psi_values = [report.psi for report in feature_reports.values()]
            return float(np.mean(psi_values))
