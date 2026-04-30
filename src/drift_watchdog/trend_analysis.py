"""Drift trend analysis over time."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class TrendPoint:
    """A single point in drift trend history."""
    
    timestamp: datetime
    overall_score: float
    feature_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "feature_scores": self.feature_scores,
        }


@dataclass
class TrendAnalysisResult:
    """Result of trend analysis."""
    
    overall_trend: str  # "increasing", "decreasing", "stable"
    trend_slope: float
    trend_strength: float
    feature_trends: Dict[str, str]
    history: List[TrendPoint]
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_trend": self.overall_trend,
            "trend_slope": self.trend_slope,
            "trend_strength": self.trend_strength,
            "feature_trends": self.feature_trends,
            "history": [point.to_dict() for point in self.history],
            "recommendation": self.recommendation,
        }


class DriftTrendAnalyzer:
    """Analyze drift trends over time."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize trend analyzer.
        
        Args:
            max_history: Maximum number of historical points to keep
        """
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
    
    def add_point(self, overall_score: float, feature_scores: Dict[str, float]) -> None:
        """
        Add a new drift check point to history.
        
        Args:
            overall_score: Overall drift score
            feature_scores: Dictionary of feature drift scores
        """
        point = TrendPoint(
            timestamp=datetime.utcnow(),
            overall_score=overall_score,
            feature_scores=feature_scores.copy(),
        )
        self.history.append(point)
    
    def analyze(self) -> TrendAnalysisResult:
        """
        Analyze drift trends from history.
        
        Returns:
            TrendAnalysisResult with trend analysis
        """
        if len(self.history) < 3:
            return TrendAnalysisResult(
                overall_trend="insufficient_data",
                trend_slope=0.0,
                trend_strength=0.0,
                feature_trends={},
                history=list(self.history),
                recommendation="Need more data points for trend analysis",
            )
        
        # Extract scores over time
        scores = [point.overall_score for point in self.history]
        timestamps = [point.timestamp.timestamp() for point in self.history]
        
        # Calculate trend slope using linear regression
        trend_slope, trend_strength = self._calculate_trend(timestamps, scores)
        
        # Determine overall trend
        if trend_slope > 0.01:
            overall_trend = "increasing"
        elif trend_slope < -0.01:
            overall_trend = "decreasing"
        else:
            overall_trend = "stable"
        
        # Analyze feature trends
        feature_trends = {}
        if len(self.history) > 0:
            all_features = set()
            for point in self.history:
                all_features.update(point.feature_scores.keys())
            
            for feature in all_features:
                feature_scores = [
                    point.feature_scores.get(feature, 0.0)
                    for point in self.history
                ]
                slope, strength = self._calculate_trend(timestamps, feature_scores)
                
                if slope > 0.01:
                    feature_trends[feature] = "increasing"
                elif slope < -0.01:
                    feature_trends[feature] = "decreasing"
                else:
                    feature_trends[feature] = "stable"
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_trend,
            trend_strength,
            scores[-1] if scores else 0.0,
        )
        
        return TrendAnalysisResult(
            overall_trend=overall_trend,
            trend_slope=trend_slope,
            trend_strength=trend_strength,
            feature_trends=feature_trends,
            history=list(self.history),
            recommendation=recommendation,
        )
    
    def _calculate_trend(self, x: List[float], y: List[float]) -> tuple[float, float]:
        """
        Calculate trend slope and strength using linear regression.
        
        Args:
            x: X values (timestamps)
            y: Y values (scores)
            
        Returns:
            Tuple of (slope, strength)
        """
        if len(x) < 2:
            return 0.0, 0.0
        
        x = np.array(x)
        y = np.array(y)
        
        # Normalize x to avoid numerical issues
        x_normalized = (x - x.mean()) / (x.std() + 1e-10)
        
        # Linear regression
        slope = np.cov(x_normalized, y)[0, 1] / np.var(x_normalized)
        
        # Calculate R-squared as strength
        y_pred = slope * x_normalized + y.mean()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return float(slope), float(r_squared)
    
    def _generate_recommendation(
        self,
        trend: str,
        strength: float,
        current_score: float,
    ) -> str:
        """
        Generate recommendation based on trend analysis.
        
        Args:
            trend: Overall trend
            strength: Trend strength
            current_score: Current drift score
            
        Returns:
            Recommendation string
        """
        if trend == "insufficient_data":
            return "Collect more drift check data for reliable trend analysis"
        
        if trend == "increasing":
            if strength > 0.7:
                return "CRITICAL: Drift is steadily increasing. Investigate immediately and consider retraining."
            elif strength > 0.4:
                return "WARNING: Drift is increasing. Monitor closely and prepare for intervention."
            else:
                return "INFO: Slight upward trend in drift. Continue monitoring."
        
        elif trend == "decreasing":
            if strength > 0.7:
                return "GOOD: Drift is steadily decreasing. Model performance is improving."
            elif strength > 0.4:
                return "INFO: Downward trend in drift. Positive sign."
            else:
                return "INFO: Slight downward trend. Continue monitoring."
        
        else:  # stable
            if current_score > 0.2:
                return "WARNING: Drift is stable but elevated. Current level may require attention."
            else:
                return "GOOD: Drift is stable at acceptable levels."
    
    def get_history(self) -> List[TrendPoint]:
        """
        Get the full history of drift points.
        
        Returns:
            List of TrendPoint objects
        """
        return list(self.history)
    
    def clear_history(self) -> None:
        """Clear all historical data."""
        self.history.clear()
