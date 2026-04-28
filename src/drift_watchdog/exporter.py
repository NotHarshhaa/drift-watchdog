"""Prometheus metrics exporter."""

import time
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from prometheus_client import (
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from drift_watchdog.models import DriftResult


class PrometheusExporter:
    """Export drift metrics to Prometheus."""
    
    def __init__(self, port: int = 9090):
        """
        Initialize Prometheus exporter.
        
        Args:
            port: Port to serve metrics on
        """
        self.port = port
        
        # Define metrics
        self.psi_gauge = Gauge(
            "drift_watchdog_psi",
            "Population Stability Index per feature",
            ["feature", "model", "baseline_version"],
        )
        
        self.ks_statistic_gauge = Gauge(
            "drift_watchdog_ks_statistic",
            "KS-test statistic per feature",
            ["feature", "model", "baseline_version"],
        )
        
        self.feature_drift_gauge = Gauge(
            "drift_watchdog_feature_drift",
            "1 if drift detected for feature, 0 otherwise",
            ["feature", "model", "baseline_version"],
        )
        
        self.overall_drift_gauge = Gauge(
            "drift_watchdog_overall_drift",
            "1 if any feature is drifting, 0 otherwise",
            ["model", "baseline_version"],
        )
        
        self.check_duration_histogram = Histogram(
            "drift_watchdog_check_duration_seconds",
            "Time taken per drift check",
        )
        
        self.last_check_timestamp_gauge = Gauge(
            "drift_watchdog_last_check_timestamp",
            "Unix timestamp of last check",
            ["model", "baseline_version"],
        )
    
    def update_metrics(self, result: DriftResult, model_name: str = "default") -> None:
        """
        Update Prometheus metrics with drift result.
        
        Args:
            result: Drift detection result
            model_name: Name of the model
        """
        labels = {
            "model": model_name,
            "baseline_version": result.baseline_version or "unknown",
        }
        
        # Update per-feature metrics
        for feature_name, report in result.features.items():
            feature_labels = {**labels, "feature": feature_name}
            
            self.psi_gauge.labels(**feature_labels).set(report.psi)
            self.ks_statistic_gauge.labels(**feature_labels).set(report.ks_statistic)
            self.feature_drift_gauge.labels(**feature_labels).set(1 if report.is_drift else 0)
        
        # Update overall metrics
        self.overall_drift_gauge.labels(**labels).set(1 if result.overall_drift else 0)
        self.last_check_timestamp_gauge.labels(**labels).set(result.timestamp.timestamp())
    
    def serve_forever(self) -> None:
        """Start the Prometheus HTTP server."""
        class MetricsHandler(BaseHTTPRequestHandler):
            def __init__(self, exporter, *args, **kwargs):
                self.exporter = exporter
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                    self.end_headers()
                    self.wfile.write(generate_latest())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress default logging
        
        def handler(*args, **kwargs):
            return MetricsHandler(self, *args, **kwargs)
        
        server = HTTPServer(("0.0.0.0", self.port), handler)
        print(f"Prometheus metrics server running on port {self.port}")
        print(f"Metrics available at http://localhost:{self.port}/metrics")
        server.serve_forever()
    
    def check_with_timing(self, check_func, *args, **kwargs) -> DriftResult:
        """
        Run drift check with timing.
        
        Args:
            check_func: Function to run
            *args: Positional arguments for check_func
            **kwargs: Keyword arguments for check_func
            
        Returns:
            DriftResult from check_func
        """
        with self.check_duration_histogram.time():
            result = check_func(*args, **kwargs)
        return result
