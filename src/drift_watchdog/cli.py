"""Command-line interface for drift-watchdog."""

import sys
import time
import pandas as pd
import click
from pathlib import Path
from typing import Optional

from drift_watchdog.baseline import BaselineStore
from drift_watchdog.detector import DriftDetector
from drift_watchdog.exporter import PrometheusExporter
from drift_watchdog.config import Config
from drift_watchdog.alerts import AlertManager


@click.group()
@click.version_option(version="1.0.0")
def main():
    """drift-watchdog: Lightweight ML model drift detection."""
    pass


@main.group()
def baseline():
    """Baseline management commands."""
    pass


@baseline.command("create")
@click.option("--data", "-d", required=True, help="Path to reference data CSV file")
@click.option("--output", "-o", required=True, help="Output path for baseline JSON")
@click.option("--name", "-n", required=True, help="Baseline name")
@click.option("--exclude", "-e", multiple=True, help="Features to exclude")
@click.option("--storage", "-s", default="local", help="Storage type (local, s3, gcs)")
def create_baseline(data: str, output: str, name: str, exclude: tuple, storage: str):
    """Create a baseline from reference data."""
    click.echo(f"Creating baseline '{name}' from {data}...")
    
    try:
        baseline = BaselineStore.create_from_csv(
            csv_path=data,
            name=name,
            exclude_features=list(exclude) if exclude else None,
        )
        
        store = BaselineStore(path=output, storage_type=storage)
        store.save(baseline)
        
        click.echo(f"✓ Baseline saved to {output}")
        click.echo(f"  Features: {', '.join(baseline.feature_names)}")
    except Exception as e:
        click.echo(f"✗ Error creating baseline: {e}", err=True)
        sys.exit(1)


@main.command("check")
@click.option("--baseline", "-b", required=True, help="Path to baseline JSON")
@click.option("--current", "-c", required=True, help="Path to current data CSV")
@click.option("--threshold", "-t", default=0.2, type=float, help="Drift threshold")
@click.option("--config", "-f", help="Path to configuration file")
@click.option("--methods", "-m", multiple=True, help="Detection methods to use")
@click.option("--alert", is_flag=True, help="Send alerts if drift detected")
def check_drift(
    baseline: str,
    current: str,
    threshold: float,
    config: Optional[str],
    methods: tuple,
    alert: bool,
):
    """Run drift detection check."""
    # Load configuration
    cfg = Config(config) if config else None
    
    click.echo(f"Loading baseline from {baseline}...")
    try:
        store = BaselineStore(path=baseline, storage_type="auto")
        baseline_obj = store.load()
    except Exception as e:
        click.echo(f"✗ Error loading baseline: {e}", err=True)
        sys.exit(1)
    
    click.echo(f"Loading current data from {current}...")
    try:
        current_df = pd.read_csv(current)
    except Exception as e:
        click.echo(f"✗ Error loading current data: {e}", err=True)
        sys.exit(1)
    
    # Get configuration values
    if cfg:
        detection_methods = list(methods) if methods else cfg.detection_methods
        exclude_features = cfg.exclude_features
    else:
        detection_methods = list(methods) if methods else ["psi", "ks_test"]
        exclude_features = []
    
    click.echo(f"Running drift detection (methods: {', '.join(detection_methods)})...")
    
    detector = DriftDetector(
        baseline=baseline_obj,
        psi_threshold=threshold,
        methods=detection_methods,
    )
    
    result = detector.check(current_df, exclude_features=exclude_features)
    
    # Display results
    click.echo()
    for feature_name, report in result.features.items():
        if report.is_drift:
            if report.drift_severity == "severe":
                symbol = "✗"
            elif report.drift_severity == "moderate":
                symbol = "⚠"
            else:
                symbol = "⚠"
        else:
            symbol = "✓"
        
        click.echo(
            f"{symbol} feature: {feature_name:20s} PSI={report.psi:.2f}  "
            f"[{report.drift_severity.upper() if report.is_drift else 'OK'}]"
        )
    
    click.echo()
    click.echo(f"Overall drift score: {result.overall_score:.2f} — " +
               ("ALERT" if result.overall_drift else "OK"))
    
    # Send alerts if configured
    if alert and result.overall_drift:
        click.echo("Sending alerts...")
        alert_manager = AlertManager(config_path=config)
        if alert_manager.send_alert(result):
            click.echo("✓ Alerts sent")
        else:
            click.echo("✗ Failed to send alerts")
    
    # Exit with error code if drift detected
    sys.exit(1 if result.overall_drift else 0)


@main.command("serve")
@click.option("--baseline", "-b", required=True, help="Path to baseline JSON")
@click.option("--data-source", "-d", required=True, help="Path or URI to data source")
@click.option("--port", "-p", default=9090, type=int, help="Port for metrics server")
@click.option("--interval", "-i", default=300, type=int, help="Check interval in seconds")
@click.option("--config", "-f", help="Path to configuration file")
def serve_exporter(
    baseline: str,
    data_source: str,
    port: int,
    interval: int,
    config: Optional[str],
):
    """Run Prometheus exporter with periodic drift checks."""
    # Load configuration
    cfg = Config(config) if config else None
    
    click.echo(f"Loading baseline from {baseline}...")
    try:
        store = BaselineStore(path=baseline, storage_type="auto")
        baseline_obj = store.load()
    except Exception as e:
        click.echo(f"✗ Error loading baseline: {e}", err=True)
        sys.exit(1)
    
    # Initialize detector
    if cfg:
        detection_methods = cfg.detection_methods
        exclude_features = cfg.exclude_features
        threshold = cfg.psi_threshold
    else:
        detection_methods = ["psi", "ks_test"]
        exclude_features = []
        threshold = 0.2
    
    detector = DriftDetector(
        baseline=baseline_obj,
        psi_threshold=threshold,
        methods=detection_methods,
    )
    
    # Initialize exporter
    exporter = PrometheusExporter(port=port)
    
    # Initialize alert manager
    alert_manager = AlertManager(config_path=config)
    
    click.echo(f"Starting drift checks every {interval} seconds...")
    click.echo(f"Prometheus metrics on port {port}")
    
    def run_check():
        """Run a single drift check."""
        try:
            # Load current data
            if data_source.endswith(".csv"):
                current_df = pd.read_csv(data_source)
            else:
                # For now, only CSV is supported
                click.echo(f"✗ Unsupported data source: {data_source}", err=True)
                return
            
            result = detector.check(current_df, exclude_features=exclude_features)
            
            # Update metrics
            exporter.update_metrics(result)
            
            # Send alerts if drift detected
            if result.overall_drift:
                click.echo(f"⚠ Drift detected (score: {result.overall_score:.3f})")
                alert_manager.send_alert(result)
            else:
                click.echo(f"✓ No drift (score: {result.overall_score:.3f})")
        
        except Exception as e:
            click.echo(f"✗ Error during check: {e}", err=True)
    
    # Run initial check
    run_check()
    
    # Start metrics server in background
    import threading
    
    server_thread = threading.Thread(target=exporter.serve_forever, daemon=True)
    server_thread.start()
    
    # Periodic checks
    try:
        while True:
            time.sleep(interval)
            run_check()
    except KeyboardInterrupt:
        click.echo("\nShutting down...")


if __name__ == "__main__":
    main()
