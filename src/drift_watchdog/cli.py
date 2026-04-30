"""Command-line interface for drift-watchdog with Rich UI."""

import sys
import time
import pandas as pd
import click
from pathlib import Path
from typing import Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.status import Status
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.syntax import Syntax

from drift_watchdog.baseline import BaselineStore
from drift_watchdog.detector import DriftDetector
from drift_watchdog.exporter import PrometheusExporter
from drift_watchdog.config import Config
from drift_watchdog.alerts import AlertManager
from drift_watchdog.models import DriftResult
from drift_watchdog.concept_drift import ConceptDriftDetector
from drift_watchdog.reporting import HTMLReportGenerator
from drift_watchdog.data_quality import DataQualityChecker
from drift_watchdog.trend_analysis import DriftTrendAnalyzer
from drift_watchdog.correlation_analysis import CorrelationAnalyzer
from drift_watchdog.schema_validator import SchemaValidator
from drift_watchdog.drift_explainer import DriftExplainer

# Create console instance
console = Console()


def get_severity_style(severity: str) -> str:
    """Get Rich style for severity level."""
    styles = {
        "severe": "bold red",
        "moderate": "bold yellow",
        "slight": "yellow",
        "none": "green",
    }
    return styles.get(severity, "white")


def get_severity_emoji(severity: str) -> str:
    """Get emoji for severity level."""
    emojis = {
        "severe": "🔴",
        "moderate": "🟡",
        "slight": "🟠",
        "none": "🟢",
    }
    return emojis.get(severity, "⚪")


def print_banner():
    """Print the enhanced drift-watchdog banner."""
    banner = """
╭─────────────────────────────────────────────────────────────╮
│                                                             │
│   🐕  [bold cyan]drift-watchdog[/bold cyan]  - ML Model Drift Detection         │
│                                                             │
│   Monitor your ML models for data drift and performance     │
│   degradation in production environments.                   │
│                                                             │
│   [dim]✨ Rich UI • 📊 Prometheus Metrics • 🚨 Real-time Alerts[/dim]│
│                                                             │
╰─────────────────────────────────────────────────────────────╯
"""
    console.print(banner)
    
    # Add a quick status indicator
    status_table = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    status_table.add_column("Status", style="green")
    status_table.add_row("✅ Ready to monitor your ML models")
    console.print(status_table)
    console.print()


def display_drift_result(result: DriftResult, threshold: float):
    """Display drift detection result in a beautiful table format."""
    # Overall status panel with enhanced visual
    status_emoji = "🔴" if result.overall_drift else "🟢"
    status_text = "DRIFT DETECTED" if result.overall_drift else "NO DRIFT"
    status_style = "bold red" if result.overall_drift else "bold green"
    
    # Create a visual progress bar for drift score
    score_percentage = min(result.overall_score / threshold, 1.0) * 100
    bar_color = "red" if result.overall_drift else "green"
    progress_bar = "█" * int(score_percentage / 5) + "░" * (20 - int(score_percentage / 5))
    
    summary_content = f"""
{status_emoji} Status: [{status_style}]{status_text}[/{status_style}]
📊 Overall Score: [{bar_color}]{result.overall_score:.4f}[/{bar_color}] [{bar_color}][{progress_bar}][/{bar_color}]
⚙️  Threshold: {threshold}
🕐 Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
📋 Baseline Version: {result.baseline_version or 'N/A'}
🔍 Features Checked: {len(result.features)}
⚠️  Drifted Features: {len([f for f in result.features.values() if f.is_drift])}
"""
    
    console.print(Panel(
        summary_content.strip(),
        title="[bold]🐕 Drift Detection Summary[/bold]",
        border_style="red" if result.overall_drift else "green",
        box=box.ROUNDED,
    ))
    console.print()
    
    # Feature details table with enhanced visuals
    table = Table(
        title="[bold]📊 Feature Analysis[/bold]",
        box=box.ROUNDED,
        header_style="bold magenta",
        border_style="blue",
        show_lines=True,
    )
    
    table.add_column("Status", justify="center", width=8)
    table.add_column("Feature", style="cyan", width=20)
    table.add_column("PSI", justify="right", width=10)
    table.add_column("PSI Bar", justify="left", width=15)
    table.add_column("KS Stat", justify="right", width=10)
    table.add_column("JS Div", justify="right", width=10)
    table.add_column("Severity", justify="center", width=12)
    
    for feature_name, report in result.features.items():
        emoji = get_severity_emoji(report.drift_severity)
        severity_style = get_severity_style(report.drift_severity)
        
        # Create mini progress bar for PSI
        psi_percentage = min(report.psi / threshold, 1.0) * 100
        psi_bar_color = "red" if report.is_drift else "green"
        psi_bar = "█" * int(psi_percentage / 10) + "░" * (10 - int(psi_percentage / 10))
        
        severity_display = f"[{severity_style}]{report.drift_severity.upper()}[/{severity_style}]" if report.is_drift else "[green]OK[/green]"
        
        table.add_row(
            emoji,
            feature_name,
            f"[{psi_bar_color}]{report.psi:.3f}[/{psi_bar_color}]",
            f"[{psi_bar_color}][{psi_bar}][/{psi_bar_color}]",
            f"{report.ks_statistic:.3f}",
            f"{report.jensen_shannon:.3f}",
            severity_display,
        )
    
    console.print(table)
    console.print()
    
    # Enhanced drifted features panel (if any)
    drifted = [(name, r) for name, r in result.features.items() if r.is_drift]
    if drifted:
        # Sort by PSI score (highest first)
        drifted_sorted = sorted(drifted, key=lambda x: x[1].psi, reverse=True)
        
        drifted_text = "\n".join([
            f"  {get_severity_emoji(r.drift_severity)} [bold]{name}[/bold] "
            f"- PSI: [{get_severity_style(r.drift_severity)}]{r.psi:.4f}[/{get_severity_style(r.drift_severity)}] "
            f"(KS: {r.ks_statistic:.3f})"
            for name, r in drifted_sorted
        ])
        
        # Add summary statistics
        total_drift_score = sum(r.psi for _, r in drifted)
        avg_drift_score = total_drift_score / len(drifted) if drifted else 0
        
        drifted_text += f"\n\n[dim]📈 Total drift score: {total_drift_score:.4f}[/dim]"
        drifted_text += f"\n[dim]📊 Average drift score: {avg_drift_score:.4f}[/dim]"
        
        console.print(Panel(
            drifted_text,
            title=f"[bold red]⚠️  {len(drifted)} Features with Drift Detected[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))
    else:
        # Success panel when no drift detected
        console.print(Panel(
            "[green]✅ All features are within acceptable thresholds[/green]\n[dim]🎉 Your model data is stable![/dim]",
            title="[bold green]✨ No Drift Detected[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="1.3.0", prog_name="drift-watchdog")
def main(ctx):
    """drift-watchdog: Lightweight ML model drift detection."""
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("[dim]Run with --help for available commands.[/dim]")
        console.print()
        console.print("[bold]Quick Start:[/bold]")
        console.print("  1. Create a baseline: [cyan]drift-watchdog baseline create --data ref.csv --output baseline.json --name my-model[/cyan]")
        console.print("  2. Check for drift:   [cyan]drift-watchdog check --baseline baseline.json --current current.csv[/cyan]")
        console.print("  3. Run exporter:      [cyan]drift-watchdog serve --baseline baseline.json --data-source current.csv[/cyan]")


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
    console.print()
    console.print(Rule(title="[bold cyan]Creating Baseline[/bold cyan]", style="cyan"))
    console.print()
    
    try:
        with Status(
            f"[bold yellow]Loading data from {data}...",
            console=console,
            spinner="dots",
        ):
            baseline_obj = BaselineStore.create_from_csv(
                csv_path=data,
                name=name,
                exclude_features=list(exclude) if exclude else None,
            )
        
        console.print(f"[green]✓[/green] Loaded [bold]{len(baseline_obj.feature_names)}[/bold] features from dataset")
        
        with Status(
            f"[bold yellow]Saving baseline to {output}...",
            console=console,
            spinner="dots",
        ):
            store = BaselineStore(path=output, storage_type=storage)
            store.save(baseline_obj)
        
        console.print(f"[green]✓[/green] Baseline saved successfully")
        console.print()
        
        # Display baseline info
        info_table = Table(box=box.ROUNDED, border_style="green")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Name", baseline_obj.name)
        info_table.add_row("Version", baseline_obj.version)
        info_table.add_row("Features", str(len(baseline_obj.feature_names)))
        info_table.add_row("Created", baseline_obj.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        info_table.add_row("Storage", storage)
        info_table.add_row("Path", output)
        
        console.print(Panel(
            info_table,
            title="[bold green]Baseline Created Successfully[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        
        # Show feature list
        if baseline_obj.feature_names:
            console.print()
            console.print("[bold]Features:[/bold]")
            feature_cols = Columns([
                f"  [cyan]•[/cyan] {feat}" for feat in baseline_obj.feature_names
            ], equal=True, expand=True)
            console.print(feature_cols)
        
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}",
            title="[bold red]Failed to Create Baseline[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    console.print()


@main.command("check")
@click.option("--baseline", "-b", required=True, help="Path to baseline JSON")
@click.option("--current", "-c", required=True, help="Path to current data CSV")
@click.option("--threshold", "-t", default=0.2, type=float, help="Drift threshold")
@click.option("--config", "-f", help="Path to configuration file")
@click.option("--methods", "-m", multiple=True, help="Detection methods to use")
@click.option("--alert", is_flag=True, help="Send alerts if drift detected")
@click.option("--report", "-r", help="Path to save HTML report")
@click.option("--feature-importance", "-fi", help="JSON file with feature importance weights")
@click.option("--custom-thresholds", "-ct", help="JSON file with custom thresholds per feature")
@click.option("--explain", "-e", is_flag=True, help="Generate drift explanation")
def check_drift(
    baseline: str,
    current: str,
    threshold: float,
    config: Optional[str],
    methods: tuple,
    alert: bool,
    report: Optional[str],
    feature_importance: Optional[str],
    custom_thresholds: Optional[str],
    explain: bool,
):
    """Run drift detection check."""
    console.print()
    console.print(Rule(title="[bold cyan]Drift Detection[/bold cyan]", style="cyan"))
    console.print()
    
    # Load configuration
    cfg = Config(config) if config else None
    
    # Load baseline
    with Status(
        f"[bold yellow]Loading baseline from {baseline}...",
        console=console,
        spinner="dots",
    ):
        try:
            store = BaselineStore(path=baseline, storage_type="auto")
            baseline_obj = store.load()
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to load baseline: {e}")
            sys.exit(1)
    console.print(f"[green]✓[/green] Baseline loaded: [bold]{baseline_obj.name}[/bold] (v{baseline_obj.version})")
    
    # Load current data
    with Status(
        f"[bold yellow]Loading current data from {current}...",
        console=console,
        spinner="dots",
    ):
        try:
            current_df = pd.read_csv(current)
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to load current data: {e}")
            sys.exit(1)
    console.print(f"[green]✓[/green] Loaded [bold]{len(current_df)}[/bold] rows with [bold]{len(current_df.columns)}[/bold] columns")
    
    # Get configuration values
    if cfg:
        detection_methods = list(methods) if methods else cfg.detection_methods
        exclude_features = cfg.exclude_features
    else:
        detection_methods = list(methods) if methods else ["psi", "ks_test"]
        exclude_features = []
    
    console.print(f"[green]✓[/green] Detection methods: [cyan]{', '.join(detection_methods)}[/cyan]")
    console.print()
    
    # Load feature importance if provided
    feature_importance_dict = {}
    if feature_importance:
        import json
        with open(feature_importance, 'r') as f:
            feature_importance_dict = json.load(f)
        console.print(f"[green]✓[/green] Loaded feature importance from [cyan]{feature_importance}[/cyan]")
    
    # Load custom thresholds if provided
    custom_thresholds_dict = {}
    if custom_thresholds:
        import json
        with open(custom_thresholds, 'r') as f:
            custom_thresholds_dict = json.load(f)
        console.print(f"[green]✓[/green] Loaded custom thresholds from [cyan]{custom_thresholds}[/cyan]")
    
    console.print()
    
    # Run detection
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running drift detection...", total=None)
        
        detector = DriftDetector(
            baseline=baseline_obj,
            psi_threshold=threshold,
            methods=detection_methods,
            feature_importance=feature_importance_dict,
            custom_thresholds=custom_thresholds_dict,
        )
        
        result = detector.check(current_df, exclude_features=exclude_features)
        progress.update(task, completed=True)
    
    # Display results
    display_drift_result(result, threshold)
    
    # Send alerts if configured
    if alert and result.overall_drift:
        console.print()
        with Status("[bold yellow]Sending alerts...", console=console, spinner="dots"):
            alert_manager = AlertManager(config_path=config)
            if alert_manager.send_alert(result):
                console.print("[green]✓[/green] Alerts sent successfully")
            else:
                console.print("[yellow]⚠[/yellow] Failed to send some alerts")
    
    # Generate HTML report if requested
    if report:
        console.print()
        with Status(f"[bold yellow]Generating HTML report at {report}...", console=console, spinner="dots"):
            html_report = HTMLReportGenerator.generate_drift_report(result, threshold)
            HTMLReportGenerator.save_report(html_report, report)
        console.print(f"[green]✓[/green] HTML report saved successfully")
    
    # Generate drift explanation if requested
    if explain:
        console.print()
        with Status("[bold yellow]Generating drift explanation...", console=console, spinner="dots"):
            explainer = DriftExplainer()
            explanation = explainer.explain(result)
        display_drift_explanation(explanation)
    
    console.print()
    console.print(Rule(style="cyan"))
    console.print()
    
    # Exit with error code if drift detected
    sys.exit(1 if result.overall_drift else 0)


@main.command("concept-check")
@click.option("--baseline-predictions", "-bp", required=True, help="Path to baseline predictions CSV")
@click.option("--baseline-labels", "-bl", required=True, help="Path to baseline labels CSV")
@click.option("--current-predictions", "-cp", required=True, help="Path to current predictions CSV")
@click.option("--current-labels", "-cl", required=True, help="Path to current labels CSV")
@click.option("--threshold", "-t", default=0.2, type=float, help="Drift threshold")
@click.option("--report", "-r", help="Path to save HTML report")
def check_concept_drift(
    baseline_predictions: str,
    baseline_labels: str,
    current_predictions: str,
    current_labels: str,
    threshold: float,
    report: Optional[str],
):
    """Run concept drift detection check on model outputs/labels."""
    console.print()
    console.print(Rule(title="[bold cyan]Concept Drift Detection[/bold cyan]", style="cyan"))
    console.print()
    
    try:
        # Load data
        with Status("[bold yellow]Loading baseline data...", console=console, spinner="dots"):
            baseline_pred_df = pd.read_csv(baseline_predictions)
            baseline_label_df = pd.read_csv(baseline_labels)
        console.print(f"[green]✓[/green] Loaded baseline data")
        
        with Status("[bold yellow]Loading current data...", console=console, spinner="dots"):
            current_pred_df = pd.read_csv(current_predictions)
            current_label_df = pd.read_csv(current_labels)
        console.print(f"[green]✓[/green] Loaded current data")
        
        # Extract arrays (assuming single column)
        baseline_preds = baseline_pred_df.iloc[:, 0].values
        baseline_labels_arr = baseline_label_df.iloc[:, 0].values
        current_preds = current_pred_df.iloc[:, 0].values
        current_labels_arr = current_label_df.iloc[:, 0].values
        
        console.print(f"[green]✓[/green] Baseline samples: [bold]{len(baseline_preds)}[/bold]")
        console.print(f"[green]✓[/green] Current samples: [bold]{len(current_preds)}[/bold]")
        console.print()
        
        # Run detection
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running concept drift detection...", total=None)
            
            detector = ConceptDriftDetector(psi_threshold=threshold)
            result = detector.check(
                baseline_predictions=baseline_preds,
                baseline_labels=baseline_labels_arr,
                current_predictions=current_preds,
                current_labels=current_labels_arr,
            )
            progress.update(task, completed=True)
        
        # Display results
        display_concept_drift_result(result, threshold)
        
        # Generate HTML report if requested
        if report:
            console.print()
            with Status(f"[bold yellow]Generating HTML report at {report}...", console=console, spinner="dots"):
                html_report = HTMLReportGenerator.generate_concept_drift_report(result, threshold)
                HTMLReportGenerator.save_report(html_report, report)
            console.print(f"[green]✓[/green] HTML report saved successfully")
        
        console.print()
        console.print(Rule(style="cyan"))
        console.print()
        
        # Exit with error code if drift detected
        sys.exit(1 if result.overall_drift else 0)
        
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}",
            title="[bold red]Concept Drift Check Failed[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))
        sys.exit(1)


def display_concept_drift_result(result, threshold: float):
    """Display concept drift detection result."""
    # Overall status panel
    status_emoji = "🔴" if result.overall_drift else "🟢"
    status_text = "DRIFT DETECTED" if result.overall_drift else "NO DRIFT"
    status_style = "bold red" if result.overall_drift else "bold green"
    
    summary_content = f"""
{status_emoji} Status: [{status_style}]{status_text}[/{status_style}]
📊 Overall Score: {result.overall_score:.4f}
⚙️  Threshold: {threshold}
🕐 Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
📋 Metrics Checked: {len(result.metrics)}
⚠️  Drifted Metrics: {len([m for m in result.metrics.values() if m.is_drift])}
"""
    
    console.print(Panel(
        summary_content.strip(),
        title="[bold]🐕 Concept Drift Detection Summary[/bold]",
        border_style="red" if result.overall_drift else "green",
        box=box.ROUNDED,
    ))
    console.print()
    
    # Metrics details table
    table = Table(
        title="[bold]📊 Metric Analysis[/bold]",
        box=box.ROUNDED,
        header_style="bold magenta",
        border_style="blue",
        show_lines=True,
    )
    
    table.add_column("Status", justify="center", width=8)
    table.add_column("Metric", style="cyan", width=15)
    table.add_column("PSI", justify="right", width=10)
    table.add_column("KS Stat", justify="right", width=10)
    table.add_column("JS Div", justify="right", width=10)
    table.add_column("Accuracy Change", justify="right", width=15)
    table.add_column("Severity", justify="center", width=12)
    
    for metric_name, report in result.metrics.items():
        emoji = get_severity_emoji(report.drift_severity)
        severity_style = get_severity_style(report.drift_severity)
        
        severity_display = f"[{severity_style}]{report.drift_severity.upper()}[/{severity_style}]" if report.is_drift else "[green]OK[/green]"
        
        acc_change_str = f"{report.accuracy_change:+.4f}" if report.metric_name == "accuracy" else "N/A"
        
        table.add_row(
            emoji,
            metric_name,
            f"{report.psi:.3f}",
            f"{report.ks_statistic:.3f}",
            f"{report.jensen_shannon:.3f}",
            acc_change_str,
            severity_display,
        )
    
    console.print(table)
    console.print()


@main.command("quality-check")
@click.option("--data", "-d", required=True, help="Path to data CSV file")
@click.option("--missing-threshold", "-m", default=0.1, type=float, help="Missing value threshold (0-1)")
@click.option("--outlier-threshold", "-o", default=0.05, type=float, help="Outlier threshold (0-1)")
@click.option("--outlier-method", default="iqr", help="Outlier detection method (iqr, zscore)")
def check_data_quality(
    data: str,
    missing_threshold: float,
    outlier_threshold: float,
    outlier_method: str,
):
    """Run data quality check on dataset."""
    console.print()
    console.print(Rule(title="[bold cyan]Data Quality Check[/bold cyan]", style="cyan"))
    console.print()
    
    try:
        # Load data
        with Status(f"[bold yellow]Loading data from {data}...", console=console, spinner="dots"):
            df = pd.read_csv(data)
        console.print(f"[green]✓[/green] Loaded [bold]{len(df)}[/bold] rows with [bold]{len(df.columns)}[/bold] columns")
        console.print()
        
        # Run quality check
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running data quality check...", total=None)
            
            checker = DataQualityChecker(
                missing_threshold=missing_threshold,
                outlier_threshold=outlier_threshold,
                outlier_method=outlier_method,
            )
            result = checker.check(df)
            progress.update(task, completed=True)
        
        # Display results
        display_data_quality_result(result, missing_threshold, outlier_threshold)
        
        console.print()
        console.print(Rule(style="cyan"))
        console.print()
        
        # Exit with error code if issues detected
        sys.exit(1 if result.has_issues else 0)
        
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}",
            title="[bold red]Data Quality Check Failed[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))
        sys.exit(1)


def display_data_quality_result(result, missing_threshold: float, outlier_threshold: float):
    """Display data quality check result."""
    # Overall status panel
    status_emoji = "🔴" if result.has_issues else "🟢"
    status_text = "ISSUES DETECTED" if result.has_issues else "NO ISSUES"
    status_style = "bold red" if result.has_issues else "bold green"
    
    summary_content = f"""
{status_emoji} Status: [{status_style}]{status_text}[/{status_style}]
📊 Quality Score: {result.overall_quality_score:.2%}
⚙️  Missing Threshold: {missing_threshold:.1%}
⚙️  Outlier Threshold: {outlier_threshold:.1%}
🕐 Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
📋 Features Checked: {len(result.features)}
⚠️  Features with Issues: {len([f for f in result.features.values() if f.is_issue])}
"""
    
    console.print(Panel(
        summary_content.strip(),
        title="[bold]🐕 Data Quality Check Summary[/bold]",
        border_style="red" if result.has_issues else "green",
        box=box.ROUNDED,
    ))
    console.print()
    
    # Feature details table
    table = Table(
        title="[bold]📊 Feature Quality Analysis[/bold]",
        box=box.ROUNDED,
        header_style="bold magenta",
        border_style="blue",
        show_lines=True,
    )
    
    table.add_column("Status", justify="center", width=8)
    table.add_column("Feature", style="cyan", width=20)
    table.add_column("Missing %", justify="right", width=12)
    table.add_column("Outlier %", justify="right", width=12)
    table.add_column("Unique %", justify="right", width=12)
    table.add_column("Severity", justify="center", width=12)
    
    for feature_name, report in result.features.items():
        emoji = "🔴" if report.is_issue else "🟢"
        severity_style = get_severity_style(report.issue_severity) if report.is_issue else "green"
        
        severity_display = f"[{severity_style}]{report.issue_severity.upper()}[/{severity_style}]" if report.is_issue else "[green]OK[/green]"
        
        table.add_row(
            emoji,
            feature_name,
            f"{report.missing_percentage:.2%}",
            f"{report.outlier_percentage:.2%}",
            f"{report.unique_percentage:.2%}",
            severity_display,
        )
    
    console.print(table)
    console.print()


def display_drift_explanation(explanation):
    """Display drift explanation result."""
    console.print()
    console.print(Rule(title="[bold cyan]Drift Explanation[/bold cyan]", style="cyan"))
    console.print()
    
    # Overall explanation
    console.print(Panel(
        explanation.overall_explanation,
        title="[bold]📋 Overall Assessment[/bold]",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print()
    
    # Primary drivers
    if explanation.primary_drivers:
        table = Table(
            title="[bold]🎯 Primary Drift Drivers[/bold]",
            box=box.ROUNDED,
            header_style="bold magenta",
            border_style="red",
            show_lines=True,
        )
        
        table.add_column("Feature", style="cyan", width=20)
        table.add_column("Contribution", justify="right", width=12)
        table.add_column("Type", width=18)
        table.add_column("Severity", justify="center", width=10)
        
        for driver in explanation.primary_drivers:
            severity_style = get_severity_style(driver.severity)
            table.add_row(
                driver.feature_name,
                f"{driver.contribution_score:.2%}",
                driver.drift_type,
                f"[{severity_style}]{driver.severity.upper()}[/{severity_style}]",
            )
        
        console.print(table)
        console.print()
        
        # Suggested actions
        console.print(Panel(
            "\n".join([f"• {driver.suggested_action}" for driver in explanation.primary_drivers]),
            title="[bold]💡 Suggested Actions[/bold]",
            border_style="yellow",
            box=box.ROUNDED,
        ))
        console.print()
    
    # Summary statistics
    stats_table = Table(box=box.ROUNDED, border_style="blue", show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    for key, value in explanation.summary_statistics.items():
        stats_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(Panel(
        stats_table,
        title="[bold]📊 Summary Statistics[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    ))
    console.print()


@main.command("schema-validate")
@click.option("--baseline", "-b", required=True, help="Path to baseline data CSV")
@click.option("--current", "-c", required=True, help="Path to current data CSV")
@click.option("--strict", is_flag=True, help="Treat all mismatches as errors")
def validate_schema(
    baseline: str,
    current: str,
    strict: bool,
):
    """Validate data schema consistency."""
    console.print()
    console.print(Rule(title="[bold cyan]Schema Validation[/bold cyan]", style="cyan"))
    console.print()
    
    try:
        # Load data
        with Status(f"[bold yellow]Loading data...", console=console, spinner="dots"):
            baseline_df = pd.read_csv(baseline)
            current_df = pd.read_csv(current)
        console.print(f"[green]✓[/green] Loaded baseline: [bold]{len(baseline_df)}[/bold] rows, [bold]{len(baseline_df.columns)}[/bold] columns")
        console.print(f"[green]✓[/green] Loaded current: [bold]{len(current_df)}[/bold] rows, [bold]{len(current_df.columns)}[/bold] columns")
        console.print()
        
        # Run validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating schema...", total=None)
            
            validator = SchemaValidator(strict=strict)
            result = validator.validate(baseline_df, current_df)
            progress.update(task, completed=True)
        
        # Display results
        display_schema_result(result)
        
        console.print()
        console.print(Rule(style="cyan"))
        console.print()
        
        # Exit with error code if invalid
        sys.exit(0 if result.is_valid else 1)
        
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}",
            title="[bold red]Schema Validation Failed[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))
        sys.exit(1)


def display_schema_result(result):
    """Display schema validation result."""
    status_emoji = "🟢" if result.is_valid else "🔴"
    status_text = "VALID" if result.is_valid else "INVALID"
    status_style = "bold green" if result.is_valid else "bold red"
    
    summary_content = f"""
{status_emoji} Status: [{status_style}]{status_text}[/{status_style}]
📋 Baseline Features: {len(result.baseline_features)}
📋 Current Features: {len(result.current_features)}
❌ Missing Features: {len(result.missing_features)}
➕ Extra Features: {len(result.extra_features)}
⚠️  Type Mismatches: {len(result.type_mismatches)}
"""
    
    console.print(Panel(
        summary_content.strip(),
        title="[bold]🐕 Schema Validation Summary[/bold]",
        border_style="green" if result.is_valid else "red",
        box=box.ROUNDED,
    ))
    console.print()
    
    # Display issues
    if result.issues:
        table = Table(
            title="[bold]⚠️ Schema Issues[/bold]",
            box=box.ROUNDED,
            header_style="bold magenta",
            border_style="yellow",
            show_lines=True,
        )
        
        table.add_column("Type", style="cyan", width=18)
        table.add_column("Feature", style="white", width=20)
        table.add_column("Expected", style="yellow", width=25)
        table.add_column("Actual", style="red", width=25)
        table.add_column("Severity", justify="center", width=10)
        
        for issue in result.issues:
            severity_style = "bold red" if issue.severity == "error" else "yellow"
            table.add_row(
                issue.issue_type,
                issue.feature_name,
                issue.expected,
                issue.actual,
                f"[{severity_style}]{issue.severity.upper()}[/{severity_style}]",
            )
        
        console.print(table)
        console.print()


@main.command("correlation-check")
@click.option("--baseline", "-b", required=True, help="Path to baseline data CSV")
@click.option("--current", "-c", required=True, help="Path to current data CSV")
@click.option("--threshold", "-t", default=0.3, type=float, help="Significance threshold for correlation change")
@click.option("--method", default="pearson", help="Correlation method (pearson, spearman, kendall)")
def check_correlation(
    baseline: str,
    current: str,
    threshold: float,
    method: str,
):
    """Analyze feature correlation changes."""
    console.print()
    console.print(Rule(title="[bold cyan]Correlation Analysis[/bold cyan]", style="cyan"))
    console.print()
    
    try:
        # Load data
        with Status(f"[bold yellow]Loading data...", console=console, spinner="dots"):
            baseline_df = pd.read_csv(baseline)
            current_df = pd.read_csv(current)
        console.print(f"[green]✓[/green] Loaded baseline: [bold]{len(baseline_df)}[/bold] rows")
        console.print(f"[green]✓[/green] Loaded current: [bold]{len(current_df)}[/bold] rows")
        console.print()
        
        # Run analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing correlations...", total=None)
            
            analyzer = CorrelationAnalyzer(significance_threshold=threshold)
            result = analyzer.analyze(baseline_df, current_df, method=method)
            progress.update(task, completed=True)
        
        # Display results
        display_correlation_result(result, threshold)
        
        console.print()
        console.print(Rule(style="cyan"))
        console.print()
        
        # Exit with error code if drift detected
        sys.exit(1 if result.is_drift_detected else 0)
        
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}",
            title="[bold red]Correlation Analysis Failed[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))
        sys.exit(1)


def display_correlation_result(result, threshold: float):
    """Display correlation analysis result."""
    status_emoji = "🔴" if result.is_drift_detected else "🟢"
    status_text = "DRIFT DETECTED" if result.is_drift_detected else "NO DRIFT"
    status_style = "bold red" if result.is_drift_detected else "bold green"
    
    summary_content = f"""
{status_emoji} Status: [{status_style}]{status_text}[/{status_style}]
📊 Overall Correlation Drift: {result.overall_correlation_drift:.4f}
⚙️  Significance Threshold: {threshold}
📋 Feature Pairs Analyzed: {result.total_pairs}
⚠️  Significant Changes: {result.significant_changes}
"""
    
    console.print(Panel(
        summary_content.strip(),
        title="[bold]🐕 Correlation Analysis Summary[/bold]",
        border_style="red" if result.is_drift_detected else "green",
        box=box.ROUNDED,
    ))
    console.print()
    
    # Display top changes
    if result.correlation_changes:
        top_changes = sorted(
            result.correlation_changes,
            key=lambda x: x.correlation_change,
            reverse=True,
        )[:10]
        
        table = Table(
            title="[bold]📊 Top Correlation Changes[/bold]",
            box=box.ROUNDED,
            header_style="bold magenta",
            border_style="blue",
            show_lines=True,
        )
        
        table.add_column("Feature 1", style="cyan", width=15)
        table.add_column("Feature 2", style="cyan", width=15)
        table.add_column("Baseline", justify="right", width=12)
        table.add_column("Current", justify="right", width=12)
        table.add_column("Change", justify="right", width=12)
        table.add_column("Significant", justify="center", width=12)
        
        for change in top_changes:
            significant = "✓" if change.is_significant else "✗"
            significant_style = "bold red" if change.is_significant else "green"
            table.add_row(
                change.feature1,
                change.feature2,
                f"{change.baseline_correlation:.3f}",
                f"{change.current_correlation:.3f}",
                f"{change.correlation_change:+.3f}",
                f"[{significant_style}]{significant}[/{significant_style}]",
            )
        
        console.print(table)
        console.print()


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
    console.print()
    console.print(Rule(title="[bold cyan]Drift Watchdog Exporter[/bold cyan]", style="cyan"))
    console.print()
    
    # Load configuration
    cfg = Config(config) if config else None
    
    # Load baseline
    with Status(
        f"[bold yellow]Loading baseline from {baseline}...",
        console=console,
        spinner="dots",
    ):
        try:
            store = BaselineStore(path=baseline, storage_type="auto")
            baseline_obj = store.load()
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Failed to load baseline: {e}")
            sys.exit(1)
    console.print(f"[green]✓[/green] Baseline loaded: [bold]{baseline_obj.name}[/bold]")
    
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
    
    # Initialize exporter and alert manager
    exporter = PrometheusExporter(port=port)
    alert_manager = AlertManager(config_path=config)
    
    # Display config panel
    config_table = Table(box=box.ROUNDED, border_style="blue")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Port", str(port))
    config_table.add_row("Check Interval", f"{interval}s")
    config_table.add_row("Data Source", data_source)
    config_table.add_row("Methods", ", ".join(detection_methods))
    config_table.add_row("Threshold", str(threshold))
    
    console.print(Panel(
        config_table,
        title="[bold]Exporter Configuration[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    ))
    console.print()
    
    # Run initial check
    def run_check():
        """Run a single drift check."""
        try:
            # Load current data
            if data_source.endswith(".csv"):
                current_df = pd.read_csv(data_source)
            else:
                console.print(f"[bold red]✗[/bold red] Unsupported data source: {data_source}")
                return
            
            result = detector.check(current_df, exclude_features=exclude_features)
            
            # Update metrics
            exporter.update_metrics(result)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Enhanced status display with visual indicators
            status_emoji = "🔴" if result.overall_drift else "🟢"
            status_text = "DRIFT DETECTED" if result.overall_drift else "STABLE"
            status_style = "bold red" if result.overall_drift else "bold green"
            
            # Create compact status line
            console.print(
                f"{status_emoji} [{timestamp}] [{status_style}]{status_text}[/{status_style}] - "
                f"Score: [bold]{result.overall_score:.4f}[/bold] - "
                f"Features: {len([f for f in result.features.values() if f.is_drift])}/{len(result.features)} drifted"
            )
            
            if result.overall_drift:
                alert_manager.send_alert(result)
        
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Error during check: {e}")
    
    console.print("[bold]Starting initial check...[/bold]")
    run_check()
    
    # Start metrics server in background
    import threading
    server_thread = threading.Thread(target=exporter.serve_forever, daemon=True)
    server_thread.start()
    
    console.print(f"[green]✓[/green] Prometheus metrics server started on [bold]port {port}[/bold]")
    console.print(f"[dim]Metrics available at: http://localhost:{port}/metrics[/dim]")
    console.print()
    console.print(f"[bold cyan]Running periodic checks every {interval} seconds[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    console.print()
    
    # Periodic checks
    try:
        while True:
            time.sleep(interval)
            run_check()
    except KeyboardInterrupt:
        console.print()
        console.print(Rule(title="[bold yellow]Shutting Down[/bold yellow]", style="yellow"))
        console.print("[dim]Goodbye! 👋[/dim]")
        console.print()


if __name__ == "__main__":
    main()
