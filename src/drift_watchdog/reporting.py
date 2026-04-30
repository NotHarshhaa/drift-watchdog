"""HTML report generation for drift detection results."""

from typing import Dict, Any
from datetime import datetime
import json

from drift_watchdog.models import DriftResult, FeatureReport
from drift_watchdog.concept_drift import ConceptDriftResult, ConceptDriftReport


class HTMLReportGenerator:
    """Generate HTML reports for drift detection results."""
    
    @staticmethod
    def generate_drift_report(result: DriftResult, threshold: float) -> str:
        """
        Generate HTML report for data drift detection.
        
        Args:
            result: Drift detection result
            threshold: Drift threshold used
            
        Returns:
            HTML string
        """
        overall_status = "DRIFT DETECTED" if result.overall_drift else "NO DRIFT"
        status_color = "#dc3545" if result.overall_drift else "#28a745"
        status_bg = "#f8d7da" if result.overall_drift else "#d4edda"
        
        # Build feature rows
        feature_rows = ""
        for feature_name, report in result.features.items():
            row_style = "background-color: #fff3cd;" if report.is_drift else ""
            status_badge = '<span style="background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">DRIFT</span>' if report.is_drift else '<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">OK</span>'
            
            feature_rows += f"""
            <tr style="{row_style}">
                <td style="padding: 12px; border: 1px solid #dee2e6;">{status_badge}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>{feature_name}</strong></td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{report.psi:.4f}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{report.ks_statistic:.4f}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{report.jensen_shannon:.4f}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{report.wasserstein:.4f}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6;">{report.drift_severity.upper()}</td>
            </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drift Detection Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #dee2e6;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
            font-size: 28px;
        }}
        .header .subtitle {{
            color: #6c757d;
            margin-top: 10px;
        }}
        .status-banner {{
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 30px;
            background-color: {status_bg};
            color: {status_color};
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        .summary-card .label {{
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            color: #333;
            font-size: 20px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        th:nth-child(n+3) {{
            text-align: right;
        }}
        td {{
            padding: 12px;
            border: 1px solid #dee2e6;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .section-title {{
            color: #333;
            font-size: 20px;
            font-weight: bold;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐕 Drift Detection Report</h1>
            <div class="subtitle">Generated by drift-watchdog</div>
        </div>
        
        <div class="status-banner">
            {overall_status}
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="label">Overall Score</div>
                <div class="value">{result.overall_score:.4f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Threshold</div>
                <div class="value">{threshold}</div>
            </div>
            <div class="summary-card">
                <div class="label">Features Checked</div>
                <div class="value">{len(result.features)}</div>
            </div>
            <div class="summary-card">
                <div class="label">Features with Drift</div>
                <div class="value">{len([f for f in result.features.values() if f.is_drift])}</div>
            </div>
            <div class="summary-card">
                <div class="label">Timestamp</div>
                <div class="value" style="font-size: 14px;">{result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="summary-card">
                <div class="label">Baseline Version</div>
                <div class="value">{result.baseline_version or 'N/A'}</div>
            </div>
        </div>
        
        <div class="section-title">Feature Analysis</div>
        <table>
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Feature</th>
                    <th>PSI</th>
                    <th>KS Statistic</th>
                    <th>Jensen-Shannon</th>
                    <th>Wasserstein</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                {feature_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by drift-watchdog on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """
        return html
    
    @staticmethod
    def generate_concept_drift_report(result: ConceptDriftResult, threshold: float) -> str:
        """
        Generate HTML report for concept drift detection.
        
        Args:
            result: Concept drift detection result
            threshold: Drift threshold used
            
        Returns:
            HTML string
        """
        overall_status = "DRIFT DETECTED" if result.overall_drift else "NO DRIFT"
        status_color = "#dc3545" if result.overall_drift else "#28a745"
        status_bg = "#f8d7da" if result.overall_drift else "#d4edda"
        
        # Build metric rows
        metric_rows = ""
        for metric_name, report in result.metrics.items():
            row_style = "background-color: #fff3cd;" if report.is_drift else ""
            status_badge = '<span style="background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">DRIFT</span>' if report.is_drift else '<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">OK</span>'
            
            acc_change_str = f"{report.accuracy_change:+.4f}" if metric_name == "accuracy" else "N/A"
            
            metric_rows += f"""
            <tr style="{row_style}">
                <td style="padding: 12px; border: 1px solid #dee2e6;">{status_badge}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>{metric_name}</strong></td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{report.psi:.4f}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{report.ks_statistic:.4f}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{report.jensen_shannon:.4f}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">{acc_change_str}</td>
                <td style="padding: 12px; border: 1px solid #dee2e6;">{report.drift_severity.upper()}</td>
            </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Concept Drift Detection Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #dee2e6;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
            font-size: 28px;
        }}
        .header .subtitle {{
            color: #6c757d;
            margin-top: 10px;
        }}
        .status-banner {{
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 30px;
            background-color: {status_bg};
            color: {status_color};
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        .summary-card .label {{
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            color: #333;
            font-size: 20px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        th:nth-child(n+3) {{
            text-align: right;
        }}
        td {{
            padding: 12px;
            border: 1px solid #dee2e6;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .section-title {{
            color: #333;
            font-size: 20px;
            font-weight: bold;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐕 Concept Drift Detection Report</h1>
            <div class="subtitle">Generated by drift-watchdog</div>
        </div>
        
        <div class="status-banner">
            {overall_status}
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="label">Overall Score</div>
                <div class="value">{result.overall_score:.4f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Threshold</div>
                <div class="value">{threshold}</div>
            </div>
            <div class="summary-card">
                <div class="label">Metrics Checked</div>
                <div class="value">{len(result.metrics)}</div>
            </div>
            <div class="summary-card">
                <div class="label">Metrics with Drift</div>
                <div class="value">{len([m for m in result.metrics.values() if m.is_drift])}</div>
            </div>
            <div class="summary-card">
                <div class="label">Timestamp</div>
                <div class="value" style="font-size: 14px;">{result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        </div>
        
        <div class="section-title">Metric Analysis</div>
        <table>
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Metric</th>
                    <th>PSI</th>
                    <th>KS Statistic</th>
                    <th>Jensen-Shannon</th>
                    <th>Accuracy Change</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                {metric_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by drift-watchdog on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """
        return html
    
    @staticmethod
    def save_report(html: str, output_path: str) -> None:
        """
        Save HTML report to file.
        
        Args:
            html: HTML string
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
