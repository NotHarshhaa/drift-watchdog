"""Alert integrations for drift detection."""

import os
import requests
from typing import Optional, Dict, Any
import json

from drift_watchdog.models import DriftResult


class SlackAlerter:
    """Send alerts to Slack."""
    
    def __init__(self, webhook_url: str, channel: str = "#ml-alerts"):
        """
        Initialize Slack alerter.
        
        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel to send alerts to
        """
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send(self, result: DriftResult) -> bool:
        """
        Send drift alert to Slack.
        
        Args:
            result: Drift detection result
            
        Returns:
            True if alert sent successfully
        """
        # Determine color based on severity
        if result.overall_score >= 0.25:
            color = "#ff0000"  # Red for severe
            severity = "SEVERE"
        elif result.overall_score >= 0.2:
            color = "#ff9900"  # Orange for moderate
            severity = "MODERATE"
        else:
            color = "#ffff00"  # Yellow for slight
            severity = "SLIGHT"
        
        # Build feature list
        drifting_features = [
            f"• {name}: PSI={report.psi:.3f} ({report.drift_severity})"
            for name, report in result.features.items()
            if report.is_drift
        ]
        
        attachment = {
            "color": color,
            "title": f"🚨 Model Drift Detected - {severity}",
            "text": f"Overall drift score: {result.overall_score:.3f}",
            "fields": [
                {
                    "title": "Drifting Features",
                    "value": "\n".join(drifting_features) if drifting_features else "None",
                    "short": False,
                },
                {
                    "title": "Baseline Version",
                    "value": result.baseline_version or "unknown",
                    "short": True,
                },
                {
                    "title": "Timestamp",
                    "value": result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "short": True,
                },
            ],
        }
        
        payload = {
            "channel": self.channel,
            "attachments": [attachment],
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
            return False


class PagerDutyAlerter:
    """Send alerts to PagerDuty."""
    
    def __init__(self, routing_key: str, severity: str = "warning"):
        """
        Initialize PagerDuty alerter.
        
        Args:
            routing_key: PagerDuty integration routing key
            severity: Alert severity (critical, error, warning, info)
        """
        self.routing_key = routing_key
        self.severity = severity
    
    def send(self, result: DriftResult) -> bool:
        """
        Send drift alert to PagerDuty.
        
        Args:
            result: Drift detection result
            
        Returns:
            True if alert sent successfully
        """
        url = "https://events.pagerduty.com/v2/enqueue"
        
        # Build summary
        drifting_count = sum(1 for r in result.features.values() if r.is_drift)
        summary = (
            f"Model drift detected: {result.overall_score:.3f} overall score, "
            f"{drifting_count} features drifting"
        )
        
        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "severity": self.severity,
                "source": "drift-watchdog",
                "timestamp": result.timestamp.isoformat(),
                "custom_details": {
                    "overall_score": result.overall_score,
                    "drifting_features": [
                        {
                            "name": name,
                            "psi": report.psi,
                            "severity": report.drift_severity,
                        }
                        for name, report in result.features.items()
                        if report.is_drift
                    ],
                    "baseline_version": result.baseline_version,
                },
            },
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send PagerDuty alert: {e}")
            return False


class WebhookAlerter:
    """Send alerts to generic webhook."""
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize webhook alerter.
        
        Args:
            url: Webhook URL
            headers: Optional HTTP headers
        """
        self.url = url
        self.headers = headers or {}
    
    def send(self, result: DriftResult) -> bool:
        """
        Send drift alert to webhook.
        
        Args:
            result: Drift detection result
            
        Returns:
            True if alert sent successfully
        """
        payload = result.to_dict()
        
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send webhook alert: {e}")
            return False


class AlertManager:
    """Manage multiple alert channels."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize alert manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.alerters = []
        self.config = {}
        
        if config_path:
            self._load_config(config_path)
        else:
            self._load_from_env()
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        import yaml
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.config = config.get("alerts", {})
        self._setup_alerters()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Slack
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            self.alerters.append(
                SlackAlerter(
                    webhook_url=slack_webhook,
                    channel=os.getenv("SLACK_CHANNEL", "#ml-alerts"),
                )
            )
        
        # PagerDuty
        pd_routing_key = os.getenv("PD_ROUTING_KEY")
        if pd_routing_key:
            self.alerters.append(
                PagerDutyAlerter(
                    routing_key=pd_routing_key,
                    severity=os.getenv("PD_SEVERITY", "warning"),
                )
            )
        
        # Webhook
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url:
            self.alerters.append(WebhookAlerter(url=webhook_url))
    
    def _setup_alerters(self) -> None:
        """Setup alerters from configuration."""
        # Slack
        slack_config = self.config.get("slack", {})
        if slack_config.get("webhook_url"):
            self.alerters.append(
                SlackAlerter(
                    webhook_url=slack_config["webhook_url"],
                    channel=slack_config.get("channel", "#ml-alerts"),
                )
            )
        
        # PagerDuty
        pd_config = self.config.get("pagerduty", {})
        if pd_config.get("routing_key"):
            self.alerters.append(
                PagerDutyAlerter(
                    routing_key=pd_config["routing_key"],
                    severity=pd_config.get("severity", "warning"),
                )
            )
        
        # Webhook
        webhook_config = self.config.get("webhook", {})
        if webhook_config.get("url"):
            self.alerters.append(
                WebhookAlerter(
                    url=webhook_config["url"],
                    headers=webhook_config.get("headers"),
                )
            )
    
    def send_alert(self, result: DriftResult) -> bool:
        """
        Send alert to all configured channels.
        
        Args:
            result: Drift detection result
            
        Returns:
            True if at least one alert was sent successfully
        """
        if not result.overall_drift:
            return True
        
        success = False
        for alerter in self.alerters:
            if alerter.send(result):
                success = True
        
        return success
