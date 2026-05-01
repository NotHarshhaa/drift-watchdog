"""Alert integrations for drift detection."""

import os
import requests
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

from drift_watchdog.models import DriftResult


class RateLimiter:
    """Simple rate limiter for alert sending."""
    
    def __init__(self, max_alerts: int = 5, time_window_seconds: int = 300):
        """
        Initialize rate limiter.
        
        Args:
            max_alerts: Maximum number of alerts allowed in time window
            time_window_seconds: Time window in seconds
        """
        self.max_alerts = max_alerts
        self.time_window = timedelta(seconds=time_window_seconds)
        self.alert_history = defaultdict(list)
    
    def can_send(self, alert_type: str = "default") -> bool:
        """
        Check if alert can be sent based on rate limit.
        
        Args:
            alert_type: Type of alert for separate rate limiting
            
        Returns:
            True if alert can be sent, False otherwise
        """
        now = datetime.utcnow()
        
        # Clean old entries outside time window
        self.alert_history[alert_type] = [
            timestamp for timestamp in self.alert_history[alert_type]
            if now - timestamp < self.time_window
        ]
        
        # Check if under limit
        return len(self.alert_history[alert_type]) < self.max_alerts
    
    def record_alert(self, alert_type: str = "default") -> None:
        """
        Record that an alert was sent.
        
        Args:
            alert_type: Type of alert
        """
        self.alert_history[alert_type].append(datetime.utcnow())
    
    def get_time_until_next_alert(self, alert_type: str = "default") -> float:
        """
        Get seconds until next alert can be sent.
        
        Args:
            alert_type: Type of alert
            
        Returns:
            Seconds until next alert, 0 if alert can be sent now
        """
        if self.can_send(alert_type):
            return 0.0
        
        if not self.alert_history[alert_type]:
            return 0.0
        
        # Calculate time until oldest alert is outside window
        oldest_alert = min(self.alert_history[alert_type])
        time_until = (oldest_alert + self.time_window - datetime.utcnow()).total_seconds()
        return max(0.0, time_until)


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
            # Log error without exposing sensitive webhook URL
            print(f"Failed to send Slack alert: {type(e).__name__}")
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
            # Log error without exposing sensitive routing key
            print(f"Failed to send PagerDuty alert: {type(e).__name__}")
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
            # Log error without exposing sensitive webhook URL
            print(f"Failed to send webhook alert: {type(e).__name__}")
            return False


class AlertManager:
    """Manage multiple alert channels."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        max_alerts: int = 5,
        time_window_seconds: int = 300,
    ):
        """
        Initialize alert manager.
        
        Args:
            config_path: Path to configuration file (optional)
            max_alerts: Maximum number of alerts in time window
            time_window_seconds: Time window for rate limiting in seconds
        """
        self.alerters = []
        self.config = {}
        self.rate_limiter = RateLimiter(max_alerts, time_window_seconds)
        
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
    
    def send_alert(self, result: DriftResult, alert_type: str = "default") -> bool:
        """
        Send alert to all configured channels with rate limiting.
        
        Args:
            result: Drift detection result
            alert_type: Type of alert for rate limiting
            
        Returns:
            True if at least one alert was sent successfully
        """
        if not result.overall_drift:
            return True
        
        # Check rate limit
        if not self.rate_limiter.can_send(alert_type):
            time_until = self.rate_limiter.get_time_until_next_alert(alert_type)
            print(f"Rate limit exceeded. Next alert allowed in {time_until:.0f} seconds")
            return False
        
        success = False
        for alerter in self.alerters:
            if alerter.send(result):
                success = True
        
        # Record that an alert was sent
        if success:
            self.rate_limiter.record_alert(alert_type)
        
        return success
