"""Configuration management."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import re


class Config:
    """Configuration for drift-watchdog."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (default: watchdog.yaml)
        """
        self.config: Dict[str, Any] = {}
        
        if config_path:
            self.load(config_path)
        else:
            # Try to find watchdog.yaml in current directory
            default_path = Path("watchdog.yaml")
            if default_path.exists():
                self.load(str(default_path))
    
    def load(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}
        
        # Validate configuration after loading
        self._validate_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation, e.g., "baseline.path")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        # Expand environment variables
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, default)
        
        return value
    
    @property
    def baseline_path(self) -> str:
        """Get baseline path."""
        return self.get("baseline.path", "baselines/v1.json")
    
    @property
    def baseline_storage(self) -> str:
        """Get baseline storage type."""
        return self.get("baseline.storage", "local")
    
    @property
    def baseline_bucket(self) -> Optional[str]:
        """Get baseline bucket name."""
        return self.get("baseline.bucket")
    
    @property
    def detection_methods(self) -> List[str]:
        """Get detection methods."""
        return self.get("detection.methods", ["psi", "ks_test"])
    
    @property
    def psi_threshold(self) -> float:
        """Get PSI threshold."""
        return self.get("detection.thresholds.psi", 0.2)
    
    @property
    def ks_pvalue_threshold(self) -> float:
        """Get KS p-value threshold."""
        return self.get("detection.thresholds.ks_pvalue", 0.05)
    
    @property
    def exclude_features(self) -> List[str]:
        """Get features to exclude."""
        return self.get("detection.features.exclude", [])
    
    @property
    def alerts_config(self) -> Dict[str, Any]:
        """Get alerts configuration."""
        return self.get("alerts", {})
    
    @property
    def exporter_port(self) -> int:
        """Get exporter port."""
        return self.get("exporter.port", 9090)
    
    @property
    def exporter_interval(self) -> int:
        """Get exporter interval in seconds."""
        return self.get("exporter.interval_seconds", 300)
    
    def _validate_config(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate PSI threshold
        psi_threshold = self.get("detection.thresholds.psi", 0.2)
        if not isinstance(psi_threshold, (int, float)) or psi_threshold < 0 or psi_threshold > 1:
            raise ValueError("detection.thresholds.psi must be a number between 0 and 1")
        
        # Validate KS p-value threshold
        ks_threshold = self.get("detection.thresholds.ks_pvalue", 0.05)
        if not isinstance(ks_threshold, (int, float)) or ks_threshold < 0 or ks_threshold > 1:
            raise ValueError("detection.thresholds.ks_pvalue must be a number between 0 and 1")
        
        # Validate detection methods
        methods = self.get("detection.methods", ["psi", "ks_test"])
        valid_methods = {"psi", "ks_test", "jensen_shannon", "wasserstein", "chi_squared"}
        if not isinstance(methods, list):
            raise ValueError("detection.methods must be a list")
        for method in methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid detection method: {method}. Valid methods: {valid_methods}")
        
        # Validate exporter port
        port = self.get("exporter.port", 9090)
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError("exporter.port must be an integer between 1 and 65535")
        
        # Validate exporter interval
        interval = self.get("exporter.interval_seconds", 300)
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("exporter.interval_seconds must be a positive integer")
        
        # Validate storage type
        storage = self.get("baseline.storage", "local")
        valid_storage = {"local", "s3", "gcs"}
        if storage not in valid_storage:
            raise ValueError(f"Invalid storage type: {storage}. Valid types: {valid_storage}")
        
        # Validate webhook URLs if present
        if "alerts" in self.config:
            alerts = self.config["alerts"]
            
            # Validate Slack webhook URL format
            if "slack" in alerts and "webhook_url" in alerts["slack"]:
                webhook_url = alerts["slack"]["webhook_url"]
                if not self._is_valid_url(webhook_url):
                    raise ValueError("Invalid Slack webhook URL format")
            
            # Validate PagerDuty routing key format
            if "pagerduty" in alerts and "routing_key" in alerts["pagerduty"]:
                # PagerDuty routing keys should be alphanumeric
                routing_key = alerts["pagerduty"]["routing_key"]
                if not isinstance(routing_key, str) or not routing_key:
                    raise ValueError("PagerDuty routing key must be a non-empty string")
            
            # Validate webhook URL format
            if "webhook" in alerts and "url" in alerts["webhook"]:
                webhook_url = alerts["webhook"]["url"]
                if not self._is_valid_url(webhook_url):
                    raise ValueError("Invalid webhook URL format")
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not isinstance(url, str) or not url:
            return False
        
        # Basic URL validation regex
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))
