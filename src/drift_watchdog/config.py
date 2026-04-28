"""Configuration management."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


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
