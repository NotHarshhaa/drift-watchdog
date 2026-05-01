"""Audit logging for drift detection operations."""

import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from drift_watchdog.models import DriftResult


@dataclass
class AuditLogEntry:
    """Audit log entry for drift detection."""
    timestamp: str
    operation: str
    baseline_version: Optional[str]
    overall_drift: bool
    overall_score: float
    features_checked: int
    features_drifted: int
    drifted_features: list
    status: str  # "success", "error"
    error_message: Optional[str] = None


class AuditLogger:
    """Audit logger for drift detection operations."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
    ):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to log file (optional)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_console: Whether to log to console
        """
        self.log_file = log_file
        self.logger = logging.getLogger("drift_watchdog.audit")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler if log file specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_drift_check(
        self,
        result: DriftResult,
        operation: str = "drift_check",
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """
        Log a drift check operation.
        
        Args:
            result: Drift detection result
            operation: Operation type
            status: Operation status
            error_message: Error message if operation failed
        """
        drifted_features = [
            name for name, report in result.features.items()
            if report.is_drift
        ]
        
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            operation=operation,
            baseline_version=result.baseline_version,
            overall_drift=result.overall_drift,
            overall_score=result.overall_score,
            features_checked=len(result.features),
            features_drifted=len(drifted_features),
            drifted_features=drifted_features,
            status=status,
            error_message=error_message,
        )
        
        # Log as structured JSON
        log_message = json.dumps(asdict(entry), indent=2)
        
        if status == "success":
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)
    
    def log_baseline_operation(
        self,
        operation: str,
        baseline_name: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Log a baseline operation.
        
        Args:
            operation: Operation type (create, load, save, delete)
            baseline_name: Name of the baseline
            status: Operation status
            error_message: Error message if operation failed
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "baseline_name": baseline_name,
            "status": status,
            "error_message": error_message,
        }
        
        log_message = json.dumps(entry, indent=2)
        
        if status == "success":
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)
    
    def log_alert_sent(
        self,
        alert_type: str,
        channel: str,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Log an alert sending operation.
        
        Args:
            alert_type: Type of alert sent
            channel: Alert channel (slack, pagerduty, webhook)
            success: Whether alert was sent successfully
            error_message: Error message if alert failed
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "alert_sent",
            "alert_type": alert_type,
            "channel": channel,
            "success": success,
            "error_message": error_message,
        }
        
        log_message = json.dumps(entry, indent=2)
        
        if success:
            self.logger.info(log_message)
        else:
            self.logger.warning(log_message)
