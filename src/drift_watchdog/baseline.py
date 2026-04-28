"""Baseline storage for drift detection."""

import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from drift_watchdog.models import Baseline
from drift_watchdog.statistics import calculate_feature_statistics


class BaselineStore:
    """Store and retrieve baselines from various backends."""
    
    def __init__(self, path: str, storage_type: str = "local"):
        """
        Initialize baseline store.
        
        Args:
            path: Path to baseline (local file path, s3://, or gs://)
            storage_type: Storage backend type (local, s3, gcs)
        """
        self.path = path
        self.storage_type = storage_type
        
        # Auto-detect storage type from path if not specified
        if storage_type == "auto":
            if path.startswith("s3://"):
                self.storage_type = "s3"
            elif path.startswith("gs://"):
                self.storage_type = "gcs"
            else:
                self.storage_type = "local"
    
    def save(self, baseline: Baseline) -> None:
        """
        Save baseline to storage.
        
        Args:
            baseline: Baseline to save
        """
        if self.storage_type == "local":
            self._save_local(baseline)
        elif self.storage_type == "s3":
            self._save_s3(baseline)
        elif self.storage_type == "gcs":
            self._save_gcs(baseline)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def load(self) -> Baseline:
        """
        Load baseline from storage.
        
        Returns:
            Loaded baseline
        """
        if self.storage_type == "local":
            return self._load_local()
        elif self.storage_type == "s3":
            return self._load_s3()
        elif self.storage_type == "gcs":
            return self._load_gcs()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _save_local(self, baseline: Baseline) -> None:
        """Save baseline to local file."""
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2)
    
    def _load_local(self) -> Baseline:
        """Load baseline from local file."""
        with open(self.path, "r") as f:
            data = json.load(f)
        return Baseline.from_dict(data)
    
    def _save_s3(self, baseline: Baseline) -> None:
        """Save baseline to S3."""
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        
        parsed = urlparse(self.path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        
        s3 = boto3.client("s3")
        
        try:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(baseline.to_dict(), indent=2),
                ContentType="application/json",
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to save baseline to S3: {e}")
    
    def _load_s3(self) -> Baseline:
        """Load baseline from S3."""
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        
        parsed = urlparse(self.path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        
        s3 = boto3.client("s3")
        
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return Baseline.from_dict(data)
        except ClientError as e:
            raise RuntimeError(f"Failed to load baseline from S3: {e}")
    
    def _save_gcs(self, baseline: Baseline) -> None:
        """Save baseline to GCS."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            )
        
        parsed = urlparse(self.path)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_string(json.dumps(baseline.to_dict(), indent=2), content_type="application/json")
    
    def _load_gcs(self) -> Baseline:
        """Load baseline from GCS."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            )
        
        parsed = urlparse(self.path)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        data = json.loads(blob.download_as_text())
        return Baseline.from_dict(data)
    
    @staticmethod
    def create_from_dataframe(
        df: pd.DataFrame,
        name: str,
        exclude_features: Optional[list[str]] = None,
    ) -> Baseline:
        """
        Create a baseline from a DataFrame.
        
        Args:
            df: Reference data DataFrame
            name: Baseline name
            exclude_features: Features to exclude from baseline
            
        Returns:
            Baseline object
        """
        exclude_features = exclude_features or []
        feature_names = [col for col in df.columns if col not in exclude_features]
        
        statistics = {}
        for feature in feature_names:
            statistics[feature] = calculate_feature_statistics(df[feature])
        
        return Baseline(
            name=name,
            statistics=statistics,
            feature_names=feature_names,
        )
    
    @staticmethod
    def create_from_csv(
        csv_path: str,
        name: str,
        exclude_features: Optional[list[str]] = None,
    ) -> Baseline:
        """
        Create a baseline from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            name: Baseline name
            exclude_features: Features to exclude from baseline
            
        Returns:
            Baseline object
        """
        df = pd.read_csv(csv_path)
        return BaselineStore.create_from_dataframe(df, name, exclude_features)
