"""
Batch Processing Module
Optimizes CSV writes and data processing with batching
"""

import pandas as pd
import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config_loader import ConfigLoader, get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """Batch process and save data efficiently"""
    
    def __init__(self, batch_size: int = None, output_file: str = None):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of records to batch before saving
            output_file: Output CSV file path
        """
        if batch_size is None:
            batch_size = ConfigLoader.get("data.batch_size", 25)
        
        self.batch_size = batch_size
        self.output_file = output_file
        self.batch = []
        self.total_processed = 0
        self.total_saved = 0
        self.file_exists = False
        self._lock = threading.Lock()
        
        if output_file and os.path.exists(output_file):
            self.file_exists = True
    
    def add(self, record: Dict[str, Any]) -> bool:
        """
        Add record to batch
        
        Args:
            record: Dictionary representing one record
            
        Returns:
            True if batch was flushed, False otherwise
        """
        should_flush = False
        with self._lock:
            self.batch.append(record)
            self.total_processed += 1
            should_flush = len(self.batch) >= self.batch_size

        if should_flush:
            self.flush()
            return True
        return False
    
    def flush(self) -> int:
        """
        Write current batch to file
        
        Returns:
            Number of records written
        """
        if not self.output_file:
            return 0

        with self._lock:
            if not self.batch:
                return 0
            batch_to_write = self.batch
            self.batch = []

        try:
            df = pd.DataFrame(batch_to_write)

            if self.file_exists:
                df.to_csv(
                    self.output_file,
                    mode='a',
                    header=False,
                    index=False,
                    encoding='utf-8'
                )
            else:
                df.to_csv(
                    self.output_file,
                    index=False,
                    encoding='utf-8'
                )
                self.file_exists = True

            records_written = len(batch_to_write)
            with self._lock:
                self.total_saved += records_written

            logger.info(f"Batch saved: {records_written} records to {self.output_file} "
                       f"(Total: {self.total_saved}/{self.total_processed})")
            return records_written

        except Exception as e:
            with self._lock:
                self.batch = batch_to_write + self.batch
            logger.error(f"Failed to save batch: {e}")
            return 0
    
    def close(self) -> int:
        """Flush any remaining records and close"""
        return self.flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DataQualityMonitor:
    """Monitor data quality throughout processing"""
    
    def __init__(self):
        self.metrics = {
            "start_time": datetime.now(),
            "records_processed": 0,
            "records_valid": 0,
            "records_invalid": 0,
            "features_extracted": 0,
            "errors": {},
            "warnings": [],
            "performance": {
                "avg_time_per_record": 0,
                "total_duration": 0
            }
        }
    
    def record_valid(self):
        """Record a valid data point"""
        self.metrics["records_valid"] += 1
        self.metrics["records_processed"] += 1
    
    def record_invalid(self, error_type: str, details: str = ""):
        """Record an invalid data point"""
        self.metrics["records_invalid"] += 1
        self.metrics["records_processed"] += 1
        
        if error_type not in self.metrics["errors"]:
            self.metrics["errors"][error_type] = []
        
        self.metrics["errors"][error_type].append(details)
    
    def record_features(self, count: int):
        """Record feature extraction count"""
        self.metrics["features_extracted"] += count
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.metrics["warnings"].append(message)
        logger.warning(message)
    
    def finalize(self) -> Dict[str, Any]:
        """Calculate final metrics"""
        duration = (datetime.now() - self.metrics["start_time"]).total_seconds()
        
        self.metrics["performance"]["total_duration"] = duration
        if self.metrics["records_processed"] > 0:
            self.metrics["performance"]["avg_time_per_record"] = \
                duration / self.metrics["records_processed"]
        
        self.metrics["quality_score"] = self._calculate_quality_score()
        self.metrics["end_time"] = datetime.now()
        
        return self.metrics
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if self.metrics["records_processed"] == 0:
            return 0.0
        
        valid_ratio = self.metrics["records_valid"] / self.metrics["records_processed"]
        error_count = sum(len(errors) for errors in self.metrics["errors"].values())
        warning_count = len(self.metrics["warnings"])
        
        # Base score on valid ratio (80% weight)
        score = valid_ratio * 80
        
        # Deduct for errors (15% weight)
        if error_count > 0:
            error_penalty = min(15, error_count * 0.5)
            score -= error_penalty
        
        # Deduct for warnings (5% weight)
        if warning_count > 0:
            warning_penalty = min(5, warning_count * 0.1)
            score -= warning_penalty
        
        return max(0, min(100, score))
    
    def save_report(self, filename: str = None) -> str:
        """Save monitoring report to JSON"""
        if filename is None:
            filename = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs("logs", exist_ok=True)
        filepath = f"logs/{filename}"
        
        metrics = self.finalize()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Quality report saved to {filepath}")
        self._log_summary(metrics)
        
        return filepath
    
    def _log_summary(self, metrics: Dict[str, Any]):
        """Log summary statistics"""
        logger.info("=" * 60)
        logger.info("DATA QUALITY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Records Processed: {metrics['records_processed']}")
        logger.info(f"Records Valid: {metrics['records_valid']}")
        logger.info(f"Records Invalid: {metrics['records_invalid']}")
        logger.info(f"Quality Score: {metrics['quality_score']:.1f}/100")
        logger.info(f"Total Duration: {metrics['performance']['total_duration']:.1f}s")
        
        if metrics['records_processed'] > 0:
            logger.info(f"Avg Time/Record: {metrics['performance']['avg_time_per_record']:.2f}s")
        
        if metrics['errors']:
            logger.info("Errors by Type:")
            for error_type, errors in metrics['errors'].items():
                logger.info(f"  {error_type}: {len(errors)} occurrences")
        
        logger.info("=" * 60)

