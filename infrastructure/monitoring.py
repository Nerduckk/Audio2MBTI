"""
Monitoring and Metrics Module
Tracks performance metrics and success rates
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
from config_loader import ConfigLoader, get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: str
    name: str
    value: float
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self, metrics_file: str = None):
        if metrics_file is None:
            metrics_file = ConfigLoader.get("monitoring.metrics_file", "logs/metrics.json")
        
        self.metrics_file = metrics_file
        self.metrics: List[MetricPoint] = []
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, List[float]] = {}
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value"""
        point = MetricPoint(
            timestamp=datetime.now().isoformat(),
            name=name,
            value=value,
            tags=tags or {}
        )
        self.metrics.append(point)
        logger.debug(f"Metric: {name}={value}")
    
    def increment_counter(self, name: str, delta: int = 1):
        """Increment a counter"""
        self.counters[name] = self.counters.get(name, 0) + delta
    
    def start_timer(self, name: str) -> str:
        """Start timing something, returns timer_id"""
        timer_id = f"{name}_{id(self)}_{time.time()}"
        self.timers[timer_id] = [time.time()]
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing and return duration in seconds"""
        if timer_id not in self.timers:
            logger.warning(f"Timer {timer_id} not found")
            return 0
        
        self.timers[timer_id].append(time.time())
        duration = self.timers[timer_id][1] - self.timers[timer_id][0]
        
        # Extract metric name from timer_id
        metric_name = timer_id.split('_')[0]
        self.record_metric(f"{metric_name}_duration_seconds", duration)
        
        return duration
    
    def get_statistics(self, metric_name: str = None) -> Dict[str, Any]:
        """Get statistics for metrics"""
        if metric_name is None:
            # Get stats for all metrics
            metric_names = set(m.name for m in self.metrics)
        else:
            metric_names = {metric_name}
        
        stats = {}
        for name in metric_names:
            values = [m.value for m in self.metrics if m.name == name]
            
            if values:
                stats[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "sum": sum(values)
                }
        
        return stats
    
    def save_metrics(self):
        """Save collected metrics to file"""
        Path(self.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": [m.to_dict() for m in self.metrics],
            "counters": self.counters,
            "statistics": self.get_statistics()
        }
        
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


class PerformanceMonitor:
    """Monitors performance of processing pipeline"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.start_time = None
        self.end_time = None
        
        # Counters
        self.songs_processed = 0
        self.songs_successful = 0
        self.songs_failed = 0
        self.features_extracted = 0
        self.api_calls_made = 0
        self.api_errors = 0
        self.records_saved = 0
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
    
    def end(self):
        """End monitoring and generate report"""
        self.end_time = time.time()
        
        if self.start_time:
            duration = self.end_time - self.start_time
            logger.info(f"Processing completed in {duration:.1f}s")
            
            self.metrics.record_metric("total_duration_seconds", duration)
            self.metrics.record_metric("songs_processed", self.songs_processed)
            self.metrics.record_metric("success_rate_percent", 
                                      self._calculate_success_rate())
    
    def record_song_success(self, feature_count: int = 0):
        """Record successful song processing"""
        self.songs_processed += 1
        self.songs_successful += 1
        self.features_extracted += feature_count
        
        self.metrics.increment_counter("songs_successful")
    
    def record_song_failure(self, error: str = ""):
        """Record failed song processing"""
        self.songs_processed += 1
        self.songs_failed += 1
        
        self.metrics.increment_counter("songs_failed")
        logger.debug(f"Song processing failed: {error}")
    
    def record_api_call(self, api_name: str, success: bool = True, duration: float = 0):
        """Record API call"""
        self.api_calls_made += 1
        
        if not success:
            self.api_errors += 1
        
        self.metrics.record_metric(
            f"{api_name}_call_duration_seconds",
            duration,
            {"success": str(success)}
        )
    
    def record_record_saved(self):
        """Record data saved to CSV"""
        self.records_saved += 1
        self.metrics.increment_counter("records_saved")
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report"""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        else:
            duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "songs_processed": self.songs_processed,
            "songs_successful": self.songs_successful,
            "songs_failed": self.songs_failed,
            "success_rate_percent": self._calculate_success_rate(),
            "features_extracted": self.features_extracted,
            "api_calls": self.api_calls_made,
            "api_errors": self.api_errors,
            "records_saved": self.records_saved,
            "avg_time_per_song": duration / self.songs_processed if self.songs_processed > 0 else 0,
            "songs_per_minute": (self.songs_processed / duration * 60) if duration > 0 else 0
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.songs_processed == 0:
            return 0.0
        return (self.songs_successful / self.songs_processed) * 100
    
    def log_report(self):
        """Log performance report"""
        report = self.get_report()
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE REPORT")
        logger.info("=" * 60)
        logger.info(f"Duration: {report['duration_seconds']:.1f}s")
        logger.info(f"Songs Processed: {report['songs_processed']}")
        logger.info(f"Success Rate: {report['success_rate_percent']:.1f}%")
        logger.info(f"Avg Time/Song: {report['avg_time_per_song']:.2f}s")
        logger.info(f"Songs/Minute: {report['songs_per_minute']:.1f}")
        logger.info(f"API Calls: {report['api_calls']} ({report['api_errors']} errors)")
        logger.info(f"Records Saved: {report['records_saved']}")
        logger.info("=" * 60)
        
        return report
    
    def save_report(self, filename: str = None) -> str:
        """Save performance report to file"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        Path("logs").mkdir(exist_ok=True)
        filepath = f"logs/{filename}"
        
        report = self.get_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {filepath}")
        return filepath


def create_metrics_dashboard(metrics_files: List[str] = None) -> Dict[str, Any]:
    """
    Create dashboard data from metric files
    
    Args:
        metrics_files: List of metric JSON files to aggregate
        
    Returns:
        Aggregated metrics for dashboard
    """
    if metrics_files is None:
        metrics_files = list(Path("logs").glob("metrics_*.json"))
    
    dashboard = {
        "generated_at": datetime.now().isoformat(),
        "metrics_files": len(metrics_files),
        "aggregated_metrics": {}
    }
    
    all_values = {}
    
    for file in metrics_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            if "statistics" in data:
                for name, stats in data["statistics"].items():
                    if name not in all_values:
                        all_values[name] = []
                    all_values[name].extend([stats["mean"]])  # Use mean from each file
        except Exception as e:
            logger.warning(f"Could not read metrics file {file}: {e}")
    
    # Aggregate
    for name, values in all_values.items():
        if values:
            dashboard["aggregated_metrics"][name] = {
                "avg": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "samples": len(values)
            }
    
    return dashboard
