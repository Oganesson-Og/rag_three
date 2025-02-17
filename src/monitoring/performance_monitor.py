"""
Performance Monitoring Module
--------------------------

Comprehensive performance monitoring system for the educational RAG pipeline.
Tracks metrics, latency, resource usage, and system health.

Key Features:
- Real-time metric tracking
- Resource utilization monitoring
- Query performance analysis
- System health checks
- Automated alerting
- Performance logging
- Trend analysis
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import psutil
import logging
import time
from dataclasses import dataclass
from collections import deque
import threading
import numpy as np
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latency: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float]
    query_count: int
    error_count: int
    cache_hits: int
    cache_misses: int
    vector_db_latency: float
    embedding_latency: float
    timestamp: datetime

class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(
        self,
        metrics_window: int = 3600,  # 1 hour default
        sampling_rate: int = 60,     # 60 seconds default
        log_dir: Optional[Path] = None
    ):
        self.metrics_window = metrics_window
        self.sampling_rate = sampling_rate
        self.log_dir = log_dir or Path("logs/performance")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history: deque = deque(maxlen=self.metrics_window)
        self.current_metrics: Dict[str, float] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _setup_logging(self) -> None:
        """Configure performance logging."""
        handler = logging.FileHandler(self.log_dir / "performance.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self._analyze_metrics(metrics)
                time.sleep(self.sampling_rate)
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        return PerformanceMetrics(
            latency=self.current_metrics.get('latency', 0.0),
            memory_usage=psutil.Process().memory_percent(),
            cpu_usage=psutil.cpu_percent(),
            gpu_usage=self._get_gpu_usage(),
            query_count=self.current_metrics.get('query_count', 0),
            error_count=self.current_metrics.get('error_count', 0),
            cache_hits=self.current_metrics.get('cache_hits', 0),
            cache_misses=self.current_metrics.get('cache_misses', 0),
            vector_db_latency=self.current_metrics.get('vector_db_latency', 0.0),
            embedding_latency=self.current_metrics.get('embedding_latency', 0.0),
            timestamp=datetime.now()
        )

    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except:
            return None

    def _analyze_metrics(self, metrics: PerformanceMetrics) -> None:
        """Analyze metrics and trigger alerts if needed."""
        # Check for concerning patterns
        if metrics.memory_usage > 90:
            self.logger.warning("High memory usage detected")
        if metrics.error_count > 100:
            self.logger.error("High error rate detected")
        if metrics.latency > 1.0:  # 1 second
            self.logger.warning("High latency detected")

    def record_query(self, latency: float, is_cache_hit: bool = False) -> None:
        """Record query performance."""
        self.current_metrics['latency'] = latency
        self.current_metrics['query_count'] = self.current_metrics.get('query_count', 0) + 1
        if is_cache_hit:
            self.current_metrics['cache_hits'] = self.current_metrics.get('cache_hits', 0) + 1
        else:
            self.current_metrics['cache_misses'] = self.current_metrics.get('cache_misses', 0) + 1

    def record_error(self, error_type: str) -> None:
        """Record error occurrence."""
        self.current_metrics['error_count'] = self.current_metrics.get('error_count', 0) + 1
        self.logger.error(f"Error recorded: {error_type}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        metrics_list = list(self.metrics_history)
        return {
            "summary": {
                "avg_latency": np.mean([m.latency for m in metrics_list]),
                "avg_memory_usage": np.mean([m.memory_usage for m in metrics_list]),
                "avg_cpu_usage": np.mean([m.cpu_usage for m in metrics_list]),
                "total_queries": sum(m.query_count for m in metrics_list),
                "error_rate": sum(m.error_count for m in metrics_list) / len(metrics_list),
                "cache_hit_rate": self._calculate_cache_hit_rate(metrics_list)
            },
            "trends": {
                "latency_trend": self._calculate_trend([m.latency for m in metrics_list]),
                "memory_trend": self._calculate_trend([m.memory_usage for m in metrics_list])
            },
            "timeframe": {
                "start": metrics_list[0].timestamp,
                "end": metrics_list[-1].timestamp
            }
        }

    def _calculate_cache_hit_rate(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate cache hit rate."""
        total_hits = sum(m.cache_hits for m in metrics_list)
        total_misses = sum(m.cache_misses for m in metrics_list)
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "insufficient_data"
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        return "stable"

    def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
        self.logger.info("Performance monitoring stopped") 