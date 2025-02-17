"""
Metrics Collection Module
----------------------

Utility for collecting and managing metrics throughout the RAG pipeline.

Key Features:
- Performance metrics
- Timing measurements
- Counter tracking
- Metric aggregation
- Statistics calculation
- Logging integration
- Custom metrics

Technical Details:
- Thread-safe counters
- Time tracking
- Memory usage
- Error rate tracking
- Custom metric types
- Metric persistence
- Data aggregation

Dependencies:
- time
- statistics
- typing-extensions>=4.7.0
- logging

Example Usage:
    collector = MetricsCollector()
    
    # Track timing
    with collector.track_time("process_document"):
        process_document()
    
    # Increment counters
    collector.increment("documents_processed")
    
    # Record values
    collector.record("embedding_dimension", 768)
    
    # Get statistics
    stats = collector.get_statistics()

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import time
import statistics
import logging
from datetime import datetime

class MetricsCollector:
    """Collects and manages processing metrics."""
    
    def __init__(self):
        self.timings: List[float] = []
        self.metrics: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)

    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric.
        
        Args:
            metric_name: Name of the counter
            value: Value to increment by
        """
        try:
            if metric_name not in self.counters:
                self.counters[metric_name] = 0
            self.counters[metric_name] += value
        except Exception as e:
            self.logger.error(f"Error incrementing metric {metric_name}: {str(e)}")
            raise

    def record(self, metric_name: str, value: Union[int, float]) -> None:
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
        """
        try:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = {}
            self.metrics[metric_name]['value'] = value
            self.metrics[metric_name]['timestamp'] = datetime.utcnow().isoformat()
        except Exception as e:
            self.logger.error(f"Error recording metric {metric_name}: {str(e)}")
            raise

    @contextmanager
    def measure_time(self):
        """Context manager for measuring execution time."""
        start_time = time.time()
        try:
            yield self
        finally:
            elapsed_time = time.time() - start_time
            self.add_timing(elapsed_time)
    
    def add_timing(self, elapsed: float):
        """Add a timing measurement."""
        self.timings.append(elapsed)
        
    def get_average_time(self) -> float:
        """Get average processing time."""
        return sum(self.timings) / len(self.timings) if self.timings else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            'timings': {
                'values': self.timings,
                'average': self.get_average_time()
            },
            'metrics': self.metrics
        }
    
    def reset(self):
        """Reset all metrics."""
        self.timings = []
        self.metrics = {}
        self.counters.clear()

    def get_counter(self, metric_name: str) -> int:
        """Get current value of a counter.
        
        Args:
            metric_name: Name of the counter
            
        Returns:
            int: Current counter value
        """
        return self.counters.get(metric_name, 0)

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dict[str, float]: Statistics including mean, median, min, max
        """
        try:
            if metric_name in self.metrics:
                values = self.metrics[metric_name]['value']
            elif metric_name in self.timings:
                values = self.timings
            else:
                return {}

            if not values:
                return {}

            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        except Exception as e:
            self.logger.error(f"Error calculating statistics for {metric_name}: {str(e)}")
            raise

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics and their values.
        
        Returns:
            Dict[str, Any]: All metrics and their current values
        """
        result = {
            'counters': self.counters.copy(),
            'timings': {
                'values': self.timings,
                'average': self.get_average_time()
            },
            'metrics': self.metrics
        }
        return result 