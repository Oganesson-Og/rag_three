"""
Metrics Collection Module
-----------------------

Comprehensive metrics collection system for monitoring RAG application performance
and system health.

Key Features:
- Query tracking
- System monitoring
- Performance metrics
- Custom metrics
- Alert management
- Metric aggregation
- Time series tracking

Technical Details:
- Context managers
- Time tracking
- System stats
- Metric storage
- Alert handling
- Data aggregation
- Performance optimization

Dependencies:
- psutil>=5.9.0
- typing (standard library)
- datetime (standard library)
- contextlib (standard library)

Example Usage:
    # Initialize collector
    collector = MetricsCollector(
        config_manager=config_manager,
        metrics_config=metrics_config
    )
    
    # Track query performance
    with collector.track_query("search query") as metrics:
        # Perform query
        results = search_index(query)
        metrics.num_chunks = len(results)
    
    # Collect system metrics
    collector.collect_system_metrics()

Metrics Types:
- Query Performance
- System Resources
- Operation Timing
- Custom Metrics
- Alerts

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from .models import MetricsConfig, QueryMetrics, SystemMetrics, PerformanceMetrics

class MetricsCollector:
    def __init__(self, config_manager, metrics_config: MetricsConfig):
        self.config_manager = config_manager
        self.config = metrics_config
        self.metrics = []
        self.custom_metrics = {}
        self.alerts = {}
        
    @contextmanager
    def track_query(self, query_text: str, labels: Dict[str, str] = None):
        start_time = time.time()
        query_metrics = QueryMetrics(
            timestamp=datetime.now(),
            query_text=query_text,
            labels=labels or {}
        )
        try:
            yield query_metrics
        finally:
            query_metrics.duration_ms = (time.time() - start_time) * 1000
            self.metrics.append(query_metrics)
            
    def collect_system_metrics(self):
        import psutil
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent
        )
        self.metrics.append(metrics)
        
    @contextmanager
    def track_operation(self, operation_name: str):
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                timestamp=datetime.now(),
                operation_name=operation_name,
                duration_ms=duration * 1000
            )) 