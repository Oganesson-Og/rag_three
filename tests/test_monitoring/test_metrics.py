"""
Test Metrics Module
----------------

Tests for metrics collection, performance monitoring, and analytics.
"""

import pytest
import time
from datetime import datetime, timedelta
from src.monitoring.metrics import MetricsCollector
from src.monitoring.models import (
    MetricsConfig,
    QueryMetrics,
    SystemMetrics,
    PerformanceMetrics
)

class TestMetrics:
    
    @pytest.fixture
    def metrics_collector(self, config_manager):
        """Fixture providing a metrics collector instance."""
        metrics_config = MetricsConfig(
            enabled=True,
            collection_interval=1,  # 1 second
            retention_days=7
        )
        return MetricsCollector(config_manager, metrics_config)
        
    def test_query_metrics(self, metrics_collector):
        """Test query-related metrics collection."""
        # Simulate query execution
        with metrics_collector.track_query("test query") as query_metrics:
            time.sleep(0.1)  # Simulate processing
            query_metrics.update(
                num_chunks=5,
                num_tokens=100,
                cache_hit=False
            )
            
        metrics = metrics_collector.get_query_metrics()
        last_query = metrics[-1]
        
        assert last_query.query_text == "test query"
        assert last_query.duration_ms >= 100
        assert last_query.num_chunks == 5
        assert last_query.num_tokens == 100
        assert not last_query.cache_hit
        
    def test_system_metrics(self, metrics_collector):
        """Test system metrics collection."""
        metrics_collector.collect_system_metrics()
        
        metrics = metrics_collector.get_system_metrics()
        last_metrics = metrics[-1]
        
        assert 0 <= last_metrics.cpu_usage <= 100
        assert last_metrics.memory_usage > 0
        assert last_metrics.disk_usage > 0
        
    def test_performance_metrics(self, metrics_collector):
        """Test performance metrics collection."""
        # Simulate multiple operations
        operations = [
            ("embedding", 0.1),
            ("search", 0.2),
            ("preprocessing", 0.15)
        ]
        
        for op_name, duration in operations:
            with metrics_collector.track_operation(op_name):
                time.sleep(duration)
                
        metrics = metrics_collector.get_performance_metrics()
        
        assert len(metrics) >= 3
        assert all(m.operation_name in ["embedding", "search", "preprocessing"] 
                  for m in metrics)
        
    def test_metrics_aggregation(self, metrics_collector):
        """Test metrics aggregation functionality."""
        # Generate sample metrics
        for _ in range(5):
            with metrics_collector.track_query("test query"):
                time.sleep(0.1)
                
        # Get aggregated metrics
        aggregated = metrics_collector.get_aggregated_metrics(
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now()
        )
        
        assert aggregated.total_queries == 5
        assert aggregated.avg_query_time > 0
        assert aggregated.success_rate >= 0
        
    def test_metrics_filtering(self, metrics_collector):
        """Test metrics filtering capabilities."""
        # Generate metrics with different labels
        with metrics_collector.track_query("query1", labels={"type": "A"}):
            pass
        with metrics_collector.track_query("query2", labels={"type": "B"}):
            pass
            
        # Filter metrics
        filtered = metrics_collector.filter_metrics(labels={"type": "A"})
        assert len(filtered) == 1
        assert filtered[0].labels["type"] == "A"
        
    def test_metrics_export(self, metrics_collector, tmp_path):
        """Test metrics export functionality."""
        # Generate some metrics
        with metrics_collector.track_query("test query"):
            pass
            
        export_path = tmp_path / "metrics_export.json"
        metrics_collector.export_metrics(export_path)
        
        assert export_path.exists()
        
    def test_metrics_cleanup(self, metrics_collector):
        """Test metrics cleanup functionality."""
        # Add old metrics
        old_time = datetime.now() - timedelta(days=10)
        metrics_collector._add_metric(
            QueryMetrics(
                timestamp=old_time,
                query_text="old query",
                duration_ms=100
            )
        )
        
        # Cleanup old metrics
        metrics_collector.cleanup_old_metrics()
        
        # Verify old metrics are removed
        all_metrics = metrics_collector.get_query_metrics()
        assert all(m.timestamp > datetime.now() - timedelta(days=7) 
                  for m in all_metrics)
        
    def test_concurrent_metrics(self, metrics_collector):
        """Test concurrent metrics collection."""
        import threading
        
        def collect_metrics(collector, thread_id):
            with collector.track_query(f"thread_{thread_id}_query"):
                time.sleep(0.1)
                
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=collect_metrics,
                args=(metrics_collector, i)
            )
            threads.append(thread)
            
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        metrics = metrics_collector.get_query_metrics()
        assert len(metrics) >= 5
        
    def test_metrics_persistence(self, metrics_collector, tmp_path):
        """Test metrics persistence across sessions."""
        # Generate metrics
        with metrics_collector.track_query("test query"):
            pass
            
        # Save metrics
        metrics_collector.save_metrics(tmp_path / "metrics.db")
        
        # Create new collector and load metrics
        new_collector = MetricsCollector(
            metrics_collector.config_manager,
            MetricsConfig(enabled=True)
        )
        new_collector.load_metrics(tmp_path / "metrics.db")
        
        assert len(new_collector.get_query_metrics()) > 0
        
    def test_custom_metrics(self, metrics_collector):
        """Test custom metrics collection."""
        # Register custom metric
        metrics_collector.register_custom_metric(
            "custom_metric",
            lambda: {"value": 42}
        )
        
        # Collect custom metrics
        metrics_collector.collect_custom_metrics()
        
        metrics = metrics_collector.get_custom_metrics()
        assert "custom_metric" in metrics
        assert metrics["custom_metric"]["value"] == 42
        
    def test_metrics_alerts(self, metrics_collector):
        """Test metrics-based alerting."""
        def alert_callback(metric_name, value, threshold):
            pass
            
        # Register alert
        metrics_collector.register_alert(
            metric_name="cpu_usage",
            threshold=90,
            callback=alert_callback
        )
        
        # Simulate high CPU usage
        metrics_collector._add_metric(
            SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=95,
                memory_usage=50,
                disk_usage=60
            )
        )
        
        # Check alerts
        alerts = metrics_collector.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].metric_name == "cpu_usage"
        
    def test_performance_profiling(self, metrics_collector):
        """Test detailed performance profiling."""
        with metrics_collector.profile_operation("complex_operation") as profiler:
            with profiler.phase("phase1"):
                time.sleep(0.1)
            with profiler.phase("phase2"):
                time.sleep(0.2)
                
        profile = metrics_collector.get_operation_profile("complex_operation")
        assert len(profile.phases) == 2
        assert profile.phases["phase2"].duration > profile.phases["phase1"].duration 