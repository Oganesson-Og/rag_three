"""
Test Logging Module
----------------

Tests for logging functionality, including event tracking and monitoring.
"""

import pytest
import logging
import json
from pathlib import Path
from datetime import datetime
from src.monitoring.logger import RagLogger
from src.monitoring.models import LogConfig, LogEvent, LogLevel

class TestLogging:
    
    @pytest.fixture
    def log_dir(self, tmp_path):
        """Fixture providing a temporary log directory."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        return log_dir
        
    @pytest.fixture
    def rag_logger(self, log_dir, config_manager):
        """Fixture providing a RAG logger instance."""
        log_config = LogConfig(
            log_dir=log_dir,
            file_name="rag.log",
            level=LogLevel.DEBUG,
            max_size_mb=10,
            backup_count=3
        )
        return RagLogger(config_manager, log_config)
        
    def test_basic_logging(self, rag_logger):
        """Test basic logging functionality."""
        # Log different levels
        rag_logger.debug("Debug message")
        rag_logger.info("Info message")
        rag_logger.warning("Warning message")
        rag_logger.error("Error message")
        
        logs = rag_logger.get_recent_logs()
        assert len(logs) == 4
        assert any("Debug message" in log.message for log in logs)
        assert any("Error message" in log.message for log in logs)
        
    def test_structured_logging(self, rag_logger):
        """Test structured logging with metadata."""
        metadata = {
            'user_id': '123',
            'query': 'test query',
            'duration_ms': 150
        }
        
        rag_logger.info("Search performed", extra=metadata)
        
        logs = rag_logger.get_recent_logs()
        last_log = logs[-1]
        assert last_log.metadata['user_id'] == '123'
        assert last_log.metadata['duration_ms'] == 150
        
    def test_log_rotation(self, rag_logger, log_dir):
        """Test log file rotation."""
        # Generate enough logs to trigger rotation
        large_message = "x" * 1000000  # 1MB message
        for _ in range(15):  # Should create multiple log files
            rag_logger.info(large_message)
            
        log_files = list(log_dir.glob("*.log*"))
        assert len(log_files) > 1
        assert all(file.stat().st_size <= 10 * 1024 * 1024 for file in log_files)  # 10MB limit
        
    def test_error_tracking(self, rag_logger):
        """Test error logging and tracking."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            rag_logger.error("Error occurred", exc_info=e)
            
        logs = rag_logger.get_recent_logs(level=LogLevel.ERROR)
        assert len(logs) == 1
        assert "ValueError" in logs[0].message
        assert logs[0].metadata.get('exc_info') is not None
        
    def test_performance_logging(self, rag_logger):
        """Test performance metric logging."""
        with rag_logger.track_duration() as duration:
            # Simulate some work
            import time
            time.sleep(0.1)
            
        logs = rag_logger.get_recent_logs()
        last_log = logs[-1]
        assert last_log.metadata['duration_ms'] >= 100  # At least 100ms
        
    def test_log_filtering(self, rag_logger):
        """Test log filtering functionality."""
        rag_logger.info("Test message 1", extra={'category': 'A'})
        rag_logger.info("Test message 2", extra={'category': 'B'})
        
        # Filter by metadata
        logs = rag_logger.get_filtered_logs(metadata_filter={'category': 'A'})
        assert len(logs) == 1
        assert logs[0].metadata['category'] == 'A'
        
    def test_log_export(self, rag_logger, tmp_path):
        """Test log export functionality."""
        rag_logger.info("Test message")
        
        export_path = tmp_path / "export.json"
        rag_logger.export_logs(export_path)
        
        with open(export_path) as f:
            exported_logs = json.load(f)
            
        assert len(exported_logs) > 0
        assert "Test message" in exported_logs[0]['message']
        
    def test_log_cleanup(self, rag_logger):
        """Test log cleanup functionality."""
        # Add old logs
        old_date = datetime(2020, 1, 1)
        rag_logger.info("Old message", timestamp=old_date)
        rag_logger.info("New message")
        
        # Cleanup old logs
        rag_logger.cleanup_old_logs(days=30)
        
        logs = rag_logger.get_recent_logs()
        assert all(log.timestamp.year >= datetime.now().year for log in logs)
        
    def test_context_logging(self, rag_logger):
        """Test context-based logging."""
        with rag_logger.log_context(operation='search', user_id='123'):
            rag_logger.info("Operation performed")
            
        logs = rag_logger.get_recent_logs()
        last_log = logs[-1]
        assert last_log.metadata['operation'] == 'search'
        assert last_log.metadata['user_id'] == '123'
        
    def test_batch_logging(self, rag_logger):
        """Test batch logging functionality."""
        events = [
            LogEvent(level=LogLevel.INFO, message="Event 1"),
            LogEvent(level=LogLevel.WARNING, message="Event 2")
        ]
        
        rag_logger.log_batch(events)
        
        logs = rag_logger.get_recent_logs()
        assert len(logs) >= 2
        assert any("Event 1" in log.message for log in logs)
        assert any("Event 2" in log.message for log in logs)
        
    def test_log_formatting(self, rag_logger):
        """Test log message formatting."""
        rag_logger.info("User {user} performed {action}", 
                       user="john", action="login")
        
        logs = rag_logger.get_recent_logs()
        assert "User john performed login" in logs[-1].message
        
    def test_error_handling(self, rag_logger):
        """Test error handling in logging operations."""
        # Test invalid log level
        with pytest.raises(ValueError):
            rag_logger.log("Invalid level", "message")
            
        # Test invalid metadata
        with pytest.raises(TypeError):
            rag_logger.info("Test", extra=lambda: None)  # Non-serializable
            
    def test_performance_impact(self, rag_logger):
        """Test logging performance impact."""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            rag_logger.debug("Performance test message")
            
        duration = time.time() - start_time
        assert duration < 1.0  # Should be fast
        
    def test_concurrent_logging(self, rag_logger):
        """Test concurrent logging operations."""
        import threading
        
        def log_operation(logger, thread_id):
            for i in range(100):
                logger.info(f"Thread {thread_id} message {i}")
                
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=log_operation,
                args=(rag_logger, i)
            )
            threads.append(thread)
            
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        logs = rag_logger.get_recent_logs()
        assert len(logs) >= 1000  # 10 threads * 100 messages 