"""
Test Error Handling
----------------

Tests for system-wide error handling and recovery mechanisms.
"""

import pytest
import logging
from src.error.error_handler import ErrorHandler
from src.error.models import (
    RAGError,
    ProcessingError,
    EmbeddingError,
    SearchError,
    ValidationError
)

class TestErrorHandling:
    
    @pytest.fixture
    def error_handler(self, config_manager):
        """Fixture providing an error handler instance."""
        return ErrorHandler(config_manager)
        
    @pytest.fixture
    def sample_error_context(self):
        """Fixture providing sample error context."""
        return {
            'operation': 'document_processing',
            'input': 'test_document.txt',
            'timestamp': '2024-01-01T12:00:00Z'
        }
        
    def test_basic_error_handling(self, error_handler):
        """Test basic error catching and handling."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_info = error_handler.handle_error(e)
            
        assert error_info.error_type == "ValueError"
        assert error_info.message == "Test error"
        assert error_info.timestamp is not None
        
    def test_custom_error_handling(self, error_handler):
        """Test handling of custom RAG errors."""
        try:
            raise ProcessingError("Document processing failed")
        except RAGError as e:
            error_info = error_handler.handle_rag_error(e)
            
        assert error_info.error_type == "ProcessingError"
        assert "processing failed" in error_info.message
        assert error_info.is_recoverable
        
    def test_error_recovery(self, error_handler):
        """Test error recovery mechanisms."""
        def failing_operation():
            raise SearchError("Search index unavailable")
            
        def fallback_operation():
            return "Fallback result"
            
        result = error_handler.execute_with_fallback(
            failing_operation,
            fallback_operation
        )
        
        assert result == "Fallback result"
        
    def test_error_logging(self, error_handler, sample_error_context):
        """Test error logging functionality."""
        error = ValidationError("Invalid input format")
        
        error_handler.log_error(error, context=sample_error_context)
        
        logs = error_handler.get_error_logs()
        assert len(logs) > 0
        assert logs[-1].error_type == "ValidationError"
        assert logs[-1].context == sample_error_context
        
    def test_error_categorization(self, error_handler):
        """Test error categorization and prioritization."""
        errors = [
            ProcessingError("Processing failed"),
            EmbeddingError("Embedding generation failed"),
            ValidationError("Invalid input")
        ]
        
        categorized = error_handler.categorize_errors(errors)
        
        assert "processing" in categorized
        assert "embedding" in categorized
        assert "validation" in categorized
        
    def test_batch_error_handling(self, error_handler):
        """Test handling of multiple errors in batch operations."""
        operations = [
            lambda: exec("invalid syntax"),
            lambda: 1/0,
            lambda: int("not a number")
        ]
        
        results = error_handler.execute_batch(operations)
        
        assert len(results.failures) == 3
        assert "SyntaxError" in str(results.failures[0])
        assert "ZeroDivisionError" in str(results.failures[1])
        assert "ValueError" in str(results.failures[2])
        
    def test_error_context_preservation(self, error_handler):
        """Test preservation of error context."""
        with error_handler.error_context(operation="test_operation") as context:
            try:
                raise ValueError("Test error")
            except Exception as e:
                context.record_error(e)
                
        assert context.has_errors()
        assert context.get_errors()[0].operation == "test_operation"
        
    def test_error_notification(self, error_handler):
        """Test error notification system."""
        critical_error = ProcessingError(
            "Critical system failure",
            severity="critical"
        )
        
        notifications = error_handler.process_error_notifications(critical_error)
        
        assert len(notifications) > 0
        assert notifications[0].severity == "critical"
        
    def test_error_rate_monitoring(self, error_handler):
        """Test error rate monitoring."""
        # Generate some errors
        for _ in range(5):
            try:
                raise ValueError("Test error")
            except Exception as e:
                error_handler.handle_error(e)
                
        error_rate = error_handler.get_error_rate(window_minutes=5)
        assert error_rate > 0
        
        # Check if error threshold is exceeded
        assert error_handler.is_error_threshold_exceeded(
            max_errors=10,
            window_minutes=5
        ) == False
        
    def test_error_recovery_strategies(self, error_handler):
        """Test different error recovery strategies."""
        strategies = {
            'retry': lambda: error_handler.retry_operation(
                lambda: 1/0,
                max_attempts=3
            ),
            'fallback': lambda: error_handler.use_fallback(
                lambda: 1/0,
                fallback_value=0
            ),
            'circuit_breaker': lambda: error_handler.with_circuit_breaker(
                lambda: 1/0
            )
        }
        
        for strategy_name, strategy in strategies.items():
            try:
                strategy()
            except Exception as e:
                assert error_handler.was_recovery_attempted(e)
                
    def test_error_aggregation(self, error_handler):
        """Test error aggregation and reporting."""
        errors = [
            ValueError("Error 1"),
            ValueError("Error 2"),
            TypeError("Error 3")
        ]
        
        report = error_handler.generate_error_report(errors)
        
        assert report.total_errors == 3
        assert report.error_types['ValueError'] == 2
        assert report.error_types['TypeError'] == 1
        
    def test_cascading_error_handling(self, error_handler):
        """Test handling of cascading errors."""
        def operation_with_cascading_errors():
            try:
                raise ValueError("Primary error")
            except Exception as e:
                try:
                    raise ProcessingError("Secondary error") from e
                except Exception as e2:
                    return error_handler.handle_cascading_errors(e2)
                    
        result = operation_with_cascading_errors()
        assert len(result.error_chain) == 2
        assert "Primary error" in str(result.error_chain[0]) 