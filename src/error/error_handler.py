"""
RAG Error Handler Module
----------------------

Centralized error handling system for RAG operations with logging,
tracking, and recovery capabilities.

Key Features:
- Centralized error handling
- Error logging and tracking
- Recovery strategies
- Fallback operations
- Error categorization
- Context preservation
- Batch error handling

Technical Details:
- Custom exception handling
- Error aggregation
- Logging integration
- Recovery mechanisms
- Context management
- Performance monitoring

Dependencies:
- logging (standard library)
- datetime (standard library)
- typing (standard library)
- .models (local module)

Example Usage:
    # Initialize handler
    handler = ErrorHandler(config_manager)
    
    # Handle specific error
    try:
        process_document()
    except Exception as e:
        error_info = handler.handle_error(e)
    
    # Execute with fallback
    result = handler.execute_with_fallback(
        main_operation=process_doc,
        fallback_operation=process_doc_simple
    )

Error Handling Features:
- Custom error types
- Error categorization
- Recovery strategies
- Logging integration
- Performance tracking
- Context preservation

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from .models import RAGError, ErrorInfo, ErrorReport, ErrorSeverity

class ErrorHandler:
    """Centralized error handling system for RAG operations."""
    
    def __init__(self, config_manager):
        """Initialize the error handler with configuration."""
        self.config_manager = config_manager
        self.error_logs: List[ErrorInfo] = []
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("rag_error_handler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: Union[str, ErrorSeverity] = ErrorSeverity.ERROR
    ) -> ErrorInfo:
        """Handle and log an error with context."""
        if isinstance(error, RAGError):
            error_info = ErrorInfo.from_exception(error)
        else:
            error_info = ErrorInfo(
                error_type=type(error).__name__,
                message=str(error),
                timestamp=datetime.now(),
                context=context or {},
                severity=ErrorSeverity(severity) if isinstance(severity, str) else severity
            )
            
        self.error_logs.append(error_info)
        self.logger.error(
            f"{error_info.error_type}: {error_info.message}",
            extra={"context": error_info.context}
        )
        
        return error_info
        
    def handle_rag_error(self, error: RAGError) -> ErrorInfo:
        """Handle specific RAG-related errors."""
        return self.handle_error(error)
        
    def execute_with_fallback(
        self,
        main_operation: Callable,
        fallback_operation: Callable,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute an operation with fallback support."""
        try:
            return main_operation()
        except Exception as e:
            error_info = self.handle_error(e, context)
            self.logger.warning(
                f"Main operation failed, attempting fallback: {error_info.message}"
            )
            try:
                return fallback_operation()
            except Exception as fallback_error:
                self.handle_error(
                    fallback_error,
                    context,
                    severity=ErrorSeverity.CRITICAL
                )
                raise
                
    def get_error_logs(self) -> List[ErrorInfo]:
        """Retrieve all logged errors."""
        return self.error_logs
        
    def get_error_report(self) -> ErrorReport:
        """Generate a comprehensive error report."""
        return ErrorReport.from_error_list(self.error_logs)
        
    def clear_logs(self):
        """Clear all error logs."""
        self.error_logs.clear()
        
    def categorize_errors(self, errors: List[Exception]) -> Dict[str, List[Exception]]:
        """Categorize errors by type."""
        categorized = {}
        for error in errors:
            category = error.__class__.__name__.lower().replace('error', '')
            categorized.setdefault(category, []).append(error)
        return categorized 