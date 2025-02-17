"""
RAG Error Models Module
---------------------

Comprehensive error handling system for RAG operations with structured error types,
tracking, and reporting capabilities.

Key Features:
- Hierarchical error classification
- Detailed error context tracking
- Severity levels support
- Error recovery hints
- Structured error reporting
- Timestamp tracking
- Context preservation

Technical Details:
- Custom exception hierarchy
- Dataclass-based error info
- Temporal error tracking
- Context serialization
- Recovery classification
- Aggregated reporting

Dependencies:
- dataclasses (standard library)
- datetime (standard library)
- typing (standard library)

Example Usage:
    # Raise a specific error
    raise ProcessingError("Failed to process PDF", severity="critical")
    
    # Create error info
    error_info = ErrorInfo(
        error_type="ProcessingError",
        message="PDF processing failed",
        timestamp=datetime.now(),
        context={"file": "doc.pdf"}
    )
    
    # Generate error report
    report = ErrorReport.from_error_list(error_list)

Error Categories:
- Processing Errors
- Embedding Errors
- Search Errors
- Validation Errors
- Configuration Errors
- Pipeline Errors

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, ClassVar, Union
from enum import Enum

class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class RAGError(Exception):
    """Base class for RAG system errors."""
    
    def __init__(
        self,
        message: str,
        severity: Union[str, ErrorSeverity] = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.severity = ErrorSeverity(severity) if isinstance(severity, str) else severity
        self.context = context or {}
        self.timestamp = datetime.now()
        super().__init__(message)

class ProcessingError(RAGError):
    """Error during document processing."""
    error_code: ClassVar[str] = "PROC_ERR"

class EmbeddingError(RAGError):
    """Error during embedding generation."""
    error_code: ClassVar[str] = "EMB_ERR"

class SearchError(RAGError):
    """Error during search operations."""
    error_code: ClassVar[str] = "SEARCH_ERR"

class ValidationError(RAGError):
    """Error during data validation."""
    error_code: ClassVar[str] = "VAL_ERR"

class ConfigurationError(RAGError):
    """Error in system configuration."""
    error_code: ClassVar[str] = "CONFIG_ERR"

class PipelineError(RAGError):
    """Error in pipeline execution."""
    error_code: ClassVar[str] = "PIPE_ERR"

class RateLimitError(RAGError):
    """Error for rate limiting violations."""
    error_code: ClassVar[str] = "RATE_LIMIT_ERR"

class ResourceExhaustionError(RAGError):
    """Error for resource exhaustion."""
    error_code: ClassVar[str] = "RESOURCE_ERR"

class ErrorCategory(Enum):
    """Combined error categories."""
    CONTENT = "content"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    PROCESSING = "processing"
    MATH = "math"
    SYSTEM = "system"
    DATABASE = "database"
    VALIDATION = "validation"

@dataclass
class ErrorInfo:
    """Detailed information about an error occurrence."""
    
    error_type: str
    message: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    is_recoverable: bool = True
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    
    @classmethod
    def from_exception(cls, error: RAGError) -> 'ErrorInfo':
        """Create ErrorInfo from a RAGError instance."""
        return cls(
            error_type=error.__class__.__name__,
            message=str(error),
            timestamp=getattr(error, 'timestamp', datetime.now()),
            context=getattr(error, 'context', {}),
            severity=getattr(error, 'severity', ErrorSeverity.ERROR),
            error_code=getattr(error, 'error_code', None)
        )

@dataclass
class ErrorReport:
    """Aggregated error report with statistics and details."""
    
    total_errors: int
    error_types: Dict[str, int]
    errors: List[ErrorInfo] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    summary: Optional[str] = None
    
    @classmethod
    def from_error_list(cls, errors: List[ErrorInfo]) -> 'ErrorReport':
        """Create an ErrorReport from a list of ErrorInfo objects."""
        error_types = {}
        for error in errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
        return cls(
            total_errors=len(errors),
            error_types=error_types,
            errors=errors,
            summary=cls._generate_summary(error_types)
        )
    
    @staticmethod
    def _generate_summary(error_types: Dict[str, int]) -> str:
        """Generate a human-readable summary of error types."""
        if not error_types:
            return "No errors reported"
        
        summary_parts = [f"{count} {error_type}(s)" 
                        for error_type, count in error_types.items()]
        return "Error Summary: " + ", ".join(summary_parts)

@dataclass
class ErrorEvent:
    """Enhanced error event tracking."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    recovery_steps: List[str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    @classmethod
    def from_error_info(cls, error_info: ErrorInfo, category: ErrorCategory) -> 'ErrorEvent':
        """Create ErrorEvent from ErrorInfo."""
        return cls(
            error_id=f"{category.value}_{datetime.now().timestamp()}",
            timestamp=error_info.timestamp,
            category=category,
            severity=error_info.severity,
            message=error_info.message,
            stack_trace=error_info.stack_trace,
            context=error_info.context,
            recovery_steps=[]
        ) 