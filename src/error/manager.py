"""Consolidated error management system."""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
from pathlib import Path
import traceback
from .models import ErrorEvent, ErrorCategory, ErrorSeverity, ErrorInfo
from .rate_limiter import RAGRateLimiter

class ErrorManager:
    """Unified error management system."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        enable_auto_recovery: bool = True
    ):
        # Initialize components
        self.config = self._load_config(config_path)
        self.log_dir = log_dir or Path("logs/errors")
        self.logger = self._setup_logging()
        self.rate_limiter = RAGRateLimiter()
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.active_errors: Dict[str, ErrorEvent] = {}
        self.error_patterns: Dict[str, Dict] = {}
        
        # Recovery configuration
        self.enable_auto_recovery = enable_auto_recovery
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        
    async def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEvent:
        """Enhanced error handling with categorization."""
        # Create error info
        error_info = ErrorInfo.from_exception(error) if isinstance(error, RAGError) \
            else ErrorInfo(
                error_type=type(error).__name__,
                message=str(error),
                timestamp=datetime.now(),
                context=context or {},
                severity=self._determine_severity(error, category)
            )
        
        # Create error event
        event = ErrorEvent.from_error_info(error_info, category)
        
        # Record error
        self.error_history.append(event)
        self.active_errors[event.error_id] = event
        
        # Log error
        self.logger.error(
            f"{category.value.upper()} ERROR: {event.message}",
            extra={"error_id": event.error_id, "context": event.context}
        )
        
        # Attempt recovery if enabled
        if self.enable_auto_recovery:
            await self._attempt_recovery(event)
        
        return event
    
    async def check_rate_limit(
        self,
        operation_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check rate limits with error handling."""
        try:
            return await self.rate_limiter.check_rate_limit(operation_type)
        except RateLimitError as e:
            await self.handle_error(
                e,
                ErrorCategory.SYSTEM,
                context={"operation": operation_type, **(context or {})}
            )
            raise
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "total_errors": len(self.error_history),
            "active_errors": len(self.active_errors),
            "errors_by_category": self._count_errors_by_category(),
            "errors_by_severity": self._count_errors_by_severity(),
            "rate_limits": {
                op: self.rate_limiter.get_remaining_quota(op)
                for op in self.rate_limiter.limits
            }
        }
    
    async def _attempt_recovery(self, event: ErrorEvent):
        """Attempt to recover from error."""
        if handler := self.recovery_handlers.get(event.category):
            try:
                await handler(event)
                event.resolved = True
                event.resolution_time = datetime.now()
            except Exception as e:
                self.logger.error(f"Recovery failed for {event.error_id}: {str(e)}")
    
    def _count_errors_by_category(self) -> Dict[str, int]:
        """Count errors by category."""
        counts = {}
        for event in self.error_history:
            counts[event.category.value] = counts.get(event.category.value, 0) + 1
        return counts
    
    def _count_errors_by_severity(self) -> Dict[str, int]:
        """Count errors by severity."""
        counts = {}
        for event in self.error_history:
            counts[event.severity.value] = counts.get(event.severity.value, 0) + 1
        return counts 