"""
Error Management Module
----------------------

Provides unified error handling, rate limiting, and recovery mechanisms for the RAG system.

Key Components:
- Error models and categories
- Error management and tracking
- Rate limiting
- Auto-recovery mechanisms
"""

from .models import (
    ErrorCategory,
    ErrorSeverity,
    ErrorEvent,
    ErrorInfo,
    RAGError,
    RateLimitError,
    ResourceExhaustionError
)
from ..llm.model_manager import ModelError, ModelErrorType


from .manager import ErrorManager
from .rate_limiter import RAGRateLimiter, RateLimit

__all__ = [
    # Error Models
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorEvent',
    'ErrorInfo',
    'RAGError',
    'ModelError',
    'ModelErrorType',
    'RateLimitError',
    'ResourceExhaustionError',
    
    # Management
    'ErrorManager',
    
    # Rate Limiting
    'RAGRateLimiter',
    'RateLimit'
]

# Version info
__version__ = '1.0.0'
