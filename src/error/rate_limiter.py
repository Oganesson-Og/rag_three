"""Rate limiting for RAG operations.
Backend App (Primary Rate Limiting)
Handles user-level rate limiting
API endpoint protection
Authentication checks
RAG Pipeline (Resource Protection)
Protects expensive operations (embeddings, LLM calls)
Prevents resource exhaustion
Monitors system health
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import asyncio
from .models import RateLimitError, ResourceExhaustionError

@dataclass
class RateLimit:
    max_requests: int
    time_window: int  # in seconds
    current_requests: int = 0
    window_start: Optional[datetime] = None

class RAGRateLimiter:
    """Rate limiter for RAG pipeline operations."""
    
    def __init__(self):
        # Rate limits for different operations
        self.limits: Dict[str, RateLimit] = {
            'embedding': RateLimit(max_requests=100, time_window=60),  # 100 per minute
            'llm_completion': RateLimit(max_requests=50, time_window=60),  # 50 per minute
            'vector_search': RateLimit(max_requests=200, time_window=60),  # 200 per minute
        }
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, operation_type: str) -> bool:
        """Check if operation is within rate limits."""
        if operation_type not in self.limits:
            return True
            
        limit = self.limits[operation_type]
        
        async with self.lock:
            now = datetime.now()
            
            # Initialize or reset window
            if not limit.window_start or \
               (now - limit.window_start).total_seconds() > limit.time_window:
                limit.window_start = now
                limit.current_requests = 0
            
            # Check limit
            if limit.current_requests >= limit.max_requests:
                raise RateLimitError(
                    f"Rate limit exceeded for {operation_type}. "
                    f"Max {limit.max_requests} requests per {limit.time_window} seconds"
                )
            
            # Increment counter
            limit.current_requests += 1
            return True
    
    def get_remaining_quota(self, operation_type: str) -> Dict[str, int]:
        """Get remaining quota for operation type."""
        if operation_type not in self.limits:
            return {"remaining": -1, "reset_in": 0}
            
        limit = self.limits[operation_type]
        now = datetime.now()
        
        if not limit.window_start:
            return {
                "remaining": limit.max_requests,
                "reset_in": limit.time_window
            }
        
        time_passed = (now - limit.window_start).total_seconds()
        if time_passed > limit.time_window:
            return {
                "remaining": limit.max_requests,
                "reset_in": 0
            }
            
        return {
            "remaining": max(0, limit.max_requests - limit.current_requests),
            "reset_in": int(limit.time_window - time_passed)
        } 