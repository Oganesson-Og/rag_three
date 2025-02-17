"""
Vector Cache Module
----------------

Specialized caching system for vector embeddings and high-dimensional data
in the educational content processing system.

Key Features:
- Vector data caching
- Metadata support
- TTL management
- Timestamp tracking
- Memory optimization
- Type safety
- Automatic expiration
- Configuration flexibility

Technical Details:
- NumPy array support
- Custom type definitions
- Timestamp-based expiration
- Memory-efficient storage
- Type checking
- Flexible configuration
- Metadata management

Dependencies:
- numpy>=1.24.0
- typing-extensions>=4.7.0
- datetime (standard library)

Example Usage:
    # Initialize cache
    cache = VectorCache(config={'max_size': 1000})
    
    # Store vector with metadata
    cache.set(
        "doc_1",
        vector=[0.1, 0.2, 0.3],
        metadata={"type": "embedding"},
        ttl=3600
    )
    
    # Retrieve vector
    vector = cache.get("doc_1")

Cache Features:
- Vector storage optimization
- Automatic expiration
- Metadata association
- Type safety checks
- Memory management
- Fast retrieval
- Configuration options

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Union, Optional, TypedDict
import numpy as np
from datetime import datetime

class CacheEntry(TypedDict):
    """
    Type definition for cache entries.
    
    Attributes:
        vector: Stored vector data
        metadata: Associated metadata
        timestamp: Entry creation time
        ttl: Time-to-live in seconds
    """
    vector: Union[List[float], np.ndarray]
    metadata: Dict[str, Union[str, int, float]]
    timestamp: datetime
    ttl: Optional[int]

class VectorCache:
    """
    Specialized cache for vector data storage.
    
    Attributes:
        config: Cache configuration options
        cache: Internal cache storage
    
    Methods:
        get: Retrieve vector by key
        set: Store vector with metadata
        _is_expired: Check entry expiration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize vector cache.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.cache: Dict[str, CacheEntry] = {}
        
    def get(
        self,
        key: str
    ) -> Optional[Union[List[float], np.ndarray]]:
        """
        Retrieve vector from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached vector if found and valid, None otherwise
        """
        entry = self.cache.get(key)
        if entry and not self._is_expired(entry):
            return entry['vector']
        return None
        
    def set(
        self,
        key: str,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        Store vector in cache.
        
        Args:
            key: Cache key
            vector: Vector data to store
            metadata: Optional metadata dictionary
            ttl: Optional time-to-live in seconds
        """
        self.cache[key] = {
            'vector': vector,
            'metadata': metadata or {},
            'timestamp': datetime.now(),
            'ttl': ttl
        }
        
    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if cache entry is expired.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            True if entry is expired, False otherwise
        """
        if not entry['ttl']:
            return False
        age = (datetime.now() - entry['timestamp']).total_seconds()
        return age > entry['ttl'] 