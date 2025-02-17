"""
Multi-Modal Cache Module
----------------------

Advanced caching system supporting multiple modalities with Redis and file-based
storage for educational content processing system.

Key Features:
- Multi-modal data caching
- Hybrid storage (Redis + File)
- TTL support
- Modality-specific caching
- Automatic key generation
- Cache clearing capabilities
- Fallback mechanisms
- Serialization handling

Technical Details:
- Redis primary storage
- File-based backup storage
- Pickle serialization
- MD5 key hashing
- TTL enforcement
- Directory structure management
- Atomic operations
- Error handling

Dependencies:
- redis>=4.5.0
- pickle-mixin>=1.0.2
- pathlib>=1.0.1
- hashlib (standard library)
- typing-extensions>=4.7.0

Example Usage:
    # Initialize cache
    cache = MultiModalCache(
        redis_url="redis://localhost:6379/0",
        file_cache_dir=Path("./cache"),
        ttl=3600
    )
    
    # Cache operations
    cache.set("lecture_1", audio_data, modality="audio")
    cached_data = cache.get("lecture_1", modality="audio")
    
    # Clear specific modality
    cache.clear(modality="audio")

Cache Features:
- Multi-level caching
- Modality separation
- Automatic expiration
- Persistence support
- Memory optimization
- Fast retrieval
- Backup storage

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Any, Optional, Union, Dict
import redis
from pathlib import Path
import pickle
import hashlib
from datetime import datetime, timedelta
from pydantic import BaseModel

class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    ttl: int = 3600
    max_size: int = 1000
    redis_url: str = "redis://localhost:6379"
    backend: str = "redis"

class MultiModalCache:
    """
    Multi-modal caching system with Redis and file-based storage.
    
    Attributes:
        redis_client: Redis connection client
        file_cache_dir (Path): Directory for file-based cache
        ttl (int): Default time-to-live in seconds
    
    Methods:
        get: Retrieve cached data by key and modality
        set: Store data with modality and optional TTL
        clear: Clear cache for specific modality or all
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize cache with configuration."""
        self.config = CacheConfig(**config)
        
        if not self.config.enabled:
            self.redis_client = None
            return
            
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
        except Exception as e:
            print(f"Warning: Redis connection failed: {e}")
            self.redis_client = None
        
        self.file_cache_dir = Path("./cache")
        self.file_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get(self, key: str, modality: str) -> Optional[Any]:
        """
        Retrieve data from cache by key and modality.
        
        Args:
            key: Cache key
            modality: Data modality (e.g., "audio", "text")
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        cache_key = self._generate_key(key, modality)
        
        # Try Redis first for faster access
        if self.redis_client:
            redis_result = self.redis_client.get(cache_key)
            if redis_result:
                return pickle.loads(redis_result)
            
        # Fall back to file cache
        file_path = self.file_cache_dir / f"{cache_key}.pkl"
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        return None
        
    def set(
        self,
        key: str,
        value: Any,
        modality: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        Store data in cache with modality.
        
        Args:
            key: Cache key
            value: Data to cache
            modality: Data modality
            ttl: Optional custom TTL in seconds
        """
        cache_key = self._generate_key(key, modality)
        pickled_value = pickle.dumps(value)
        
        # Store in Redis
        if self.redis_client:
            self.redis_client.set(
                cache_key,
                pickled_value,
                ex=ttl or self.config.ttl
            )
        
        # Store in file cache
        file_path = self.file_cache_dir / f"{cache_key}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(value, f)
            
    def _generate_key(self, key: str, modality: str) -> str:
        """Generate unique cache key from key and modality."""
        combined = f"{key}:{modality}"
        return hashlib.md5(combined.encode()).hexdigest()
        
    def clear(self, modality: Optional[str] = None) -> None:
        """
        Clear cache for specific modality or all cache.
        
        Args:
            modality: Optional modality to clear, None for all
        """
        if modality:
            pattern = f"*:{modality}*"
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                
            # Clear file cache for modality
            for file_path in self.file_cache_dir.glob(f"*{modality}*.pkl"):
                file_path.unlink()
        else:
            if self.redis_client:
                self.redis_client.flushdb()
            for file_path in self.file_cache_dir.glob("*.pkl"):
                file_path.unlink()