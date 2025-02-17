"""
Cache Models Module
-----------------

Core cache models for managing cached data with metadata tracking and expiration handling.

Key Features:
- Metadata tracking
- Expiration management
- Size monitoring
- Access tracking
- Type validation
- Automatic timestamps
- Memory optimization

Technical Details:
- Pydantic models
- Type validation
- Timestamp management
- Memory tracking
- Expiration checks
- Access monitoring
- Data persistence

Dependencies:
- pydantic>=2.0.0
- typing (standard library)
- datetime (standard library)

Example Usage:
    # Create cache entry
    entry = CacheEntry(
        key="embedding_123",
        value=np.array([0.1, 0.2, 0.3]),
        metadata=CacheMetadata(
            size_bytes=1024,
            expires_at=datetime.now() + timedelta(hours=24)
        )
    )
    
    # Check expiration
    if not entry.is_expired():
        cached_value = entry.value

Cache Types:
- Vector Embeddings
- Document Metadata
- Search Results
- Computation Results
- Intermediate Data

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class CacheMetadata(BaseModel):
    """
    Cache metadata model.
    
    Attributes:
        created_at (datetime): Timestamp when cache entry was created
        expires_at (Optional[datetime]): Optional expiration timestamp
        size_bytes (int): Size of cached data in bytes
        last_accessed (Optional[datetime]): Last access timestamp
    """
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    size_bytes: int = Field(gt=0)
    last_accessed: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

class CacheEntry(BaseModel):
    """
    Cache entry model.
    
    Attributes:
        key (str): Unique identifier for cached item
        value (Any): Cached data
        metadata (CacheMetadata): Associated metadata
    """
    key: str = Field(..., min_length=1)
    value: Any
    metadata: CacheMetadata = Field(default_factory=CacheMetadata)

    class Config:
        arbitrary_types_allowed = True
    
    def is_expired(self) -> bool:
        """
        Check if cache entry has expired.
        
        Returns:
            bool: True if expired, False otherwise
        """
        if not self.metadata.expires_at:
            return False
        return datetime.now() > self.metadata.expires_at 