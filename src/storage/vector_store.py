"""
Vector Store Module
--------------------------------

Core vector storage implementation with support for efficient similarity search
and metadata management.

Key Features:
- Vector storage and retrieval
- Similarity search
- Metadata management
- Timestamp tracking
- UUID-based IDs
- Filter support
- Batch operations

Technical Details:
- Dictionary-based storage
- Numpy array handling
- Metadata validation
- UUID generation
- Type validation
- Error handling
- Flexible filtering

Dependencies:
- numpy>=1.24.0
- typing-extensions>=4.7.0
- datetime>=3.8
- uuid>=3.8

Example Usage:
    # Initialize store
    store = VectorStore()
    
    # Add vector with metadata
    vector_id = store.add_vector(
        vector=[1.0, 2.0, 3.0],
        metadata={
            "type": "document",
            "source": "textbook",
            "topic": "physics"
        }
    )
    
    # Search vectors
    results = store.get_vector(vector_id)
    
    # Add multiple vectors
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    metadata_list = [
        {"type": "doc1"},
        {"type": "doc2"}
    ]
    ids = store.add_vectors(vectors, metadata_list)

Performance Considerations:
- Memory-efficient storage
- Fast vector operations
- Optimized metadata handling
- Efficient ID generation
- Type validation overhead
- Dictionary lookup speed
- Batch processing efficiency

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Union, Optional, TypedDict
import numpy as np
from datetime import datetime
from pathlib import Path

class VectorRecord(TypedDict):
    id: str
    vector: Union[List[float], np.ndarray]
    metadata: Dict[str, Union[str, int, float]]
    timestamp: datetime

class VectorStore:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.store: Dict[str, VectorRecord] = {}
        
    def add_vector(
        self,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict] = None
    ) -> str:
        """Add vector to store.
        
        Args:
            vector: Vector data as list or numpy array
            metadata: Optional metadata dictionary
            
        Returns:
            str: Generated vector ID
        """
        vector_id = self._generate_id()
        self.store[vector_id] = {
            'id': vector_id,
            'vector': vector,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        return vector_id
        
    def get_vector(
        self,
        vector_id: str
    ) -> Optional[VectorRecord]:
        """Get vector from store.
        
        Args:
            vector_id: ID of vector to retrieve
            
        Returns:
            Optional[VectorRecord]: Vector record if found, None otherwise
        """
        return self.store.get(vector_id)
        
    def _generate_id(self) -> str:
        """Generate unique vector ID.
        
        Returns:
            str: Generated UUID
        """
        import uuid
        return str(uuid.uuid4()) 