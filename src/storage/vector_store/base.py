"""
Base Vector Store Module
--------------------------------

Abstract base class defining the interface for vector storage implementations.

Key Features:
- Vector storage and retrieval
- Similarity search
- Metadata management
- Vector deletion
- Metadata updates

Technical Details:
- Abstract base class implementation
- Type hints for vectors and metadata
- Numpy array support
- Flexible filtering

Dependencies:
- numpy>=1.24.0
- typing-extensions>=4.7.0

Example Usage:
    class MyVectorStore(BaseVectorStore):
        def add_vector(self, vector, metadata=None):
            # Implementation
            pass
            
        def search(self, query_vector, k=10, filters=None):
            # Implementation
            pass

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
""" 

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
from numpy.typing import NDArray

# Type alias for vector
Vector = Union[List[float], NDArray[np.float32]]

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_vector(self, vector: Vector, metadata: Dict[str, Any] = None) -> str:
        """Add a vector to the store and return its ID."""
        pass

    @abstractmethod
    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        """Retrieve a stored vector."""
        pass

    @abstractmethod
    def search(
        self, 
        query_vector: Vector, 
        k: int = 10, 
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity (or another metric)."""
        pass

    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> int:
        """Delete vector entries by IDs."""
        pass

    @abstractmethod
    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a stored vector."""
        pass

