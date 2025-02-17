"""
Vector Store Package
--------------------------------

Vector storage implementations for efficient similarity search.

Available Stores:
- InMemoryVectorStore: Simple in-memory implementation
- PGVectorStore: PostgreSQL with pgvector extension
- QdrantVectorStore: Qdrant vector database implementation

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from .base import BaseVectorStore
from .in_memory_store import InMemoryVectorStore
from .pgvector_store import PGVectorStore
from .qdrant_store import QdrantVectorStore

__all__ = [
    'BaseVectorStore',
    'InMemoryVectorStore',
    'PGVectorStore',
    'QdrantVectorStore'
]
