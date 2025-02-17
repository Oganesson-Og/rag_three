"""
Qdrant Vector Store Module
--------------------------------

Qdrant-based vector store implementation for efficient similarity search.

Key Features:
- High-performance vector search
- Automatic collection management
- Rich filtering capabilities
- Batch operations support
- Payload indexing
- Collection optimization
- Real-time updates

Technical Details:
- Qdrant client integration
- GRPC communication
- Collection initialization
- Payload indexing
- Vector normalization
- Efficient filtering
- Batch processing

Dependencies:
- qdrant-client>=1.7.0
- numpy>=1.24.0
- typing-extensions>=4.7.0

Example Usage:
    store = QdrantVectorStore(
        collection_name="vectors",
        vector_size=384,
        host="localhost"
    )
    vector_id = await store.add_vector([1.0, 2.0, 3.0], {"type": "test"})
    results = await store.search([1.0, 2.0, 3.0], k=5)

Performance Considerations:
- Optimized vector operations
- Efficient payload filtering
- Connection pooling
- Batch processing
- Query optimization
- Index management
- Cache strategies

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from .base import BaseVectorStore, Vector

class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation"""
    
    def __init__(
        self,
        collection_name: str = "educational_vectors",
        vector_size: int = 384,  # Default for MiniLM
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = True
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(
            host=host,
            port=port,
            prefer_grpc=prefer_grpc
        )
        self._init_collection()
        
    def _init_collection(self) -> None:
        """Initialize Qdrant collection with proper schema"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="subject",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="grade_level",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content_type",
                field_schema="keyword"
            )

    async def add_vector(
        self,
        vector: Vector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add single vector to Qdrant"""
        point_id = self._generate_point_id()
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=self._prepare_vector(vector),
                    payload=metadata or {}
                )
            ]
        )
        
        return str(point_id)

    async def bulk_add_vectors(
        self,
        vectors: List[Tuple[Vector, Dict[str, Any]]]
    ) -> List[str]:
        """Efficient bulk vector insertion"""
        point_ids = [self._generate_point_id() for _ in vectors]
        
        points = [
            models.PointStruct(
                id=point_id,
                vector=self._prepare_vector(vector),
                payload=metadata or {}
            )
            for point_id, (vector, metadata) in zip(point_ids, vectors)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return [str(pid) for pid in point_ids]

    async def search(
        self,
        query_vector: Vector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search vectors with filtering"""
        search_filters = None
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            search_filters = models.Filter(
                must=filter_conditions
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=self._prepare_vector(query_vector),
            limit=k,
            query_filter=search_filters
        )
        
        return [
            {
                'id': str(hit.id),
                'score': hit.score,
                'metadata': hit.payload
            }
            for hit in results
        ]

    def _prepare_vector(self, vector: Vector) -> List[float]:
        """Prepare vector for Qdrant storage"""
        if isinstance(vector, np.ndarray):
            return vector.tolist()
        return vector

    def _generate_point_id(self) -> int:
        """Generate unique point ID"""
        import uuid
        return uuid.uuid4().int & (1<<64)-1  # Convert UUID to integer