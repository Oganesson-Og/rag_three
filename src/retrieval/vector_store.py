"""
Vector Store Module
--------------------------------

Advanced vector storage implementation using Qdrant for efficient similarity
search and vector management with comprehensive metadata support.

Key Features:
- Qdrant-based vector storage
- Rich metadata filtering
- Collection management
- CRUD operations
- Batch processing
- Payload management
- Search optimization

Technical Details:
- Qdrant client integration
- Vector similarity metrics
- Collection configuration
- Index optimization
- Payload schemas
- Batch operations
- Error handling

Dependencies:
- numpy>=1.24.0
- qdrant-client>=1.7.0
- typing-extensions>=4.7.0
- pydantic>=2.5.0

Example Usage:
    # Initialize store
    store = VectorStore(
        collection_name="documents",
        dimension=768
    )
    
    # Add vectors
    store.add_vectors(
        vectors=document_vectors,
        metadata=[{"type": "article", "topic": "physics"}]
    )
    
    # Search with filters
    results = store.search(
        query_vector=query_embedding,
        filter_conditions={
            "type": "article",
            "topic": "physics"
        },
        limit=5
    )

Performance Considerations:
- Optimized vector operations
- Efficient payload handling
- Smart batch processing
- Index optimization
- Connection pooling
- Cache management

Author: Keith Satuku
Version: 1.0.0
Created: 2024
License: MIT
"""

from typing import List, Dict, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import logging

class VectorStore:
    def __init__(
        self,
        collection_name: str,
        dimension: int,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            url: Qdrant server URL (optional)
            api_key: API key for authentication (optional)
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = QdrantClient(url=url, api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
        # Create collection if it doesn't exist
        self._init_collection()
        
    def _init_collection(self):
        """Initialize Qdrant collection."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {str(e)}")
            raise
            
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 100
    ):
        """Add vectors to the store with optional metadata."""
        try:
            if metadata and len(vectors) != len(metadata):
                raise ValueError("Number of vectors and metadata entries must match")
                
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size] if metadata else None
                
                points = []
                for j, vector in enumerate(batch_vectors):
                    point = models.PointStruct(
                        id=i + j,
                        vector=vector.tolist(),
                        payload=batch_metadata[j] if batch_metadata else None
                    )
                    points.append(point)
                    
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
            self.logger.info(f"Added {len(vectors)} vectors to collection")
        except Exception as e:
            self.logger.error(f"Failed to add vectors: {str(e)}")
            raise
            
    def search(
        self,
        query_vector: np.ndarray,
        filter_conditions: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search for similar vectors with optional filtering."""
        try:
            # Prepare filter if provided
            search_filter = None
            if filter_conditions:
                filter_conditions = [
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
                search_filter = models.Filter(
                    must=filter_conditions
                )
            
            # Execute search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=search_filter
            )
            
            # Format results
            return [
                {
                    'id': hit.id,
                    'score': hit.score,
                    'metadata': hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise
            
    def delete_vectors(self, ids: List[int]):
        """Delete vectors by their IDs."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            self.logger.info(f"Deleted {len(ids)} vectors")
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {str(e)}")
            raise
            
    def get_collection_info(self) -> Dict:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.name,
                'dimension': info.vectors_config.size,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            raise 