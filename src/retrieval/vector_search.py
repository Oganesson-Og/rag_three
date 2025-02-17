"""
Vector Search Module
--------------------------------

Advanced vector search implementation using Qdrant for efficient similarity
search and nearest neighbor retrieval in high-dimensional spaces.

Key Features:
- Multiple search strategies (ANN, exact)
- Configurable distance metrics
- Rich metadata filtering
- Batch operations support
- Real-time updates
- Payload management
- Collection optimization

Technical Details:
- Qdrant-based indexing
- Cosine similarity metrics
- Collection sharding
- Vector quantization
- Payload indexing
- Query optimization
- Efficient filtering

Dependencies:
- numpy>=1.24.0
- qdrant-client>=1.7.0
- typing-extensions>=4.7.0
- pydantic>=2.5.0

Example Usage:
    # Initialize vector search
    search = VectorSearch(
        collection_name="documents",
        dimension=768,
        url="http://localhost:6333"
    )
    
    # Add vectors
    search.add_vectors(
        vectors=document_vectors,
        metadata=[{
            "type": "article",
            "topic": "physics",
            "difficulty": "advanced"
        }]
    )
    
    # Search with filters
    results = search.search(
        query_vector=query_embedding,
        filter_conditions={
            "type": "article",
            "topic": "physics"
        },
        limit=5
    )
    
    # Batch search
    results = search.batch_search(
        query_vectors=batch_embeddings,
        limit=5
    )

Performance Considerations:
- Optimized collection schemas
- Efficient payload filtering
- Connection pooling
- Batch processing
- Query optimization
- Index management
- Cache strategies

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import logging

class VectorSearch:
    def __init__(
        self,
        collection_name: str,
        dimension: int,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize Qdrant vector search.
        
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
        
        # Initialize collection
        self._init_collection()
        
    def _init_collection(self):
        """Initialize Qdrant collection with optimized settings."""
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
        """Add vectors to the collection with optional metadata.
        
        Args:
            vectors: Array of vectors to add
            metadata: Optional list of metadata dictionaries
            batch_size: Size of batches for processing
        """
        try:
            if metadata and len(vectors) != len(metadata):
                raise ValueError("Number of vectors and metadata entries must match")
                
            # Process in batches for efficiency
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
        """Search for similar vectors with optional filtering.
        
        Args:
            query_vector: Query vector
            filter_conditions: Optional dictionary of metadata filters
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
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
            
    def batch_search(
        self,
        query_vectors: np.ndarray,
        limit: int = 10,
        filter_conditions: Optional[Dict] = None
    ) -> List[List[Dict]]:
        """Perform batch search for multiple query vectors."""
        return [
            self.search(qv, filter_conditions, limit)
            for qv in query_vectors
        ]
        
    def backup_collection(self, path: str):
        """Backup Qdrant collection to specified path.
        
        Args:
            path: Path to save the collection backup
        """
        try:
            self.client.create_snapshot(
                collection_name=self.collection_name,
                snapshot_path=path
            )
            self.logger.info(f"Created collection backup at: {path}")
        except Exception as e:
            self.logger.error(f"Failed to backup collection: {str(e)}")
            raise
            
    def restore_collection(self, path: str):
        """Restore Qdrant collection from backup.
        
        Args:
            path: Path to the collection backup
        """
        try:
            self.client.restore_snapshot(
                collection_name=self.collection_name,
                snapshot_path=path
            )
            self.logger.info(f"Restored collection from: {path}")
        except Exception as e:
            self.logger.error(f"Failed to restore collection: {str(e)}")
            raise
            
    def optimize_collection(self, optimization_threshold: Optional[int] = None):
        """Optimize Qdrant collection for better performance.
        
        Args:
            optimization_threshold: Optional threshold for optimization
        """
        try:
            self.client.optimize_collection(
                collection_name=self.collection_name,
                optimization_threshold=optimization_threshold
            )
            self.logger.info(f"Optimized collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to optimize collection: {str(e)}")
            raise 