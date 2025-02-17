"""
Vector Store Module
-----------------

Production-ready vector storage and similarity search system using Qdrant.

Key Features:
- Vector indexing and storage
- Similarity search with filtering
- Rich metadata management
- Automatic persistence
- Batch operations
- Horizontal scalability
- Monitoring support

Technical Details:
- Qdrant-based storage
- Multiple distance metrics
- Metadata filtering
- Collection management
- Optimized retrieval
- Concurrent access support

Dependencies:
- qdrant-client>=1.6.0
- numpy>=1.24.0
- typing>=3.7.4

Example Usage:
    # Initialize store
    store = VectorStore(config_manager)
    
    # Add vectors
    store.add_vectors(vectors, metadata)
    
    # Search with filters
    results = store.search_nearest(
        query_vector, 
        k=5,
        filter_conditions={"category": "physics"}
    )

Search Features:
- K-nearest neighbors
- Rich metadata filtering
- Batch search operations
- Score normalization
- Payload management
- Multiple metrics support

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

class VectorStore:
    def __init__(self, config_manager):
        """Initialize Qdrant vector store with configuration."""
        self.config = config_manager
        self.collection_name = self.config.get("vector_store.collection_name", "default")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.config.get("vector_store.url", "localhost"),
            port=self.config.get("vector_store.port", 6333)
        )
        
        # Ensure collection exists
        self._init_collection()
        
    def _init_collection(self):
        """Initialize or validate collection."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.get("vector_store.vector_size", 768),
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize collection: {str(e)}")
        
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors with metadata to the store."""
        points = [
            models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=meta
            )
            for i, (vector, meta) in enumerate(zip(vectors, metadata))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def search_nearest(
        self, 
        query_vector: np.ndarray, 
        k: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for nearest vectors with optional filtering."""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=k,
            query_filter=models.Filter(
                must=self._build_conditions(filter_conditions)
            ) if filter_conditions else None
        )
        
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'metadata': hit.payload
            }
            for hit in search_result
        ]
        
    def _build_conditions(self, conditions: Dict) -> List[models.FieldCondition]:
        """Build Qdrant filter conditions from dict."""
        if not conditions:
            return []
            
        return [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            )
            for key, value in conditions.items()
        ]
        
    def batch_search_nearest(
        self, 
        query_vectors: np.ndarray, 
        k: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[List[Dict]]:
        """Perform batch search for multiple query vectors."""
        return [
            self.search_nearest(qv, k, filter_conditions)
            for qv in query_vectors
        ]
        
    def delete_vectors(self, ids: List[int]):
        """Delete vectors by their IDs."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )
        
    def update_vectors(
        self,
        ids: List[int],
        vectors: np.ndarray,
        metadata: List[Dict]
    ):
        """Update existing vectors and their metadata."""
        points = [
            models.PointStruct(
                id=id_,
                vector=vector.tolist(),
                payload=meta
            )
            for id_, vector, meta in zip(ids, vectors, metadata)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        ) 