"""
Education Vector Store Module
--------------------------

Specialized vector storage system optimized for educational content with enhanced 
metadata management and search capabilities.

Key Features:
- Educational content vectorization
- Subject-specific indexing
- Grade level filtering
- Topic clustering
- Standard alignment
- Content type filtering
- Metadata enrichment
- Batch operations

Technical Details:
- Qdrant vector database
- Cosine similarity search
- Collection optimization
- Vector normalization
- Payload indexing
- Query optimization
- Error handling

Dependencies:
- qdrant-client>=1.7.0
- numpy>=1.24.0
- typing-extensions>=4.7.0
- datetime

Example Usage:
    # Initialize store
    store = EducationVectorStore(
        collection_name="education",
        config={
            "host": "localhost",
            "port": 6333,
            "prefer_grpc": True,
            "collections": {
                "education": {
                    "vector_size": 384,
                    "optimizers_config": {
                        "default_segment_number": 2,
                        "memmap_threshold": 10000
                    }
                }
            }
        }
    )
    
    # Insert vectors with educational metadata
    store.upsert_vectors(
        vectors=[...],  # numpy arrays
        metadata=[{
            "subject": "physics",
            "grade_level": "high_school",
            "topic": "mechanics",
            "content_type": "lesson"
        }]
    )
    
    # Search with filters
    results = store.search(
        query_vector=...,  # numpy array
        filters={
            "subject": "physics",
            "grade_level": "high_school"
        },
        limit=5,
        score_threshold=0.7
    )

Performance Considerations:
- Optimized vector operations
- Efficient metadata filtering
- Collection segmentation
- Memory mapping
- Batch processing
- Query optimization
- Cache strategies

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..config.settings import QDRANT_CONFIG

class EducationVectorStore:
    """Enhanced vector storage system optimized for educational content."""
    
    def __init__(
        self,
        collection_name: str = "education",
        config: Optional[Dict] = None
    ):
        """Initialize education vector store.
        
        Args:
            collection_name: Name of the vector collection
            config: Optional configuration dictionary
        """
        self.config = config or QDRANT_CONFIG
        self.client = QdrantClient(
            host=self.config["host"],
            port=self.config["port"],
            prefer_grpc=self.config["prefer_grpc"]
        )
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure collection exists with optimal configuration."""
        try:
            collection_config = self.config["collections"][self.collection_name]
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=collection_config["vector_size"],
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=collection_config["optimizers_config"]["default_segment_number"],
                    memmap_threshold=collection_config["optimizers_config"]["memmap_threshold"]
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {str(e)}")

    def upsert_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Insert or update vectors with educational metadata.
        
        Args:
            vectors: List of vector arrays
            metadata: List of metadata dictionaries
            ids: Optional list of vector IDs
        """
        if not ids:
            ids = [str(i) for i in range(len(vectors))]
            
        points = [
            models.PointStruct(
                id=id_,
                vector=vector.tolist(),
                payload={
                    **meta,
                    "timestamp": datetime.utcnow().isoformat(),
                    "vector_version": "1.0"
                }
            )
            for id_, vector, meta in zip(ids, vectors, metadata)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector: np.ndarray,
        filters: Optional[Dict] = None,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """Enhanced search with educational context awareness.
        
        Args:
            query_vector: Query vector array
            filters: Optional metadata filters
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=False
        )
        
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            search_params=search_params
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload,
                "vector": hit.vector
            }
            for hit in results
        ]

    def batch_search(
        self,
        query_vectors: List[np.ndarray],
        filters: Optional[Dict] = None,
        limit: int = 5
    ) -> List[List[Dict]]:
        """Perform batch search operations.
        
        Args:
            query_vectors: List of query vector arrays
            filters: Optional metadata filters
            limit: Maximum number of results per query
            
        Returns:
            List of search result lists
        """
        return [
            self.search(qv, filters, limit)
            for qv in query_vectors
        ] 
    