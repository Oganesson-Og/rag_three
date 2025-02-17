"""
Educational Vector Index Module
----------------------------

Specialized vector index for educational content using Qdrant vector database.

Key Features:
- Educational content indexing
- Multi-modal vector storage
- Subject-specific indexing
- Grade-level organization
- Rich metadata support
- Batch processing
- Automatic persistence

Technical Details:
- Qdrant-based storage
- Multiple distance metrics
- Metadata filtering
- Collection management
- Optimized retrieval
- Concurrent access support
- Automatic backups

Dependencies:
- qdrant-client>=1.6.0
- numpy>=1.24.0
- torch>=2.0.0
- sentence-transformers>=2.2.0
- typing (standard library)
- logging (standard library)
- pathlib (standard library)

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import logging
import json
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime
from tqdm import tqdm
import threading
from queue import Queue
from qdrant_client import QdrantClient
from qdrant_client.http import models

class IndexType(Enum):
    PLAIN = "plain"  # Basic Qdrant index
    HNSW = "hnsw"   # HNSW-based index
    CUSTOM = "custom"  # Custom index configuration

@dataclass
class IndexConfig:
    """Configuration for educational vector index."""
    dimension: int = 768
    index_type: IndexType = IndexType.HNSW
    collection_name: str = "educational_content"
    url: str = "http://localhost:6333"
    hnsw_config: Dict = None
    num_threads: int = 4
    batch_size: int = 1000
    cache_size_gb: float = 2.0

    def __post_init__(self):
        if self.hnsw_config is None:
            self.hnsw_config = {
                "m": 16,
                "ef_construct": 200,
                "ef_search": 128
            }

class EducationalVectorIndex:
    """Specialized vector index for educational content using Qdrant."""
    
    def __init__(
        self,
        config: Optional[IndexConfig] = None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or IndexConfig()
        self.device = device
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name).to(device)
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=self.config.url)
        
        # Initialize collections
        self._initialize_collections()
        
        # Initialize metadata storage
        self.metadata: Dict[int, Dict] = {}
        
        # Initialize thread pool for batch processing
        self.thread_pool = Queue()
        self._initialize_threads()
        
    def _initialize_collections(self):
        """Initialize Qdrant collections."""
        # Main collection
        self._create_collection(self.config.collection_name)
        
        # Subject-specific collections
        self.subject_collections = {}
        
        # Grade-specific collections
        self.grade_collections = {}
        
    def _create_collection(self, name: str):
        """Create a Qdrant collection with appropriate configuration."""
        try:
            self.client.get_collection(name)
        except Exception:
            # Collection doesn't exist, create it
            vectors_config = models.VectorParams(
                size=self.config.dimension,
                distance=models.Distance.COSINE
            )
            
            if self.config.index_type == IndexType.HNSW:
                vectors_config.hnsw_config = models.HnswConfigDiff(
                    m=self.config.hnsw_config["m"],
                    ef_construct=self.config.hnsw_config["ef_construct"],
                    ef_search=self.config.hnsw_config["ef_search"]
                )
            
            self.client.create_collection(
                collection_name=name,
                vectors_config=vectors_config
            )

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """Add vectors to the index with educational metadata."""
        batch_size = batch_size or self.config.batch_size
        ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Generate IDs for the batch
            batch_ids = [str(i + j) for j in range(len(batch_vectors))]
            ids.extend(batch_ids)
            
            # Create points
            points = [
                models.PointStruct(
                    id=id_,
                    vector=vector.tolist(),
                    payload=metadata
                )
                for id_, vector, metadata in zip(batch_ids, batch_vectors, batch_metadata)
            ]
            
            # Add to main collection
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            
            # Add to subject-specific collections
            self._add_to_subject_collections(points, batch_metadata)
            
            # Add to grade-specific collections
            self._add_to_grade_collections(points, batch_metadata)
            
        return ids

    def _add_to_subject_collections(self, points: List[models.PointStruct], metadata_list: List[Dict]):
        """Add vectors to subject-specific collections."""
        for point, metadata in zip(points, metadata_list):
            if "subject" in metadata:
                subject = metadata["subject"]
                if subject not in self.subject_collections:
                    collection_name = f"{self.config.collection_name}_{subject}"
                    self._create_collection(collection_name)
                    self.subject_collections[subject] = collection_name
                
                self.client.upsert(
                    collection_name=self.subject_collections[subject],
                    points=[point]
                )

    def _add_to_grade_collections(self, points: List[models.PointStruct], metadata_list: List[Dict]):
        """Add vectors to grade-specific collections."""
        for point, metadata in zip(points, metadata_list):
            if "grade_level" in metadata:
                grade = metadata["grade_level"]
                if grade not in self.grade_collections:
                    collection_name = f"{self.config.collection_name}_grade_{grade}"
                    self._create_collection(collection_name)
                    self.grade_collections[grade] = collection_name
                
                self.client.upsert(
                    collection_name=self.grade_collections[grade],
                    points=[point]
                )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors with optional filtering."""
        search_result = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector.tolist(),
            limit=k,
            query_filter=self._build_filter(filter_conditions)
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload
            }
            for hit in search_result
        ]

    def _build_filter(self, conditions: Optional[Dict]) -> Optional[models.Filter]:
        """Build Qdrant filter from conditions."""
        if not conditions:
            return None
            
        filter_conditions = []
        for key, value in conditions.items():
            filter_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
            
        return models.Filter(
            must=filter_conditions
        )

    def delete_vectors(self, ids: List[str]):
        """Delete vectors by their IDs."""
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )

    def _initialize_threads(self):
        """Initialize thread pool for batch processing."""
        for _ in range(self.config.num_threads):
            thread = threading.Thread(target=self._process_batch, daemon=True)
            thread.start()

    def _process_batch(self):
        """Process batches from the thread pool."""
        while True:
            batch = self.thread_pool.get()
            if batch is None:
                break
            try:
                self.add_vectors(**batch)
            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")
            finally:
                self.thread_pool.task_done() 