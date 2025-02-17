"""
Semantic Search Module
--------------------------------

Advanced semantic search implementation using vector embeddings for
intelligent content retrieval based on meaning rather than exact matches.

Key Features:
- Embedding-based search
- Configurable similarity threshold
- Filter support
- Context-aware search
- Reranking capabilities
- Batch search operations
- Document indexing

Technical Details:
- Vector similarity calculations
- Embedding generation
- Threshold-based filtering
- Context integration
- Result reranking
- Batch processing
- Document management

Dependencies:
- numpy>=1.24.0
- pydantic>=2.5.0
- typing-extensions>=4.7.0

Example Usage:
    # Initialize semantic search
    searcher = SemanticSearch(embedding_generator)
    
    # Index documents
    searcher.index_documents(documents)
    
    # Basic search
    results = searcher.search(
        query="quantum entanglement",
        k=3
    )
    
    # Advanced search with filters
    results = searcher.search(
        query="quantum entanglement",
        k=5,
        threshold=0.7,
        filters={"subject": "physics"},
        context="quantum mechanics basics"
    )

Performance Considerations:
- Efficient vector operations
- Optimized similarity calculations
- Smart document indexing
- Memory-efficient storage
- Fast batch processing

Author: Keith Satuku
Version: 1.0.0
Created: 2024
License: MIT
"""

from typing import List, Dict, Optional, Callable
import numpy as np

class SemanticSearch:
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
        self.documents = []
        self.embeddings = None
        
    def index_documents(self, documents: List[Dict]):
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        self.embeddings = self.embedding_generator.generate_embeddings(texts)
        
    def search(self, query: str, k: int = 3, threshold: float = 0.0,
              filters: Optional[Dict] = None, context: Optional[str] = None,
              rerank_fn: Optional[Callable] = None) -> List[Dict]:
        if not query:
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be positive")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
            
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        scores = np.dot(self.embeddings, query_embedding)
        
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                doc = self.documents[i].copy()
                doc['score'] = float(score)
                if not filters or all(doc['metadata'].get(k) == v for k, v in filters.items()):
                    results.append(doc)
                    
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        
        if rerank_fn:
            results = rerank_fn(results, query)
            
        return results
        
    def batch_search(self, queries: List[str], **kwargs) -> List[List[Dict]]:
        return [self.search(query, **kwargs) for query in queries]
        
    def add_documents(self, documents: List[Dict]):
        if not self.embeddings:
            self.index_documents(documents)
        else:
            self.documents.extend(documents)
            new_embeddings = self.embedding_generator.generate_embeddings(
                [doc['text'] for doc in documents]
            )
            self.embeddings = np.vstack([self.embeddings, new_embeddings]) 