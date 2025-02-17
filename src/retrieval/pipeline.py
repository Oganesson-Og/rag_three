"""
Retrieval Pipeline Module
-----------------------

Implements a hybrid retrieval system combining vector similarity and keyword search,
with additional re-ranking and filtering capabilities.

Features:
- Hybrid retrieval (vector + keyword)
- Re-ranking with cross-encoder
- Result filtering and deduplication
- Query preprocessing
- Configurable retrieval parameters

Components:
1. VectorRetriever: Dense vector similarity search
2. KeywordRetriever: Sparse keyword-based search
3. Reranker: Cross-encoder based reranking
4. ResultFilter: Post-processing and filtering

Technical Details:
- Async processing support
- Batch operation capabilities
- Configurable fusion strategies
- Cache management

Dependencies:
- numpy>=1.19.0
- sentence-transformers>=2.2.0
- rank_bm25>=0.2.2

Author: Keith Satuku
Created: 2024
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from ..embeddings import EmbeddingGenerator
from ..utils.text_cleaner import TextCleaner
from ..config.embedding_config import EMBEDDING_CONFIG

@dataclass
class RetrievalResult:
    """Data class for retrieval results."""
    document_id: str
    content: str
    score: float
    metadata: Dict
    source: str = "hybrid"  # vector, keyword, or hybrid

class RetrievalPipeline:
    def __init__(
        self,
        embedding_model: str = "minilm",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        hybrid_weight: float = 0.7,  # Weight for vector scores vs keyword scores
        use_reranking: bool = True
    ):
        """Initialize the retrieval pipeline.
        
        Args:
            embedding_model: Model for vector embeddings
            reranker_model: Model for reranking
            top_k: Number of results to retrieve
            hybrid_weight: Weight for vector vs keyword scores (0-1)
            use_reranking: Whether to apply reranking
        """
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.text_cleaner = TextCleaner()
        self.top_k = top_k
        self.hybrid_weight = hybrid_weight
        self.use_reranking = use_reranking
        
        if use_reranking:
            self.reranker = CrossEncoder(reranker_model)
        
        # Initialize BM25 for keyword search
        self.bm25 = None
        self.document_store = {}
        
    def index_documents(self, documents: List[Dict[str, Union[str, Dict]]]):
        """Index documents for retrieval.
        
        Args:
            documents: List of documents with content and metadata
        """
        # Process and store documents
        processed_docs = []
        for doc in documents:
            cleaned_text = self.text_cleaner.clean(doc['content'])
            self.document_store[doc['id']] = {
                'content': cleaned_text,
                'metadata': doc.get('metadata', {}),
                'embedding': self.embedding_generator.generate(cleaned_text)
            }
            processed_docs.append(cleaned_text)
            
        # Initialize BM25 with processed documents
        tokenized_docs = [doc.split() for doc in processed_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return (optional)
            
        Returns:
            List of RetrievalResult objects
        """
        if top_k is None:
            top_k = self.top_k
            
        # Clean and process query
        cleaned_query = self.text_cleaner.clean(query)
        query_embedding = self.embedding_generator.generate(cleaned_query)
        
        # Vector similarity search
        vector_scores = {}
        for doc_id, doc in self.document_store.items():
            similarity = np.dot(query_embedding, doc['embedding'])
            vector_scores[doc_id] = similarity
            
        # Keyword search with BM25
        tokenized_query = cleaned_query.split()
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores
        hybrid_scores = {}
        for doc_id, vector_score in vector_scores.items():
            keyword_score = keyword_scores[int(doc_id)]  # Assuming sequential IDs
            hybrid_score = (
                self.hybrid_weight * vector_score +
                (1 - self.hybrid_weight) * keyword_score
            )
            hybrid_scores[doc_id] = hybrid_score
            
        # Get top results
        top_doc_ids = sorted(
            hybrid_scores.keys(),
            key=lambda x: hybrid_scores[x],
            reverse=True
        )[:top_k]
        
        # Prepare results
        results = []
        for doc_id in top_doc_ids:
            doc = self.document_store[doc_id]
            results.append(
                RetrievalResult(
                    document_id=doc_id,
                    content=doc['content'],
                    score=hybrid_scores[doc_id],
                    metadata=doc['metadata']
                )
            )
            
        # Apply reranking if enabled
        if self.use_reranking:
            self._rerank_results(query, results)
            
        return results
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]):
        """Rerank results using cross-encoder.
        
        Args:
            query: Original query
            results: List of retrieval results to rerank
        """
        pairs = [(query, result.content) for result in results]
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores and sort
        for result, score in zip(results, rerank_scores):
            result.score = score
            result.source = "reranked"
            
        results.sort(key=lambda x: x.score, reverse=True)
        
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a specific document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document data if found, None otherwise
        """
        return self.document_store.get(doc_id)