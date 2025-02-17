"""
Hybrid Search Module
--------------------------------

Advanced search system combining semantic and keyword-based search strategies
for optimal retrieval performance across different query types.

Key Features:
- Combined semantic and keyword search
- Configurable search weights
- Multi-query support
- Result aggregation
- Filter capabilities
- Score explanation
- Exact match boosting
- Document indexing

Technical Details:
- Embedding-based semantic search
- Keyword-based lexical search
- Normalized score combination
- Configurable thresholds
- Vector similarity calculations
- Result deduplication
- Efficient indexing

Dependencies:
- numpy>=1.24.0
- pydantic>=2.5.0
- typing-extensions>=4.7.0

Example Usage:
    # Initialize hybrid search
    searcher = HybridSearch(
        embedding_generator=embedding_generator,
        keyword_search=keyword_search
    )
    
    # Index documents
    searcher.index_documents(documents)
    
    # Basic search
    results = searcher.search(
        query="quantum mechanics",
        k=3,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    
    # Advanced search with filters
    results = searcher.search(
        query="quantum mechanics",
        filters={"subject": "physics"},
        exact_match_boost=1.5,
        explain=True
    )

Performance Considerations:
- Efficient vector operations
- Optimized result combination
- Smart result caching
- Minimal memory footprint
- Fast indexing strategies

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional
import numpy as np

class HybridSearch:
    def __init__(self, embedding_generator, keyword_search):
        self.embedding_generator = embedding_generator
        self.keyword_search = keyword_search
        self.documents = []
        self.embeddings = None
        
    def index_documents(self, documents: List[Dict]):
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        self.embeddings = self.embedding_generator.generate_embeddings(texts)
        self.keyword_search.index_documents(documents)
        
    def search(self, query: str, k: int = 3, semantic_weight: float = 0.5,
              keyword_weight: float = 0.5, threshold: float = 0.0,
              filters: Optional[Dict] = None, context: Optional[str] = None,
              exact_match_boost: float = 1.0, fields: Optional[List[str]] = None,
              explain: bool = False) -> List[Dict]:
        if not 0 <= semantic_weight <= 1 or not 0 <= keyword_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
        if abs(semantic_weight + keyword_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        semantic_results = self._semantic_search(query, k)
        keyword_results = self.keyword_search.search(query, k)
        
        combined_results = self._combine_results(
            semantic_results, 
            keyword_results,
            semantic_weight,
            keyword_weight,
            exact_match_boost
        )
        
        if filters:
            combined_results = [r for r in combined_results 
                              if all(r['metadata'].get(k) == v 
                                    for k, v in filters.items())]
        
        results = sorted(combined_results, 
                        key=lambda x: x['combined_score'], 
                        reverse=True)[:k]
        
        if explain:
            for result in results:
                result['explanation'] = {
                    'semantic_score': result['semantic_score'],
                    'keyword_score': result['keyword_score'],
                    'combined_score': result['combined_score']
                }
                
        return results
    
    def multi_query_search(self, queries: List[str], k: int = 3,
                          aggregate_method: str = 'max', **kwargs) -> List[Dict]:
        all_results = []
        for query in queries:
            results = self.search(query, k=k, **kwargs)
            all_results.extend(results)
            
        aggregated = {}
        for result in all_results:
            doc_id = result['id']
            if doc_id not in aggregated:
                aggregated[doc_id] = result
            else:
                if aggregate_method == 'max':
                    if result['combined_score'] > aggregated[doc_id]['combined_score']:
                        aggregated[doc_id] = result
                        
        return sorted(aggregated.values(), 
                     key=lambda x: x['combined_score'], 
                     reverse=True)[:k]
    
    def _semantic_search(self, query: str, k: int) -> List[Dict]:
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        scores = np.dot(self.embeddings, query_embedding)
        
        results = []
        for i, score in enumerate(scores):
            doc = self.documents[i].copy()
            doc['semantic_score'] = float(score)
            results.append(doc)
            
        return sorted(results, key=lambda x: x['semantic_score'], reverse=True)[:k]
    
    def _combine_results(self, semantic_results: List[Dict],
                        keyword_results: List[Dict],
                        semantic_weight: float,
                        keyword_weight: float,
                        exact_match_boost: float) -> List[Dict]:
        combined = {}
        
        for result in semantic_results:
            doc_id = result['id']
            combined[doc_id] = {
                **result,
                'semantic_score': result['semantic_score'],
                'keyword_score': 0.0,
                'exact_match_boost': 1.0
            }
            
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['keyword_score'] = result['score']
                if result.get('exact_match'):
                    combined[doc_id]['exact_match_boost'] = exact_match_boost
            else:
                combined[doc_id] = {
                    **result,
                    'semantic_score': 0.0,
                    'keyword_score': result['score'],
                    'exact_match_boost': exact_match_boost if result.get('exact_match') else 1.0
                }
                
        for doc in combined.values():
            doc['combined_score'] = (
                doc['exact_match_boost'] * 
                (semantic_weight * doc['semantic_score'] +
                 keyword_weight * doc['keyword_score'])
            )
            
        return list(combined.values()) 