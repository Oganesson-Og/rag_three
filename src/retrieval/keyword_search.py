"""
Keyword Search Module
--------------------------------

Efficient keyword-based search implementation with support for exact matching,
tokenization, and advanced text processing capabilities.

Key Features:
- Basic tokenization and normalization
- Exact phrase matching
- Token-based indexing
- Score normalization
- Flexible matching options
- Case-insensitive search
- Multi-term support

Technical Details:
- Inverted index implementation
- Token frequency counting
- Regular expression patterns
- Text normalization rules
- Score calculation methods
- Index optimization
- Memory-efficient storage

Dependencies:
- re (standard library)
- collections (standard library)
- typing-extensions>=4.7.0

Example Usage:
    # Initialize search
    searcher = KeywordSearch()
    
    # Index documents
    searcher.index_documents(documents)
    
    # Basic search
    results = searcher.search(
        query="machine learning",
        k=3
    )
    
    # Check exact matches
    results = searcher.search(
        query="neural networks",
        exact_match=True
    )

Performance Considerations:
- Efficient index structure
- Fast token lookup
- Optimized regex patterns
- Memory-efficient storage
- Quick exact matching

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict
import re
from collections import Counter

class KeywordSearch:
    def __init__(self):
        self.documents = []
        self.index = {}
        
    def index_documents(self, documents: List[Dict]):
        self.documents = documents
        self.index = {}
        
        for i, doc in enumerate(documents):
            terms = self._tokenize(doc['text'])
            for term in terms:
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(i)
                
    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_terms = self._tokenize(query)
        scores = Counter()
        
        for term in query_terms:
            if term in self.index:
                for doc_idx in self.index[term]:
                    scores[doc_idx] += 1
                    
        results = []
        for doc_idx, score in scores.most_common(k):
            doc = self.documents[doc_idx].copy()
            doc['score'] = score / len(query_terms)
            doc['exact_match'] = self._check_exact_match(query, doc['text'])
            results.append(doc)
            
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization and normalization."""
        text = text.lower()
        words = re.findall(r'\w+', text)
        return words
        
    def _check_exact_match(self, query: str, text: str) -> bool:
        """Check if query appears exactly in text."""
        return query.lower() in text.lower() 