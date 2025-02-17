"""
Query Module
----------

Advanced query processing for the RAG pipeline with backwards compatibility.

Key Features:
- Query processing
- Token handling
- Hybrid similarity
- Backwards compatibility
- Weight management
- Query enhancement
- Performance optimization

Technical Details:
- Token processing
- Vector operations
- Query building
- Similarity scoring
- Weight normalization
- Expression building
- Error handling

Dependencies:
- numpy>=1.24.0
- typing (standard library)
- re (standard library)

Example Usage:
    # Initialize processor
    processor = QueryProcessor()
    
    # Process query
    query_expr, keywords = processor.process_query(
        query="quantum mechanics",
        min_match=0.6
    )
    
    # Calculate hybrid similarity
    scores, token_scores, vector_scores = processor.hybrid_similarity(
        query_vec=query_embeddings,
        doc_vecs=document_embeddings,
        query_tokens=query_tokens,
        doc_tokens=document_tokens
    )

Query Types:
- Educational queries
- Research questions
- Content searches
- Semantic matching

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Tuple, Optional
import re
import numpy as np
from .tokenizer import UnifiedTokenizer, Token, TokenType
from ..utils.text_cleaner import TextCleaner

class QueryProcessor:
    """Advanced query processing."""
    
    def __init__(self):
        self.tokenizer = UnifiedTokenizer()
        self.text_cleaner = TextCleaner()
        self.query_fields = [
            "title^10",
            "keywords^30",
            "content^2",
            "summary^5"
        ]
    
    def process_query(
        self,
        query: str,
        min_match: float = 0.6
    ) -> Tuple[Dict, List[str]]:
        """Process and enhance query."""
        # Clean and normalize query
        query = self.text_cleaner.clean_text(query)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(query)
        
        # Extract keywords (only words and numbers)
        keywords = [token.text for token in tokens 
                   if token.type in [TokenType.WORD, TokenType.NUMBER]]
        
        # Build query expression
        query_expr = self._build_query_expression(tokens, min_match)
        
        return query_expr, keywords
    
    def _build_query_expression(
        self,
        tokens: List[Token],
        min_match: float
    ) -> Dict:
        """Build Elasticsearch-style query expression."""
        should_clauses = []
        
        # Process single tokens
        for token in tokens:
            if token.type in [TokenType.WORD, TokenType.NUMBER]:
                should_clauses.append({
                    "match": {
                        "_all": {
                            "query": token.text,
                            "boost": token.weight
                        }
                    }
                })
        
        # Process token pairs for phrases
        for i in range(len(tokens) - 1):
            if (tokens[i].type in [TokenType.WORD, TokenType.NUMBER] and 
                tokens[i + 1].type in [TokenType.WORD, TokenType.NUMBER]):
                phrase = f"{tokens[i].text} {tokens[i + 1].text}"
                should_clauses.append({
                    "match_phrase": {
                        "_all": {
                            "query": phrase,
                            "boost": max(tokens[i].weight, tokens[i + 1].weight) * 2
                        }
                    }
                })
        
        return {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": f"{int(min_match * 100)}%"
                }
            }
        }
    
    def hybrid_similarity(
        self,
        query_vec: np.ndarray,
        doc_vecs: List[np.ndarray],
        query_tokens: List[Token],
        doc_tokens: List[List[Token]],
        token_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate hybrid similarity scores."""
        # Vector similarity
        vector_sim = self._cosine_similarity(query_vec, doc_vecs)
        
        # Token similarity
        token_sim = self._token_similarity(query_tokens, doc_tokens)
        
        # Combine similarities
        hybrid_sim = (np.array(vector_sim) * vector_weight + 
                     np.array(token_sim) * token_weight)
        
        return hybrid_sim, token_sim, vector_sim
    
    def _cosine_similarity(
        self,
        query_vec: np.ndarray,
        doc_vecs: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate cosine similarity."""
        doc_vecs = np.array(doc_vecs)
        return np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        )
    
    def _token_similarity(
        self,
        query_tokens: List[Token],
        doc_tokens: List[List[Token]]
    ) -> List[float]:
        """Calculate token-based similarity."""
        query_weights = {t.text: t.weight for t in query_tokens}
        similarities = []
        
        for doc_toks in doc_tokens:
            doc_weights = {t.text: t.weight for t in doc_toks}
            
            # Calculate intersection score
            score = sum(
                min(query_weights.get(token, 0), weight)
                for token, weight in doc_weights.items()
            )
            
            # Normalize by query weights
            total_query_weight = sum(query_weights.values())
            if total_query_weight > 0:
                score /= total_query_weight
                
            similarities.append(score)
        
        return similarities

class FulltextQueryer:
    """Compatibility class for old query interface."""
    
    def __init__(self):
        self._processor = QueryProcessor()
    
    def token_similarity(self, atks, btkss):
        """Maintain compatibility with old token_similarity method."""
        if isinstance(atks, str):
            atks = self._processor.tokenizer.tokenize(atks)
        
        if isinstance(btkss[0], str):
            btkss = [self._processor.tokenizer.tokenize(b) for b in btkss]
            
        return self._processor._token_similarity(atks, btkss)
    
    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        """Maintain compatibility with old hybrid_similarity method."""
        return self._processor.hybrid_similarity(
            avec, bvecs,
            self._processor.tokenizer.tokenize(atks) if isinstance(atks, str) else atks,
            [self._processor.tokenizer.tokenize(b) if isinstance(b, str) else b for b in btkss],
            token_weight=tkweight,
            vector_weight=vtweight
        )
    
    def question(self, txt, tbl="qa", min_match: float = 0.6):
        """Maintain compatibility with old question method."""
        return self._processor.process_query(txt, min_match)

# Initialize default query processor
default_query_processor = QueryProcessor()
