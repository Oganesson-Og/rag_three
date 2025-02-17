"""
Base Query Processing Module
--------------------------------

Core query processing system designed for general content analysis and retrieval,
providing foundational RAG pipeline functionality.

Key Features:
- Query analysis and preprocessing
- Entity extraction and keyword identification
- Query type classification
- Query expansion capabilities
- Embedding generation
- Domain-specific processing adaptability

Technical Details:
- Uses SpaCy for linguistic analysis
- Implements configurable preprocessing pipeline
- Supports domain-specific configurations
- Generates embeddings for semantic search
- Handles various query types (questions, commands, topics)

Dependencies:
- spacy>=3.7.2
- numpy>=1.24.0
- pydantic>=2.5.0

Example Usage:
    # Basic query processing
    processor = QueryProcessor(domain='general')
    result = processor.process_query(query_text)

    # Advanced processing with options
    result = processor.process_query(
        query_text,
        expand=True,
        classify=True
    )

Performance Considerations:
- Efficient preprocessing pipeline
- Optimized embedding generation
- Configurable processing steps
- Domain-specific optimizations

Author: Keith Satuku
Version: 1.0.0
Created: 2024
License: MIT
"""

from typing import List, Dict, Optional
import spacy
from ..embeddings.enhanced_generator import EnhancedEmbeddingGenerator
from ..config.embedding_config import DOMAIN_CONFIGS

class QueryProcessor:
    """Process and enhance queries for improved retrieval."""
    
    def __init__(self, domain: str = 'general'):
        self.domain = domain
        self.domain_config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS['general'])
        
        # Initialize embedding generator with domain-specific config
        self.embedding_generator = EnhancedEmbeddingGenerator(
            model_key=self.domain_config['model']
        )
        
        # Load NLP model for query analysis
        self.nlp = spacy.load('en_core_web_sm')
        
    def process_query(
        self,
        query: str,
        expand: bool = True,
        classify: bool = True
    ) -> Dict:
        """Process and enhance the query.
        
        Args:
            query: Original query text
            expand: Whether to perform query expansion
            classify: Whether to classify query type
            
        Returns:
            Dictionary containing processed query information
        """
        # Basic query cleanup
        query = query.strip()
        
        # Analyze query
        doc = self.nlp(query)
        
        # Extract key information
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        
        # Classify query type
        query_type = self._classify_query(doc) if classify else None
        
        # Expand query if requested
        expanded_queries = self._expand_query(query) if expand else []
        
        # Generate embeddings
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        return {
            'original_query': query,
            'processed_query': self._preprocess_query(query),
            'query_type': query_type,
            'entities': entities,
            'keywords': keywords,
            'expanded_queries': expanded_queries,
            'embedding': query_embedding.tolist()
        }
    
    def _preprocess_query(self, query: str) -> str:
        """Apply domain-specific preprocessing."""
        # Apply configured preprocessing steps
        for step in self.domain_config['preprocessing']:
            if step == 'clean_latex':
                query = self._clean_latex(query)
            elif step == 'expand_abbreviations':
                query = self._expand_abbreviations(query)
            elif step == 'normalize_unicode':
                query = self._normalize_unicode(query)
            # Add more preprocessing steps as needed
            
        return query
    
    def _classify_query(self, doc: spacy.tokens.Doc) -> str:
        """Classify query type based on structure and content."""
        # Question classification
        if doc[0].text.lower() in ['what', 'who', 'where', 'when', 'why', 'how']:
            return 'question'
        
        # Command classification
        if doc[0].pos_ == 'VERB':
            return 'command'
            
        # Topic classification
        if len(doc) < 3 and all(token.pos_ in ['NOUN', 'ADJ'] for token in doc):
            return 'topic'
            
        return 'statement'
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with related terms and variations."""
        doc = self.nlp(query)
        expanded = []
        
        # Add noun chunks
        expanded.extend([chunk.text for chunk in doc.noun_chunks])
        
        # Add entity variations
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                # Add without qualifiers
                expanded.append(ent.text.split()[-1])
            
        # Remove duplicates and original query
        expanded = list(set(expanded) - {query})
        
        return expanded
    
    def _clean_latex(self, text: str) -> str:
        """Clean LaTeX markup from text."""
        import re
        # Remove inline math
        text = re.sub(r'\$[^$]+\$', '', text)
        # Remove display math
        text = re.sub(r'\\\[[^\]]+\\\]', '', text)
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand domain-specific abbreviations."""
        # Implementation depends on domain
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        import unicodedata
        return unicodedata.normalize('NFKC', text)