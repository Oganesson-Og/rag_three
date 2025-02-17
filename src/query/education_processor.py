"""
Educational Query Processing Module
--------------------------------

Specialized query processing system designed for educational content analysis and
enhancement, with focus on academic intent understanding and syllabus alignment.

Key Features:
- Subject-specific query processing
- Grade level adaptation and complexity analysis
- Educational intent detection and classification
- Syllabus mapping and topic alignment
- Query expansion with educational terms
- Multi-language support for educational content
- Context-aware query enhancement

Technical Details:
- Uses SpaCy for linguistic analysis
- Implements custom educational entity recognition
- Maintains subject-specific keyword dictionaries
- Supports multiple educational taxonomies
- Implements query caching for performance

Dependencies:
- spacy>=3.7.2
- nltk>=3.8.1
- scikit-learn>=1.3.0
- pandas>=2.1.0
- pydantic>=2.5.0

Example Usage:
    # Basic query processing
    processor = EducationQueryProcessor(
        subject='chemistry',
        level='a-level'
    )
    processed_query = processor.process_query(query_text)

    # Advanced processing with options
    result = processor.process_query(
        query_text,
        expand=True,
        classify=True
    )

Performance Considerations:
- Caches processed queries for efficiency
- Optimizes memory usage for large-scale processing
- Implements batch processing for multiple queries
- Uses efficient string matching algorithms

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional, Union, Any
import numpy as np
from numpy.typing import NDArray
import logging
from datetime import datetime


from .processor import QueryProcessor
from ..config.domain_config import get_domain_config, PREPROCESSING_CONFIGS
import spacy
import re



class EducationQueryProcessor(QueryProcessor):
    """Process educational queries with domain awareness."""
    
    def __init__(
        self,
        subject: str,
        level: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.subject = subject
        self.level = level
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.domain_config = get_domain_config(subject, level)
        super().__init__(domain=self.domain_config['domain'])
        
        # Load subject-specific patterns
        self.subject_patterns = self._load_subject_patterns()
        
    def _load_subject_patterns(self) -> Dict:
        """Load subject-specific patterns for query understanding."""
        return {
            'concept_patterns': [
                r'what\s+is\s+(?:a|an|the)?\s*([^?]+)',
                r'define\s+([^?]+)',
                r'explain\s+(?:the)?\s*([^?]+)'
            ],
            'calculation_patterns': [
                r'calculate\s+(?:the)?\s*([^?]+)',
                r'find\s+(?:the)?\s*([^?]+)',
                r'solve\s+for\s+([^?]+)'
            ],
            'application_patterns': [
                r'apply\s+(?:the)?\s*([^?]+)',
                r'use\s+(?:the)?\s*([^?]+)\s+to',
                r'how\s+(?:does|do|can)\s+([^?]+)'
            ]
        }
    
    def process_query(
        self,
        query: str,
        options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """Process educational query."""
        try:
            options = options or {}
            
            # Add logging
            self.logger.info(
                f"Processing query: {query}, "
                f"subject: {self.subject}, level: {self.level}"
            )
            
            result = {
                'original_query': query,
                'processed_query': self._preprocess_query(query),
                'subject': self.subject,
                'level': self.level,
                'timestamp': datetime.now().isoformat()
            }
            
            if options.get('extract_intent', True):
                result['intent'] = self._extract_intent(query)
                
            if options.get('extract_concepts', True):
                result['concepts'] = self._extract_concepts(query)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing error: {str(e)}")
            raise
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query text."""
        for step in self.domain_config['preprocessing']:
            config = PREPROCESSING_CONFIGS.get(step, {})
            
            # Apply patterns
            for pattern in config.get('patterns', []):
                query = re.sub(pattern, ' ', query)
            
            # Apply replacements
            for old, new in config.get('replacements', {}).items():
                query = query.replace(old, new)
        
        return query.strip().lower()
    
    def _extract_intent(self, query: str) -> str:
        """Extract educational intent."""
        query = query.lower()
        
        # Check concept understanding
        for pattern in self.subject_patterns['concept_patterns']:
            if re.search(pattern, query):
                return 'concept_understanding'
        
        # Check calculation/problem solving
        for pattern in self.subject_patterns['calculation_patterns']:
            if re.search(pattern, query):
                return 'problem_solving'
        
        # Check application
        for pattern in self.subject_patterns['application_patterns']:
            if re.search(pattern, query):
                return 'application'
        
        return 'general'
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract educational concepts."""
        matches = []
        doc = self.nlp(query.lower())
        
        # Match against domain keywords
        keywords = self.domain_config['keywords']
        for token in doc:
            if token.text in keywords:
                matches.append(token.text)
        
        return matches 