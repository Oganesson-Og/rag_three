"""
Text Utilities
-----------

Text processing and manipulation utilities.

Key Features:
- Text normalization
- Text sanitization
- Email extraction
- URL extraction
- Text metrics
- Case conversion
- Text truncation

Technical Details:
- Regex processing
- String manipulation
- Metrics calculation
- Case handling
- Pattern matching

Dependencies:
- re
- typing-extensions>=4.7.0

Example Usage:
    utils = TextUtils()
    
    # Normalize text
    clean = utils.normalize_text("Some   messy    text")
    
    # Extract emails
    emails = utils.extract_emails("Contact user@example.com")
    
    # Get text metrics
    metrics = utils.get_text_metrics("Sample text for analysis")
    
    # Convert case
    snake = utils.to_snake_case("camelCase")
    camel = utils.to_camel_case("snake_case")

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Union, Optional
import re
from datetime import datetime

class TextUtils:
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
        
    def sanitize_text(self, text: str) -> str:
        """Remove potentially unsafe content from text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
        
    def extract_emails(self, text: str) -> List[str]:
        return re.findall(r'[\w\.-]+@[\w\.-]+', text)
        
    def extract_urls(self, text: str) -> List[str]:
        return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
    def get_text_metrics(self, text: str) -> Dict[str, Union[int, float]]:
        """Calculate various text metrics."""
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text else 0
        }
        
    def to_snake_case(self, text: str) -> str:
        text = re.sub(r'([A-Z])', r'_\1', text)
        return text.lower().lstrip('_')
        
    def to_camel_case(self, text: str) -> str:
        components = text.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
        
    def to_pascal_case(self, text: str) -> str:
        return ''.join(x.title() for x in text.split('_'))
        
    def truncate(self, text: str, max_length: int) -> str:
        """Truncate text to specified length."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..." 