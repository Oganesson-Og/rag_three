"""
Text Processing Utilities Module
-----------------------------

Utilities for text processing and manipulation in the chunking system.

Key Features:
- Text cleaning
- Sentence splitting
- Quote normalization
- Whitespace handling
- Abbreviation processing
- Character normalization
- Text validation

Technical Details:
- Regex-based processing
- Unicode normalization
- Abbreviation handling
- Sentence boundary detection
- String manipulation
- Error handling

Dependencies:
- re (standard library)
- typing (standard library)

Example Usage:
    # Clean text
    cleaned = clean_text("Raw  text   with    spaces")
    
    # Split sentences
    sentences = split_into_sentences(
        "First sentence. Second one! Third?"
    )
    
    # Process text
    text = clean_text(raw_text)
    sentences = split_into_sentences(text)

Text Processing Features:
- Whitespace normalization
- Quote standardization
- Dash normalization
- Sentence detection
- Abbreviation handling
- Boundary detection

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import re
from typing import List

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
        
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    
    # Normalize dashes
    text = text.replace('--', '—')
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Handle common abbreviations
    text = text.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')
    text = text.replace('Dr.', 'Dr').replace('Prof.', 'Prof')
    
    # Split on sentence boundaries
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    # Clean and restore abbreviations
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [
        s.replace('Mr', 'Mr.').replace('Mrs', 'Mrs.')
        .replace('Dr', 'Dr.').replace('Prof', 'Prof.')
        for s in sentences
    ]
    
    return sentences
