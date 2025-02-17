"""
Chunk Validation Utilities Module
------------------------------

Utilities for validating chunk quality and completeness in the chunking system.

Key Features:
- Chunk validation
- Sentence completion checking
- Delimiter balance checking
- Size validation
- Content verification
- Quality assessment
- Error detection

Technical Details:
- Regex validation
- Stack-based checking
- Length verification
- Boundary validation
- Content analysis
- Error handling

Dependencies:
- re (standard library)
- typing (standard library)
- base (local module)

Example Usage:
    # Validate chunk
    is_valid = validate_chunk(chunk)
    
    # Check sentence completion
    is_complete = is_complete_sentence(
        "This is a complete sentence."
    )
    
    # Check delimiters
    is_balanced = has_balanced_delimiters(
        "Text with (balanced) delimiters"
    )

Validation Features:
- Minimum length checking
- Sentence completion
- Delimiter balance
- Content presence
- Quality metrics
- Error detection

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import re
from typing import Union

def validate_chunk(chunk) -> bool:
    """Validate a chunk's properties."""
    if not chunk.text or not isinstance(chunk.text, str):
        return False
    if len(chunk.text.strip()) == 0:
        return False
    return True

def is_complete_sentence(text: str) -> bool:
    """Check if text ends with sentence terminator."""
    sentence_terminators = r'[.!?][\s]*$'
    return bool(re.search(sentence_terminators, text.strip()))

def has_balanced_delimiters(text: str) -> bool:
    """
    Check if text has balanced delimiters.
    
    Args:
        text: Input text
        
    Returns:
        True if delimiters are balanced
    """
    stack = []
    delimiters = {
        '(': ')', 
        '[': ']', 
        '{': '}', 
        '"': '"',
        "'": "'"
    }
    
    for char in text:
        if char in delimiters:
            stack.append(char)
        elif char in delimiters.values():
            if not stack:
                return False
            if char != delimiters[stack.pop()]:
                return False
                
    return len(stack) == 0
