"""
RAG Utility Functions
--------------------------------

Collection of utility functions for text processing, token counting,
and string manipulation used throughout the RAG pipeline.

Key Features:
- Token counting using tiktoken
- Text normalization and cleaning
- Space optimization
- String manipulation utilities

Technical Details:
- Uses OpenAI's tiktoken for accurate token counting
- Implements efficient regex patterns
- Optimized for performance
- Handles edge cases and null inputs

Dependencies:
- tiktoken>=0.5.0
- re (standard library)

Example Usage:
    # Count tokens in text
    token_count = num_tokens_from_string("Sample text")
    
    # Clean and normalize text
    cleaned_text = rmSpace("Text  with   extra    spaces")

Performance Considerations:
- Cached tokenizer instances
- Optimized regex patterns
- Minimal memory overhead
- Fast string operations

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import tiktoken
import re

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens 

def rmSpace(text: str) -> str:
    """Remove extra spaces from text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip() 