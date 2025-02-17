"""
Utility Functions Module
----------------------

Common utility functions for the RAG pipeline.

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens 