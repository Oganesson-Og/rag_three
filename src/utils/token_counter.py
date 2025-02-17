"""
Token Counter Module
------------------

Token counting utilities using tiktoken.

Key Features:
- Token counting
- Token ID extraction
- Token decoding
- Multiple encodings support
- Error handling
- Configuration options

Technical Details:
- tiktoken integration
- Encoding management
- Error handling
- Logging support
- Type safety

Dependencies:
- tiktoken>=0.5.0
- typing-extensions>=4.7.0

Example Usage:
    counter = TokenCounter()
    
    # Count tokens
    count = counter.count_tokens("Some text to count")
    
    # Get token IDs
    ids = counter.get_token_ids("Convert to token IDs")
    
    # Decode tokens
    text = counter.decode_tokens([1234, 5678])

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Optional, Any
import tiktoken
import logging
from datetime import datetime

class TokenCounter:
    """Token counting utilities."""
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.encoding_name = encoding_name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self._init_encoding()

    def _init_encoding(self) -> None:
        """Initialize tiktoken encoding."""
        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
        except Exception as e:
            self.logger.error(f"Encoding initialization error: {str(e)}")
            raise

    def count_tokens(
        self,
        text: str,
        options: Optional[Dict[str, bool]] = None
    ) -> int:
        """Count tokens in text."""
        try:
            options = options or {}
            return len(self.encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Token counting error: {str(e)}")
            raise

    def get_token_ids(
        self,
        text: str,
        options: Optional[Dict[str, bool]] = None
    ) -> list:
        """Get token IDs for text."""
        try:
            options = options or {}
            return self.encoding.encode(text)
        except Exception as e:
            self.logger.error(f"Token ID extraction error: {str(e)}")
            raise

    def decode_tokens(
        self,
        token_ids: list,
        options: Optional[Dict[str, bool]] = None
    ) -> str:
        """Decode token IDs back to text."""
        try:
            options = options or {}
            return self.encoding.decode(token_ids)
        except Exception as e:
            self.logger.error(f"Token decoding error: {str(e)}")
            raise

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens 