"""
Utils Package
------------

Utility functions and classes for the RAG pipeline.

Components:
- Text cleaning and normalization
- Token counting and management
- Image processing utilities
- RAG-specific tokenization
- File operations
- Validation utilities
- Text utilities
- Conversion helpers

Key Features:
- Modular utility functions
- Consistent error handling
- Logging integration
- Configuration management
- Type safety
- Performance optimization

Dependencies:
- tiktoken>=0.5.0
- numpy>=1.24.0
- opencv-python>=4.8.0
- Pillow>=10.0.0
- nltk>=3.8.0
- hanziconv>=0.3.2
- datrie>=0.8.0

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Optional, Any
import logging
from datetime import datetime

# Import utilities
from .text_cleaner import TextCleaner
from .token_counter import TokenCounter
from .image_utils import ImageUtils
from .rag_tokenizer import RagTokenizer

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Type aliases
ConfigDict = Dict[str, Any]

def init_utils(config: Optional[ConfigDict] = None) -> Dict[str, Any]:
    """Initialize utility components."""
    try:
        config = config or {}
        
        return {
            'text_cleaner': TextCleaner(config.get('text_cleaner')),
            'token_counter': TokenCounter(config.get('token_counter')),
            'image_utils': ImageUtils(config.get('image_utils')),
            'rag_tokenizer': RagTokenizer(config.get('rag_tokenizer'))
        }
        
    except Exception as e:
        logger.error(f"Utils initialization error: {str(e)}")
        raise

# Export utilities
__all__ = [
    'TextCleaner',
    'TokenCounter',
    'ImageUtils',
    'RagTokenizer',
    'init_utils'
] 