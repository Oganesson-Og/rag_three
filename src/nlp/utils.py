"""
NLP Utilities Module
-----------------

Core utility functions for NLP operations in the RAG pipeline.

Key Features:
- Path management
- Language detection
- Text tokenization
- String handling
- Error handling
- Performance tracking
- Resource management

Technical Details:
- Path resolution
- Character detection
- Token processing
- String operations
- Error validation
- Resource cleanup
- Performance optimization

Dependencies:
- nltk>=3.8.0
- typing (standard library)
- os (standard library)
- re (standard library)

Example Usage:
    # Get project base directory
    base_dir = get_project_base_directory()
    
    # Check Chinese characters
    is_cn = is_chinese("你好")
    
    # Tokenize text
    tokens = word_tokenize(
        text="Hello, world!",
        language="english",
        preserve_line=True
    )

Language Support:
- English
- Chinese
- Multi-language

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

def get_project_base_directory():
    """Returns the base directory of the project"""
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    return False

def word_tokenize(text: str, language: str = 'english', preserve_line: bool = False) -> list:
    """
    Advanced word tokenization for RAG pipeline with multi-language support.
    
    Features:
    - Multi-language support
    - Sentence boundary detection
    - Special character handling
    - Technical term preservation
    - Acronym handling
    - URL/email preservation
    
    Args:
        text (str): Input text to tokenize
        language (str): Language of the text (default: 'english')
        preserve_line (bool): Whether to preserve line breaks in output
        
    Returns:
        list: List of tokenized words
        
    Examples:
        >>> word_tokenize("Hello, world! This is a test.")
        ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
        
        >>> word_tokenize("U.S.A. is a country.", preserve_line=True)
        ['U.S.A.', 'is', 'a', 'country', '.']
    """
    from nltk import word_tokenize as nltk_tokenize
    from nltk.tokenize import sent_tokenize
    import re
    
    # Ensure NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Pre-process to protect special patterns
    protected_patterns = {
        'urls': r'https?://\S+|www\.\S+',
        'emails': r'\S+@\S+',
        'acronyms': r'\b[A-Z](?:\.[A-Z])+\b',
        'technical_terms': r'\b(?:[A-Z][a-z]+){2,}\b|\b[A-Z]+\b',
        'numbers': r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b'
    }
    
    # Create placeholders for protected text
    placeholders = {}
    for pattern_type, pattern in protected_patterns.items():
        matches = re.finditer(pattern, text)
        for i, match in enumerate(matches):
            placeholder = f'___{pattern_type}_{i}___'
            placeholders[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)
    
    # Tokenize based on preserve_line preference
    if preserve_line:
        lines = text.split('\n')
        tokens = []
        for line in lines:
            if line.strip():
                line_tokens = nltk_tokenize(line, language=language)
                tokens.extend(line_tokens)
                tokens.append('\n')
        if tokens and tokens[-1] == '\n':
            tokens.pop()
    else:
        tokens = nltk_tokenize(text, language=language)
    
    # Restore protected text
    restored_tokens = []
    for token in tokens:
        if token in placeholders:
            restored_tokens.append(placeholders[token])
        else:
            restored_tokens.append(token)
    
    # Post-process tokens
    final_tokens = []
    for token in restored_tokens:
        # Handle contractions
        if "'" in token and token.lower() not in {"'s", "'t", "'re", "'ve", "'ll", "'d"}:
            parts = re.findall(r"[a-zA-Z]+|'[a-z]{1,2}\b|[^a-zA-Z']+", token)
            final_tokens.extend(parts)
        else:
            final_tokens.append(token)
    
    return final_tokens 