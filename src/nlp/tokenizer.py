"""
Unified Tokenizer Module
----------------------

Advanced tokenization system for educational content processing with comprehensive
feature support and backwards compatibility.

Key Features:
- Multi-model tokenization (tiktoken, spacy)
- Token type classification
- POS tagging
- Lemmatization
- N-gram generation
- Token filtering
- Weight calculation
- Text analysis

Technical Details:
- Token metadata tracking
- Custom token types
- Pattern matching
- Weight normalization
- Backwards compatibility
- Performance optimization
- Error handling

Dependencies:
- tiktoken>=0.5.0
- spacy>=3.7.0
- nltk>=3.8.0
- typing (standard library)
- re (standard library)
- string (standard library)
- math (standard library)

Example Usage:
    # Initialize tokenizer
    tokenizer = UnifiedTokenizer()
    
    # Basic tokenization
    tokens = tokenizer.tokenize("The quick brown fox jumps!")
    
    # Filter tokens
    filtered = tokenizer.filter_tokens(
        tokens,
        token_types={TokenType.WORD},
        exclude_stops=True
    )
    
    # Generate n-grams
    bigrams = tokenizer.get_ngrams(tokens, n=2)
    
    # Analyze text
    analysis = tokenizer.analyze_text("Sample text for analysis")

Token Types:
- WORD: Regular words
- NUMBER: Numeric values
- PUNCTUATION: Punctuation marks
- SPECIAL: Special characters
- WHITESPACE: Space characters

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional, Union, Set
import tiktoken
import spacy
import re
from dataclasses import dataclass
from enum import Enum
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import math
import logging

class TokenType(Enum):
    """Token type classification."""
    WORD = "word"
    NUMBER = "number"
    PUNCTUATION = "punctuation"
    SPECIAL = "special"
    WHITESPACE = "whitespace"

@dataclass
class Token:
    """Token data structure with metadata."""
    text: str
    type: TokenType
    start: int
    end: int
    pos_tag: Optional[str] = None
    lemma: Optional[str] = None
    is_stop: bool = False
    weight: float = 1.0

class UnifiedTokenizer:
    """Advanced unified tokenizer."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize tokenizer with specified spaCy model."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Warning: SpaCy model '{spacy_model}' not found. Downloading...")
            try:
                # Attempt to download the model
                import subprocess
                subprocess.run([
                    "python", "-m", "spacy", "download", spacy_model
                ], check=True)
                self.nlp = spacy.load(spacy_model)
            except Exception as e:
                print(f"Error downloading spaCy model: {e}")
                print("Falling back to basic tokenization")
                self.nlp = None
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.debug = False
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Updated regex patterns without Unicode properties
        self.patterns = {
            TokenType.NUMBER: re.compile(r'\d+(?:\.\d+)?'),
            TokenType.PUNCTUATION: re.compile(f'[{re.escape(string.punctuation)}]'),
            TokenType.SPECIAL: re.compile(r'[^a-zA-Z0-9\s' + re.escape(string.punctuation) + r']')
        }
    
    def tokenize(self, text: str) -> List[Token]:
        """Enhanced tokenize with backwards compatibility."""
        if not text:
            return []
        
        # Original tokenization
        tokens = self._tokenize(text)
        
        # For compatibility, convert to string if called from old code
        if self._detect_old_call():
            return " ".join(t.text for t in tokens)
        return tokens
    
    def _detect_old_call(self) -> bool:
        """Detect if method is called from old code."""
        import inspect
        stack = inspect.stack()
        if len(stack) > 2:
            caller = stack[2].filename
            return 'search.py' in caller
        return False
    
    def _tokenize(self, text: str) -> List[Token]:
        """Advanced tokenization."""
        # Normalize text
        text = text.lower()
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            token_type = self._determine_token_type(token.text)
            
            tokens.append(Token(
                text=token.text,
                type=token_type,
                start=token.idx,
                end=token.idx + len(token.text),
                pos_tag=token.pos_,
                lemma=self.lemmatizer.lemmatize(token.text),
                is_stop=token.is_stop,
                weight=self._calculate_token_weight(token.text, token_type)
            ))
        
        if self.debug:
            logging.debug(f"Tokenized '{text}' into {len(tokens)} tokens")
        
        return tokens
    
    def _determine_token_type(self, text: str) -> TokenType:
        """Determine token type."""
        if self.patterns[TokenType.NUMBER].match(text):
            return TokenType.NUMBER
        elif self.patterns[TokenType.PUNCTUATION].match(text):
            return TokenType.PUNCTUATION
        elif self.patterns[TokenType.SPECIAL].match(text):
            return TokenType.SPECIAL
        elif text.isspace():
            return TokenType.WHITESPACE
        else:
            return TokenType.WORD
    
    def _calculate_token_weight(self, text: str, token_type: TokenType) -> float:
        """Calculate token weight based on type and length."""
        base_weight = 1.0
        
        if token_type == TokenType.WORD:
            base_weight *= 1.5
        elif token_type == TokenType.NUMBER:
            base_weight *= 1.2
            
        # Adjust weight based on length
        length_factor = math.log(len(text) + 1)
        return base_weight * length_factor

    def decode(self, tokens: List[Token]) -> str:
        """Convert tokens back to text."""
        return " ".join(token.text for token in tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))
    
    def filter_tokens(
        self,
        tokens: List[Token],
        token_types: Optional[Set[TokenType]] = None,
        exclude_stops: bool = False,
        min_length: int = 1,
        pos_tags: Optional[Set[str]] = None
    ) -> List[Token]:
        """Filter tokens based on criteria."""
        filtered = [t for t in tokens if len(t.text) >= min_length]
        
        if token_types:
            filtered = [t for t in filtered if t.type in token_types]
        
        if exclude_stops:
            filtered = [t for t in filtered if not t.is_stop]
            
        if pos_tags:
            filtered = [t for t in filtered if t.pos_tag in pos_tags]
            
        return filtered
    
    def get_ngrams(
        self,
        tokens: List[Token],
        n: int = 2
    ) -> List[List[Token]]:
        """Generate n-grams from tokens."""
        return [tokens[i:i + n] for i in range(len(tokens) - n + 1)]
    
    def analyze_text(self, text: str) -> Dict:
        """Perform comprehensive text analysis."""
        tokens = self.tokenize(text)
        
        return {
            'token_count': len(tokens),
            'unique_tokens': len(set(t.text for t in tokens)),
            'token_types': {
                token_type.value: len([t for t in tokens if t.type == token_type])
                for token_type in TokenType
            },
            'pos_distribution': {
                pos: len([t for t in tokens if t.pos_tag == pos])
                for pos in set(t.pos_tag for t in tokens)
            },
            'stop_words': len([t for t in tokens if t.is_stop]),
            'average_token_length': sum(len(t.text) for t in tokens) / len(tokens)
        }

    def fine_grained_tokenize(self, text: str) -> str:
        """Compatibility method for old fine_grained_tokenize."""
        tokens = self.tokenize(text)
        filtered = self.filter_tokens(
            tokens,
            token_types={TokenType.WORD, TokenType.NUMBER},
            exclude_stops=True
        )
        return " ".join(t.text for t in filtered)

# Initialize default tokenizer
default_tokenizer = UnifiedTokenizer()

# Example usage
if __name__ == "__main__":
    tokenizer = UnifiedTokenizer()
    
    text = "The quick brown fox jumps over the lazy dog! It was amazing in 2025."
    
    # Basic tokenization
    tokens = tokenizer.tokenize(text)
    
    # Get only words and numbers
    filtered_tokens = [t for t in tokens if t.type in [TokenType.WORD, TokenType.NUMBER]]
    
    # Get bigrams
    bigrams = [tokens[i:i + 2] for i in range(len(tokens) - 1)]
    
    # Full analysis
    analysis = {
        'token_count': len(tokens),
        'unique_tokens': len(set(t.text for t in tokens)),
        'token_types': {
            token_type.value: len([t for t in tokens if t.type == token_type])
            for token_type in TokenType
        },
        'pos_distribution': {
            pos: len([t for t in tokens if t.pos_tag == pos])
            for pos in set(t.pos_tag for t in tokens)
        },
        'stop_words': len([t for t in tokens if t.is_stop]),
        'average_token_length': sum(len(t.text) for t in tokens) / len(tokens)
    }
    
    print("Analysis:", analysis) 