"""
RAG Tokenizer Module
----------------
Simple English text tokenization and normalization.
"""

from typing import Dict, List, Optional, Any
import logging
import os
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class RagTokenizer:
    """RAG-specific tokenization utilities focused on English text."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> None:
        self.config = config or {}
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(
        self,
        text: str,
        options: Optional[Dict[str, bool]] = None
    ) -> str:
        """
        Tokenize English text.
        
        Args:
            text: Input text to tokenize
            options: Optional configuration for tokenization
            
        Returns:
            Tokenized and normalized text string
        """
        try:
            options = options or {}
            
            # Clean and normalize text
            text = re.sub(r"\W+", " ", text)
            text = self._convert_fullwidth(text).lower()
            
            # Tokenize and normalize
            tokens = word_tokenize(text)
            normalized = self._normalize_english(tokens)
            
            return " ".join(normalized)
            
        except Exception as e:
            self.logger.error(f"Tokenization error: {str(e)}")
            raise

    def _convert_fullwidth(self, text: str) -> str:
        """Convert fullwidth characters to halfwidth."""
        result = ""
        for char in text:
            code = ord(char)
            if code == 0x3000:
                code = 0x0020
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            result += chr(code)
        return result

    def _normalize_english(self, tokens: List[str]) -> List[str]:
        """
        Normalize English tokens using stemming and lemmatization.
        
        Args:
            tokens: List of tokens to normalize
            
        Returns:
            List of normalized tokens
        """
        return [
            self.stemmer.stem(self.lemmatizer.lemmatize(token))
            for token in tokens
        ]

    def fine_grained_tokenize(self, text: str) -> str:
        """
        Perform fine-grained tokenization of text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Fine-grained tokenized text
        """
        tokens = text.split()
        result = []
        
        for token in tokens:
            # Handle special cases
            if len(token) < 3 or re.match(r"[0-9,\.-]+$", token):
                result.append(token)
                continue
                
            # Split on camelCase and special characters
            subtokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', token)
            if subtokens:
                result.extend(self._normalize_english(subtokens))
            else:
                result.append(token)
                
        return " ".join(result)


# Create default instance
tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize

if __name__ == '__main__':
    tknzr = RagTokenizer(debug=True)
    
    # Test with English examples
    test_cases = [
        "Scripts are compiled and cached",
        "Unity3D development experience testing engineer",
        "Data analysis project manager | data mining | SQL Python Hive Tableau",
        "The quick brown fox jumps over the lazy dog",
        "Testing multiple   spaces   and punctuation!!!",
        "Testing123 with numbers 456 and special-chars_"
    ]
    
    for text in test_cases:
        tks = tknzr.tokenize(text)
        logging.info(f"Original: {text}")
        logging.info(f"Tokenized: {tks}")
        logging.info(f"Fine-grained: {tknzr.fine_grained_tokenize(tks)}\n")
