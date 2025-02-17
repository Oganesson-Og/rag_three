"""
Text Cleaning Module
------------------

Text cleaning and preprocessing utilities.

Key Features:
- Basic text cleaning
- Special character removal
- LaTeX cleaning
- HTML cleaning
- Whitespace normalization
- Line break fixing
- Unicode normalization

Technical Details:
- Regex-based cleaning
- Unicode handling
- Error handling
- Configurable options
- Logging integration

Dependencies:
- re
- unicodedata
- typing-extensions>=4.7.0

Example Usage:
    cleaner = TextCleaner()
    
    # Basic cleaning
    clean_text = cleaner.clean_text("Some  messy    text!")
    
    # Remove special characters
    clean_text = cleaner.remove_special_characters("Hello! @#$%", keep_chars="!")
    
    # Clean LaTeX
    clean_text = cleaner.clean_latex("Text with $\alpha$ math")
    
    # Clean HTML
    clean_text = cleaner.clean_html("<p>Some text</p>")

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

import re
import unicodedata
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

class TextCleaner:
    """Text cleaning and normalization utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def clean_text(
        self,
        text: str,
        options: Optional[Dict[str, bool]] = None
    ) -> str:
        """Basic text cleaning."""
        try:
            options = options or {}
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Apply additional cleaning if requested
            if options.get('remove_special_chars', False):
                text = self.remove_special_characters(text)
                
            if options.get('clean_latex', False):
                text = self.clean_latex(text)
                
            if options.get('clean_html', False):
                text = self.clean_html(text)
                
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Text cleaning error: {str(e)}")
            raise

    def remove_special_characters(
        self,
        text: str,
        keep_chars: Optional[str] = None
    ) -> str:
        """Remove special characters except those specified."""
        try:
            if keep_chars:
                pattern = f'[^a-zA-Z0-9\s{re.escape(keep_chars)}]'
            else:
                pattern = '[^a-zA-Z0-9\s]'
            return re.sub(pattern, '', text)
            
        except Exception as e:
            self.logger.error(f"Character removal error: {str(e)}")
            raise

    def clean_latex(self, text: str) -> str:
        """Clean LaTeX commands and environments."""
        try:
            # Remove inline math
            text = re.sub(r'\$[^$]+\$', '', text)
            # Remove display math
            text = re.sub(r'\\\[[^\]]+\\\]', '', text)
            # Remove LaTeX commands
            text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
            return text
            
        except Exception as e:
            self.logger.error(f"LaTeX cleaning error: {str(e)}")
            raise

    def clean_html(self, text: str) -> str:
        """Remove HTML tags."""
        try:
            return re.sub(r'<[^>]+>', '', text)
        except Exception as e:
            self.logger.error(f"HTML cleaning error: {str(e)}")
            raise

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        try:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            # Remove spaces at start/end of lines
            text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Whitespace normalization error: {str(e)}")
            raise

    def fix_line_breaks(self, text: str) -> str:
        """Fix incorrect line breaks."""
        try:
            # Remove breaks in middle of sentences
            text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)
            # Keep paragraph breaks
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text
            
        except Exception as e:
            self.logger.error(f"Line break fixing error: {str(e)}")
            raise