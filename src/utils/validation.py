"""
Validation Utils Module
-------------------

Input validation utilities.

Key Features:
- Text length validation
- Numeric range validation
- Email format validation
- URL format validation
- Date format validation
- Error handling
- Logging support

Technical Details:
- Regex validation
- Type checking
- Range checking
- Format verification
- Error handling
- Logging integration

Dependencies:
- re
- typing-extensions>=4.7.0
- datetime

Example Usage:
    validator = ValidationUtils()
    
    # Validate text length
    is_valid = validator.validate_text_length("text", min_length=2, max_length=10)
    
    # Validate numeric range
    is_valid = validator.validate_numeric_range(5, min_value=0, max_value=10)
    
    # Validate email
    is_valid = validator.validate_email("user@example.com")
    
    # Validate URL
    is_valid = validator.validate_url("https://example.com")

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import re

class ValidationUtils:
    """Validation utilities."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def validate_text_length(
        self,
        text: str,
        min_length: int = 1,
        max_length: Optional[int] = None
    ) -> bool:
        """Validate text length."""
        try:
            if len(text) < min_length:
                return False
            if max_length and len(text) > max_length:
                return False
            return True
            
        except Exception as e:
            self.logger.error(f"Text length validation error: {str(e)}")
            raise

    def validate_numeric_range(
        self,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> bool:
        """Validate numeric range."""
        try:
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False
            return True
            
        except Exception as e:
            self.logger.error(f"Numeric range validation error: {str(e)}")
            raise

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        try:
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))
            
        except Exception as e:
            self.logger.error(f"Email validation error: {str(e)}")
            raise

    def validate_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            pattern = r'^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$'
            return bool(re.match(pattern, url))
            
        except Exception as e:
            self.logger.error(f"URL validation error: {str(e)}")
            raise

    def validate_date_format(
        self,
        date_str: str,
        format: str = '%Y-%m-%d'
    ) -> bool:
        """Validate date string format."""
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False
        except Exception as e:
            self.logger.error(f"Date validation error: {str(e)}")
            raise 