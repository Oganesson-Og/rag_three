"""
Validation Utilities
-----------------

Comprehensive validation utilities for input verification and data quality checks.

Key Features:
- Email format validation
- URL format validation
- Numeric range validation
- Text length validation
- JSON validation
- File type validation
- Metadata validation
- Custom validation rules

Technical Details:
- Regex pattern matching
- Type checking
- Range verification
- Format validation
- Error handling
- Configurable rules
- Extensible design

Dependencies:
- re
- json
- pathlib
- typing-extensions>=4.7.0

Example Usage:
    validator = ValidationUtils()
    
    # Validate email
    is_valid = validator.is_valid_email("user@example.com")
    
    # Validate URL
    is_valid = validator.is_valid_url("https://example.com")
    
    # Check numeric range
    is_valid = validator.is_in_range(5, min_val=0, max_val=10)
    
    # Validate JSON
    is_valid = validator.is_valid_json('{"key": "value"}')
    
    # Check file type
    is_valid = validator.validate_file_type(
        "document.pdf",
        allowed_types=['.pdf', '.doc']
    )
    
    # Validate text length
    is_valid = validator.validate_text_length(
        "text",
        min_length=2,
        max_length=10
    )
    
    # Check metadata
    is_valid = validator.validate_metadata(
        metadata={"author": "John", "date": "2025-01-01"},
        required_fields=["author", "date"]
    )

Performance Considerations:
- Optimized regex patterns
- Efficient type checking
- Fast validation rules
- Memory-efficient operations
- Cached compilations
- Error handling overhead
- Validation order

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

import re
from typing import Any, Dict, Union, Optional, List
from pathlib import Path

class ValidationUtils:
    def is_valid_email(self, email: str) -> bool:
        """Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            bool: True if valid email format, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
        
    def is_valid_url(self, url: str) -> bool:
        """Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if valid URL format, False otherwise
        """
        pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return bool(re.match(pattern, url))
        
    def is_in_range(self, value: float, min_val: float, max_val: float) -> bool:
        """Check if value is within specified range.
        
        Args:
            value: Value to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            bool: True if value is within range, False otherwise
        """
        return min_val <= value <= max_val
        
    def is_valid_json(self, text: str) -> bool:
        """Validate JSON string.
        
        Args:
            text: JSON string to validate
            
        Returns:
            bool: True if valid JSON, False otherwise
        """
        try:
            import json
            json.loads(text)
            return True
        except ValueError:
            return False
        
    def validate_file_type(self, file_path: Union[str, Path], 
                          allowed_types: Optional[List[str]] = None) -> bool:
        """Validate file type against allowed types.
        
        Args:
            file_path: Path to file
            allowed_types: List of allowed file extensions
            
        Returns:
            bool: True if file type is allowed, False otherwise
        """
        if allowed_types is None:
            allowed_types = ['.txt', '.pdf', '.doc', '.docx']
            
        file_path = Path(file_path)
        return file_path.suffix.lower() in allowed_types
        
    def validate_text_length(self, text: str, 
                           min_length: int = 1, 
                           max_length: Optional[int] = None) -> bool:
        """Validate text length.
        
        Args:
            text: Text to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            
        Returns:
            bool: True if text length is valid, False otherwise
        """
        if len(text) < min_length:
            return False
        if max_length and len(text) > max_length:
            return False
        return True
        
    def validate_metadata(self, metadata: Dict[str, Any], 
                         required_fields: Optional[List[str]] = None) -> bool:
        """Validate metadata contains required fields.
        
        Args:
            metadata: Metadata dictionary to validate
            required_fields: List of required field names
            
        Returns:
            bool: True if all required fields present, False otherwise
        """
        if required_fields is None:
            return True
        return all(field in metadata for field in required_fields) 