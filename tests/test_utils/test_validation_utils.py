"""
Test Validation Utilities
----------------------
"""

import pytest
from pathlib import Path
from src.utils.validation_utils import ValidationUtils

class TestValidationUtils:
    @pytest.fixture
    def validation_utils(self):
        return ValidationUtils()
        
    @pytest.fixture
    def temp_files(self, tmp_path):
        """Create temporary test files."""
        files = {
            'text': tmp_path / "test.txt",
            'pdf': tmp_path / "test.pdf",
            'invalid': tmp_path / "test.xyz"
        }
        for f in files.values():
            f.touch()
        return files
        
    def test_validate_file_type(self, validation_utils, temp_files):
        """Test file type validation."""
        assert validation_utils.validate_file_type(temp_files['text'])
        assert validation_utils.validate_file_type(temp_files['pdf'])
        assert not validation_utils.validate_file_type(temp_files['invalid'])
        
        # Test with custom allowed types
        assert validation_utils.validate_file_type(
            temp_files['invalid'],
            allowed_types=['.xyz']
        )
        
    def test_validate_text_length(self, validation_utils):
        """Test text length validation."""
        assert validation_utils.validate_text_length("Test", min_length=1)
        assert not validation_utils.validate_text_length("", min_length=1)
        assert validation_utils.validate_text_length(
            "Test",
            min_length=1,
            max_length=10
        )
        assert not validation_utils.validate_text_length(
            "Long text",
            max_length=5
        )
        
    def test_validate_metadata(self, validation_utils):
        """Test metadata validation."""
        metadata = {
            'title': 'Test',
            'author': 'John Doe'
        }
        
        assert validation_utils.validate_metadata(
            metadata,
            required_fields=['title']
        )
        assert not validation_utils.validate_metadata(
            metadata,
            required_fields=['date']
        )
        assert validation_utils.validate_metadata(metadata)  # No required fields 