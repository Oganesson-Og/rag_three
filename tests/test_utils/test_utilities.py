"""
Test Utility Functions
-------------------

Tests for helper functions and utility operations.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.utils.text_utils import TextUtils
from src.utils.file_utils import FileUtils
from src.utils.validation_utils import ValidationUtils
from src.utils.conversion_utils import ConversionUtils

class TestUtilities:
    
    @pytest.fixture
    def text_utils(self):
        """Fixture providing text utility instance."""
        return TextUtils()
    
    @pytest.fixture
    def file_utils(self):
        """Fixture providing file utility instance."""
        return FileUtils()
    
    @pytest.fixture
    def sample_text(self):
        """Fixture providing sample text for testing."""
        return """
        This is a sample text with multiple sentences.
        It contains numbers like 123 and special chars @#$.
        Also has some URLs https://example.com and
        email@domain.com addresses.
        """
        
    def test_text_normalization(self, text_utils, sample_text):
        """Test text normalization functions."""
        normalized = text_utils.normalize_text(sample_text)
        
        assert "  " not in normalized  # No double spaces
        assert "\n" not in normalized  # No newlines
        assert normalized.strip() == normalized  # No leading/trailing spaces
        
    def test_text_sanitization(self, text_utils):
        """Test text sanitization functions."""
        unsafe_text = "<script>alert('xss')</script> Hello & World"
        safe_text = text_utils.sanitize_text(unsafe_text)
        
        assert "<script>" not in safe_text
        assert "alert" not in safe_text
        assert "&" not in safe_text
        assert safe_text.strip() == "Hello World"
        
    def test_file_operations(self, file_utils):
        """Test file handling utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "test.txt"
            content = "Test content"
            file_utils.write_text(test_file, content)
            
            # Read and verify
            assert file_utils.read_text(test_file) == content
            assert file_utils.get_file_size(test_file) > 0
            assert file_utils.get_mime_type(test_file) == "text/plain"
            
    def test_json_handling(self, file_utils):
        """Test JSON file handling."""
        data = {
            "name": "test",
            "values": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "test.json"
            
            # Write JSON
            file_utils.write_json(json_file, data)
            
            # Read and verify
            loaded_data = file_utils.read_json(json_file)
            assert loaded_data == data
            
    def test_validation_functions(self):
        """Test data validation utilities."""
        validation = ValidationUtils()
        
        # Email validation
        assert validation.is_valid_email("test@example.com")
        assert not validation.is_valid_email("invalid-email")
        
        # URL validation
        assert validation.is_valid_url("https://example.com")
        assert not validation.is_valid_url("not-a-url")
        
        # Number range validation
        assert validation.is_in_range(5, 0, 10)
        assert not validation.is_in_range(15, 0, 10)
        
    def test_conversion_utilities(self):
        """Test data conversion utilities."""
        conversion = ConversionUtils()
        
        # Size conversions
        assert conversion.bytes_to_mb(1024 * 1024) == 1.0
        assert conversion.mb_to_bytes(1.0) == 1024 * 1024
        
        # Time conversions
        assert conversion.seconds_to_minutes(120) == 2
        assert conversion.minutes_to_seconds(2) == 120
        
    def test_path_handling(self, file_utils):
        """Test path manipulation utilities."""
        path = Path("/test/path/file.txt")
        
        assert file_utils.get_extension(path) == ".txt"
        assert file_utils.get_filename(path) == "file.txt"
        assert file_utils.get_directory(path) == Path("/test/path")
        
    def test_text_extraction(self, text_utils):
        """Test text extraction utilities."""
        text = "Contact us at email@domain.com or visit https://example.com"
        
        emails = text_utils.extract_emails(text)
        urls = text_utils.extract_urls(text)
        
        assert "email@domain.com" in emails
        assert "https://example.com" in urls
        
    def test_date_handling(self):
        """Test date handling utilities."""
        date_utils = ConversionUtils()
        
        # Date string conversion
        date_str = "2024-01-01"
        date_obj = date_utils.str_to_date(date_str)
        assert isinstance(date_obj, datetime)
        assert date_utils.date_to_str(date_obj) == date_str
        
        # Timestamp conversion
        timestamp = date_utils.datetime_to_timestamp(date_obj)
        assert date_utils.timestamp_to_datetime(timestamp) == date_obj
        
    def test_text_metrics(self, text_utils, sample_text):
        """Test text metric utilities."""
        metrics = text_utils.get_text_metrics(sample_text)
        
        assert metrics['char_count'] > 0
        assert metrics['word_count'] > 0
        assert metrics['sentence_count'] > 0
        assert 'avg_word_length' in metrics
        
    def test_file_checksums(self, file_utils):
        """Test file checksum utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            content = "Test content for checksum"
            file_utils.write_text(test_file, content)
            
            md5 = file_utils.get_md5(test_file)
            sha256 = file_utils.get_sha256(test_file)
            
            assert len(md5) == 32  # MD5 is 32 characters
            assert len(sha256) == 64  # SHA256 is 64 characters
            
    def test_compression_utils(self, file_utils):
        """Test compression utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            source_file = Path(temp_dir) / "test.txt"
            content = "Test content" * 1000  # Create larger content
            file_utils.write_text(source_file, content)
            
            # Compress
            compressed_file = file_utils.compress_file(source_file)
            assert compressed_file.exists()
            assert file_utils.get_file_size(compressed_file) < len(content)
            
            # Decompress
            decompressed_file = file_utils.decompress_file(compressed_file)
            assert file_utils.read_text(decompressed_file) == content
            
    def test_string_utils(self, text_utils):
        """Test string manipulation utilities."""
        # Case conversion
        assert text_utils.to_snake_case("TestString") == "test_string"
        assert text_utils.to_camel_case("test_string") == "testString"
        assert text_utils.to_pascal_case("test_string") == "TestString"
        
        # String truncation
        long_text = "This is a very long text that needs truncation"
        truncated = text_utils.truncate(long_text, max_length=20)
        assert len(truncated) <= 20
        assert truncated.endswith("...") 