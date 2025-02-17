"""
Test Utilities Integration
-----------------------

Tests for interactions between different utility modules.
"""

import pytest
from pathlib import Path
from datetime import datetime
import json
from src.utils.text_utils import TextUtils
from src.utils.file_utils import FileUtils
from src.utils.validation_utils import ValidationUtils
from src.utils.conversion_utils import ConversionUtils

class TestUtilsIntegration:
    @pytest.fixture
    def utils_bundle(self):
        """Fixture providing all utility instances."""
        return {
            'text': TextUtils(),
            'file': FileUtils(),
            'validation': ValidationUtils(),
            'conversion': ConversionUtils()
        }
        
    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create a test directory with sample files."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        return data_dir
        
    def test_text_processing_pipeline(self, utils_bundle, test_data_dir):
        """Test complete text processing pipeline."""
        text = """<p>This is a sample text with HTML</p>
        It has multiple   spaces and special chars: @#$
        We'll process it through multiple steps."""
        
        text_utils = utils_bundle['text']
        file_utils = utils_bundle['file']
        validation_utils = utils_bundle['validation']
        
        # Process text
        normalized = text_utils.normalize_text(text)
        sanitized = text_utils.sanitize_text(normalized)
        
        # Validate
        assert validation_utils.validate_text_length(sanitized, min_length=10)
        
        # Save and read back
        file_path = test_data_dir / "processed_text.txt"
        file_utils.write_text(file_path, sanitized)
        
        # Validate file
        assert validation_utils.validate_file_type(file_path)
        read_back = file_utils.read_text(file_path)
        assert read_back == sanitized
        
    def test_document_metadata_pipeline(self, utils_bundle, test_data_dir):
        """Test document metadata processing pipeline."""
        conversion_utils = utils_bundle['conversion']
        file_utils = utils_bundle['file']
        validation_utils = utils_bundle['validation']
        
        # Create metadata
        metadata = {
            'title': 'Test Document',
            'created_at': conversion_utils.date_to_str(datetime.now()),
            'file_info': {
                'size': 0,
                'md5': ''
            }
        }
        
        # Create test document
        doc_path = test_data_dir / "test_doc.txt"
        content = "Test content for metadata processing"
        file_utils.write_text(doc_path, content)
        
        # Update metadata with file info
        metadata['file_info']['size'] = file_utils.get_file_size(doc_path)
        metadata['file_info']['md5'] = file_utils.get_md5(doc_path)
        
        # Validate metadata
        assert validation_utils.validate_metadata(
            metadata,
            required_fields=['title', 'created_at']
        )
        
        # Save metadata
        meta_path = test_data_dir / "metadata.json"
        file_utils.write_text(
            meta_path,
            conversion_utils.to_json(metadata, pretty=True)
        )
        
    def test_data_conversion_pipeline(self, utils_bundle, test_data_dir):
        """Test data conversion and validation pipeline."""
        text_utils = utils_bundle['text']
        conversion_utils = utils_bundle['conversion']
        validation_utils = utils_bundle['validation']
        
        # Process text with metrics
        text = "Sample text for metrics"
        metrics = text_utils.get_text_metrics(text)
        
        # Add timestamp to metrics
        metrics['timestamp'] = conversion_utils.datetime_to_timestamp(
            datetime.now()
        )
        
        # Convert to JSON and validate
        json_data = conversion_utils.to_json(metrics)
        assert validation_utils.validate_metadata(
            json.loads(json_data),
            required_fields=['char_count', 'word_count']
        )
        
    def test_file_processing_pipeline(self, utils_bundle, test_data_dir):
        """Test complete file processing pipeline."""
        file_utils = utils_bundle['file']
        text_utils = utils_bundle['text']
        validation_utils = utils_bundle['validation']
        
        # Create and process file
        original_path = test_data_dir / "original.txt"
        content = "Original content with some   extra   spaces"
        file_utils.write_text(original_path, content)
        
        # Process content
        processed_content = text_utils.normalize_text(
            file_utils.read_text(original_path)
        )
        
        # Save processed content
        processed_path = test_data_dir / "processed.txt"
        file_utils.write_text(processed_path, processed_content)
        
        # Compress file
        compressed_path = file_utils.compress_file(processed_path)
        
        # Validate results
        assert validation_utils.validate_file_type(processed_path)
        assert validation_utils.validate_file_type(
            compressed_path,
            allowed_types=['.gz']
        )
        assert file_utils.get_file_size(compressed_path) > 0
        
    def test_error_handling_integration(self, utils_bundle, test_data_dir):
        """Test error handling across utilities."""
        file_utils = utils_bundle['file']
        validation_utils = utils_bundle['validation']
        conversion_utils = utils_bundle['conversion']
        
        # Test with non-existent file
        non_existent = test_data_dir / "non_existent.txt"
        with pytest.raises(FileNotFoundError):
            file_utils.read_text(non_existent)
            
        # Test with invalid date string
        with pytest.raises(ValueError):
            conversion_utils.str_to_date("invalid-date")
            
        # Test with invalid file type
        assert not validation_utils.validate_file_type(
            "test.invalid",
            allowed_types=['.txt']
        ) 