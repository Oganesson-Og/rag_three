"""
Test File Utilities
----------------
"""

import pytest
import os
from pathlib import Path
from src.utils.file_utils import FileUtils

class TestFileUtils:
    @pytest.fixture
    def file_utils(self):
        return FileUtils()
        
    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test file with content."""
        file_path = tmp_path / "test.txt"
        content = "Test content for file operations"
        file_path.write_text(content)
        return file_path
        
    def test_read_write_text(self, file_utils, tmp_path):
        """Test reading and writing text files."""
        file_path = tmp_path / "test.txt"
        content = "Test content"
        
        file_utils.write_text(file_path, content)
        assert file_utils.read_text(file_path) == content
        
    def test_get_file_size(self, file_utils, test_file):
        """Test file size calculation."""
        size = file_utils.get_file_size(test_file)
        assert size > 0
        assert size == os.path.getsize(test_file)
        
    def test_get_md5(self, file_utils, test_file):
        """Test MD5 hash calculation."""
        md5 = file_utils.get_md5(test_file)
        assert len(md5) == 32
        assert isinstance(md5, str)
        
        # Test consistency
        assert file_utils.get_md5(test_file) == md5
        
    def test_compress_file(self, file_utils, test_file):
        """Test file compression."""
        compressed_path = file_utils.compress_file(test_file)
        assert compressed_path.exists()
        assert compressed_path.suffix == '.gz'
        assert os.path.getsize(compressed_path) > 0 