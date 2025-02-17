"""
Test Helper Functions Tests
-------------------------

Tests for the common utility functions used across tests.
"""

import pytest
import numpy as np
import time
from pathlib import Path
from .helpers import (
    wait_for_processing,
    load_test_document,
    compare_embeddings,
    create_test_metadata,
    setup_test_environment
)

class TestHelpers:
    
    def test_wait_for_processing(self):
        """Test wait_for_processing function."""
        # Test successful completion
        def mock_success():
            return True
        assert wait_for_processing(mock_success, timeout=1) == True
        
        # Test timeout
        start_time = time.time()
        def mock_timeout():
            return False
        assert wait_for_processing(mock_timeout, timeout=1) == False
        assert time.time() - start_time >= 1
        
        # Test eventual completion
        counter = [0]
        def mock_eventual():
            counter[0] += 1
            return counter[0] >= 3
        assert wait_for_processing(mock_eventual, timeout=5, interval=0.1) == True
        
    def test_load_test_document(self, tmp_path):
        """Test load_test_document function."""
        # Create test file
        test_content = "Test document content"
        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)
        
        # Test loading
        loaded_content = load_test_document(test_file)
        assert loaded_content == test_content
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            load_test_document(tmp_path / "nonexistent.txt")
            
    def test_compare_embeddings(self):
        """Test compare_embeddings function."""
        # Test identical embeddings
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([1.0, 2.0, 3.0])
        assert compare_embeddings(emb1, emb2)
        
        # Test similar embeddings within tolerance
        emb3 = np.array([1.0000001, 2.0000001, 3.0000001])
        assert compare_embeddings(emb1, emb3, tolerance=1e-6)
        
        # Test different embeddings
        emb4 = np.array([1.1, 2.1, 3.1])
        assert not compare_embeddings(emb1, emb4, tolerance=1e-6)
        
    def test_create_test_metadata(self):
        """Test create_test_metadata function."""
        # Test single metadata creation
        metadata = create_test_metadata()
        assert len(metadata) == 1
        assert metadata[0]['id'] == 'doc_0'
        assert metadata[0]['type'] == 'test'
        
        # Test multiple metadata creation
        metadata = create_test_metadata(num_docs=3)
        assert len(metadata) == 3
        assert all(meta['title'].startswith('Test Document') for meta in metadata)
        
        # Test with base metadata
        base_meta = {'category': 'test_category', 'language': 'en'}
        metadata = create_test_metadata(base_metadata=base_meta)
        assert metadata[0]['category'] == 'test_category'
        assert metadata[0]['language'] == 'en'
        
    def test_setup_test_environment(self):
        """Test setup_test_environment function."""
        try:
            dirs = setup_test_environment()
            
            # Check all directories are created
            assert all(path.exists() for path in dirs.values())
            assert all(path.is_dir() for path in dirs.values())
            
            # Check required directories exist
            assert 'documents' in dirs
            assert 'vectors' in dirs
            assert 'cache' in dirs
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(dirs['root']) 