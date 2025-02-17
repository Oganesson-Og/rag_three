"""
Test Data Handling Security
-------------------------

Tests for secure data handling within the RAG pipeline.
"""

import pytest
from pathlib import Path
from src.document_processing.processor import DocumentProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator

class TestDataHandling:
    
    @pytest.fixture
    def sample_sensitive_document(self, tmp_path):
        """Fixture providing a sample document with sensitive information."""
        doc_path = tmp_path / "sensitive_doc.txt"
        doc_path.write_text("This is a test document containing sensitive information.")
        return doc_path
    
    def test_document_cleanup(self, config_manager, sample_sensitive_document):
        """Test proper cleanup of temporary document data."""
        processor = DocumentProcessor(config_manager)
        
        # Process document
        result = processor.process_document(sample_sensitive_document)
        
        # Check temporary files are cleaned up
        temp_dir = processor.get_temp_directory()
        assert not any(temp_dir.iterdir())
        
    def test_embedding_data_handling(self, config_manager):
        """Test secure handling of data during embedding generation."""
        embedding_generator = EmbeddingGenerator(config_manager)
        
        # Generate embedding
        text = "Test sensitive information"
        embedding = embedding_generator.generate_embedding(text)
        
        # Verify no text data is stored in the embedding generator
        assert not hasattr(embedding_generator, '_last_text')
        assert not hasattr(embedding_generator, '_text_cache')
        
    def test_temp_file_permissions(self, config_manager, sample_sensitive_document):
        """Test temporary file permissions during processing."""
        processor = DocumentProcessor(config_manager)
        
        # Process document and check temp file permissions
        result = processor.process_document(sample_sensitive_document)
        temp_dir = processor.get_temp_directory()
        
        for temp_file in temp_dir.iterdir():
            # Check file is only readable by the process
            assert oct(temp_file.stat().st_mode)[-3:] in ['400', '600']
            
    def test_memory_cleanup(self, config_manager, sample_sensitive_document):
        """Test cleanup of sensitive data from memory."""
        processor = DocumentProcessor(config_manager)
        
        # Process document
        result = processor.process_document(sample_sensitive_document)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Verify no sensitive data remains in memory
        assert not hasattr(processor, '_current_document')
        assert not hasattr(processor, '_document_cache') 