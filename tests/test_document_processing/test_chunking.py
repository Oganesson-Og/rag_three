"""
Test Document Chunking Module
---------------------------

Tests for document chunking strategies and functionality.
"""

import pytest
from src.document_processing.processor import DocumentProcessor
from src.document_processing.preprocessor import DocumentPreprocessor
from tests.test_utils.helpers import wait_for_processing

class TestDocumentChunking:
    
    @pytest.fixture
    def long_text(self):
        """Fixture providing a long text for chunking tests."""
        return """
        Chapter 1: Introduction to Mathematics
        Mathematics is the study of numbers, quantities, and shapes. It is fundamental
        to understanding the world around us and forms the basis of many sciences.
        
        Chapter 2: Basic Algebra
        Algebra is a branch of mathematics dealing with symbols and the rules for
        manipulating these symbols. In basic algebra, we use letters to represent numbers.
        
        Chapter 3: Geometry
        Geometry is the study of shapes, sizes, and positions of things. It includes
        points, lines, angles, surfaces, and solids.
        """
    
    def test_fixed_size_chunking(self, long_text):
        """Test chunking with fixed size strategy."""
        preprocessor = DocumentPreprocessor()
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        chunk_size = 100
        overlap = 20
        chunks = doc_processor.chunk_text(
            long_text,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy='fixed_size'
        )
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Check that chunks are roughly the specified size
        assert all(len(chunk) <= chunk_size for chunk in chunks)
        
    def test_semantic_chunking(self, long_text):
        """Test chunking based on semantic boundaries."""
        preprocessor = DocumentPreprocessor()
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        chunks = doc_processor.chunk_text(
            long_text,
            strategy='semantic'
        )
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Check that chunks maintain chapter boundaries
        assert any('Chapter' in chunk for chunk in chunks)
        
    def test_sentence_chunking(self, long_text):
        """Test chunking based on sentence boundaries."""
        preprocessor = DocumentPreprocessor()
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        chunks = doc_processor.chunk_text(
            long_text,
            strategy='sentence'
        )
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Verify sentences aren't split mid-sentence
        assert all(chunk.strip().endswith(('.', '!', '?')) for chunk in chunks)
        
    def test_overlap_handling(self, long_text):
        """Test chunk overlap functionality."""
        preprocessor = DocumentPreprocessor()
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        overlap = 50
        chunks = doc_processor.chunk_text(
            long_text,
            chunk_size=200,
            overlap=overlap,
            strategy='fixed_size'
        )
        
        # Check for overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][-overlap:]
            chunk2_start = chunks[i + 1][:overlap]
            assert chunk1_end.strip() == chunk2_start.strip()
            
    def test_chunk_metadata(self, long_text):
        """Test metadata preservation in chunks."""
        preprocessor = DocumentPreprocessor()
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        chunks_with_metadata = doc_processor.chunk_text(
            long_text,
            strategy='semantic',
            return_metadata=True
        )
        
        assert isinstance(chunks_with_metadata, list)
        for chunk in chunks_with_metadata:
            assert isinstance(chunk, dict)
            assert 'text' in chunk
            assert 'metadata' in chunk
            assert 'position' in chunk['metadata']
            
    def test_empty_input_handling(self):
        """Test chunking with empty or invalid input."""
        preprocessor = DocumentPreprocessor()
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        # Test with empty string
        assert len(doc_processor.chunk_text("")) == 0
        
        # Test with whitespace
        assert len(doc_processor.chunk_text("   ")) == 0
        
        # Test with None
        with pytest.raises(ValueError):
            doc_processor.chunk_text(None)
            
    def test_custom_chunking_strategy(self, long_text):
        """Test custom chunking strategy."""
        preprocessor = DocumentPreprocessor()
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        def custom_chunk_strategy(text, **kwargs):
            """Simple custom strategy that splits on 'Chapter'"""
            return [chunk.strip() for chunk in text.split('Chapter') if chunk.strip()]
        
        chunks = doc_processor.chunk_text(
            long_text,
            strategy=custom_chunk_strategy
        )
        
        assert isinstance(chunks, list)
        assert len(chunks) == 3  # Should have 3 chapters
