"""
Test Text Chunking Module
----------------------

Tests for text chunking, segmentation, and preprocessing functionality.
"""

import pytest
from src.chunking.base import TextChunker
from src.document_processing.models import ChunkingConfig, Chunk

class TestTextChunking:
    
    @pytest.fixture
    def text_chunker(self, config_manager):
        """Fixture providing a text chunker instance."""
        return TextChunker(config_manager)
    
    @pytest.fixture
    def sample_text(self):
        """Fixture providing sample text for chunking."""
        return """
        Machine learning is a field of artificial intelligence. It focuses on developing
        systems that can learn from data. These systems can improve their performance
        over time without being explicitly programmed.
        
        Deep learning is a subset of machine learning. It uses neural networks with many
        layers. These networks can automatically learn representations from data.
        
        Natural Language Processing (NLP) is another important area. It helps computers
        understand and work with human language. Many modern applications use NLP
        techniques.
        """
        
    def test_basic_chunking(self, text_chunker, sample_text):
        """Test basic text chunking functionality."""
        chunks = text_chunker.chunk_text(
            text=sample_text,
            chunk_size=100,
            overlap=20
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.text) <= 100 for chunk in chunks)
        
    def test_overlap_handling(self, text_chunker):
        """Test chunk overlap functionality."""
        text = "This is a test sentence. " * 10
        chunks = text_chunker.chunk_text(
            text=text,
            chunk_size=50,
            overlap=25
        )
        
        # Check consecutive chunks for overlap
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i].text
            chunk2 = chunks[i + 1].text
            
            # Should find some common text between consecutive chunks
            assert any(sent in chunk2 for sent in chunk1.split('.'))
            
    def test_sentence_boundary_respect(self, text_chunker):
        """Test that chunking respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = text_chunker.chunk_text(
            text=text,
            chunk_size=30,
            respect_sentences=True
        )
        
        # Each chunk should end with a sentence boundary
        assert all(chunk.text.strip().endswith('.') for chunk in chunks)
        
    def test_chunk_metadata(self, text_chunker, sample_text):
        """Test chunk metadata generation."""
        chunks = text_chunker.chunk_text(sample_text, chunk_size=100)
        
        for chunk in chunks:
            assert hasattr(chunk, 'metadata')
            assert 'start_char' in chunk.metadata
            assert 'end_char' in chunk.metadata
            assert 'chunk_id' in chunk.metadata
            
    def test_different_chunk_sizes(self, text_chunker, sample_text):
        """Test chunking with different size configurations."""
        sizes = [50, 100, 200]
        
        for size in sizes:
            chunks = text_chunker.chunk_text(
                text=sample_text,
                chunk_size=size
            )
            assert all(len(chunk.text) <= size for chunk in chunks)
            
    def test_minimum_chunk_size(self, text_chunker):
        """Test minimum chunk size enforcement."""
        text = "Short text."
        min_size = 20
        
        config = ChunkingConfig(
            min_chunk_size=min_size,
            chunk_size=100
        )
        
        chunks = text_chunker.chunk_text(text, config=config)
        assert len(chunks) == 1  # Should be kept as single chunk
        
    def test_custom_separators(self, text_chunker):
        """Test chunking with custom separators."""
        text = "Item1|Item2|Item3|Item4|Item5"
        
        chunks = text_chunker.chunk_text(
            text=text,
            chunk_size=10,
            separators=['|']
        )
        
        assert all('|' not in chunk.text for chunk in chunks[:-1])
        
    def test_chunk_cleaning(self, text_chunker):
        """Test chunk text cleaning."""
        text = "Text with  multiple   spaces\n\nand newlines\t\tand tabs."
        chunks = text_chunker.chunk_text(text, chunk_size=100)
        
        cleaned_text = chunks[0].text
        assert "  " not in cleaned_text  # No multiple spaces
        assert "\t" not in cleaned_text  # No tabs
        assert "\n\n" not in cleaned_text  # No multiple newlines
        
    def test_empty_input_handling(self, text_chunker):
        """Test handling of empty or whitespace-only input."""
        empty_inputs = ["", " ", "\n", "\t"]
        
        for text in empty_inputs:
            with pytest.raises(ValueError):
                text_chunker.chunk_text(text)
                
    def test_chunk_consistency(self, text_chunker, sample_text):
        """Test consistency of chunking results."""
        # Multiple runs should produce same results
        chunks1 = text_chunker.chunk_text(sample_text, chunk_size=100)
        chunks2 = text_chunker.chunk_text(sample_text, chunk_size=100)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.text == c2.text
            assert c1.metadata == c2.metadata
            
    def test_special_characters(self, text_chunker):
        """Test handling of special characters."""
        text = "Text with special chars: @#$%^&*()_+{}[]|\\:;"
        chunks = text_chunker.chunk_text(text, chunk_size=100)
        
        assert chunks[0].text == text
        
    def test_unicode_handling(self, text_chunker):
        """Test handling of Unicode characters."""
        text = "Unicode text with émojis 🌟 and accents éàüñ"
        chunks = text_chunker.chunk_text(text, chunk_size=100)
        
        assert "émojis" in chunks[0].text
        assert "🌟" in chunks[0].text
        assert "éàüñ" in chunks[0].text
        
    def test_chunk_numbering(self, text_chunker, sample_text):
        """Test chunk sequential numbering."""
        chunks = text_chunker.chunk_text(sample_text, chunk_size=50)
        
        chunk_ids = [chunk.metadata['chunk_id'] for chunk in chunks]
        assert chunk_ids == list(range(len(chunks)))
        
    def test_performance_large_text(self, text_chunker):
        """Test chunking performance with large text."""
        large_text = "Sample sentence. " * 1000
        
        import time
        start_time = time.time()
        
        chunks = text_chunker.chunk_text(large_text, chunk_size=100)
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0  # Should process quickly
        assert len(chunks) > 0 