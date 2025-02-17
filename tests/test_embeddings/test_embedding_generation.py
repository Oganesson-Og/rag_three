"""
Test Embedding Generation
----------------------

Tests for embedding model integration and vector generation.
"""

import pytest
import numpy as np
import torch
import time
from pathlib import Path
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.models import EmbeddingConfig, EmbeddingOutput

class TestEmbeddingGeneration:
    
    @pytest.fixture
    def embedding_generator(self, config_manager):
        """Fixture providing an embedding generator instance."""
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            batch_size=32,
            device="cpu"
        )
        return EmbeddingGenerator(config_manager, config)
    
    @pytest.fixture
    def sample_texts(self):
        """Fixture providing sample texts for embedding."""
        return [
            "This is a test sentence for embedding generation.",
            "Machine learning models can process text data.",
            "Vector representations help in similarity search.",
            "Embeddings capture semantic meaning of text."
        ]
        
    def test_basic_embedding(self, embedding_generator):
        """Test basic embedding generation."""
        text = "This is a test sentence."
        embedding = embedding_generator.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # Check dimension
        assert not np.isnan(embedding).any()  # No NaN values
        
    def test_batch_embedding(self, embedding_generator, sample_texts):
        """Test batch embedding generation."""
        embeddings = embedding_generator.generate_embeddings(sample_texts)
        
        assert len(embeddings) == len(sample_texts)
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (384,) for emb in embeddings)
        
    def test_embedding_consistency(self, embedding_generator):
        """Test consistency of generated embeddings."""
        text = "Test sentence for consistency."
        
        embedding1 = embedding_generator.generate_embedding(text)
        embedding2 = embedding_generator.generate_embedding(text)
        
        assert np.allclose(embedding1, embedding2)
        
    def test_embedding_similarity(self, embedding_generator):
        """Test similarity between related and unrelated embeddings."""
        similar_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "AI includes various machine learning techniques."
        ]
        
        different_text = "The weather is sunny today."
        
        similar_embeddings = embedding_generator.generate_embeddings(similar_texts)
        different_embedding = embedding_generator.generate_embedding(different_text)
        
        # Calculate cosine similarities
        similar_sim = embedding_generator.compute_similarity(
            similar_embeddings[0],
            similar_embeddings[1]
        )
        different_sim = embedding_generator.compute_similarity(
            similar_embeddings[0],
            different_embedding
        )
        
        assert similar_sim > different_sim
        
    def test_empty_input_handling(self, embedding_generator):
        """Test handling of empty or invalid inputs."""
        with pytest.raises(ValueError):
            embedding_generator.generate_embedding("")
            
        with pytest.raises(ValueError):
            embedding_generator.generate_embeddings([])
            
    def test_long_text_handling(self, embedding_generator):
        """Test handling of long text inputs."""
        long_text = "test " * 1000  # Very long text
        embedding = embedding_generator.generate_embedding(long_text)
        
        assert embedding.shape == (384,)
        
    def test_special_character_handling(self, embedding_generator):
        """Test handling of special characters."""
        special_text = "Test with @#$%^& special chars and émojis 🌟"
        embedding = embedding_generator.generate_embedding(special_text)
        
        assert not np.isnan(embedding).any()
        
    def test_multilingual_support(self, embedding_generator):
        """Test multilingual embedding support."""
        texts = [
            "English text",
            "Texto en español",
            "Deutscher Text",
            "中文文本"
        ]
        
        embeddings = embedding_generator.generate_embeddings(texts)
        assert len(embeddings) == len(texts)
        assert all(emb.shape == (384,) for emb in embeddings)
        
    def test_embedding_caching(self, embedding_generator):
        """Test embedding caching functionality."""
        text = "Test caching functionality."
        
        # First generation (cache miss)
        start_time = time.time()
        first_embedding = embedding_generator.generate_embedding(text)
        first_time = time.time() - start_time
        
        # Second generation (cache hit)
        start_time = time.time()
        second_embedding = embedding_generator.generate_embedding(text)
        second_time = time.time() - start_time
        
        assert np.array_equal(first_embedding, second_embedding)
        assert second_time < first_time
        
    def test_batch_size_handling(self, embedding_generator):
        """Test handling of different batch sizes."""
        texts = ["Test text"] * 100
        
        # Test with different batch sizes
        for batch_size in [1, 10, 50]:
            embedding_generator.config.batch_size = batch_size
            embeddings = embedding_generator.generate_embeddings(texts)
            assert len(embeddings) == len(texts)
            
    def test_model_loading(self, embedding_generator):
        """Test model loading and initialization."""
        assert embedding_generator.model is not None
        assert embedding_generator.tokenizer is not None
        
        # Test model reloading
        embedding_generator.reload_model()
        assert embedding_generator.model is not None
        
    def test_embedding_metadata(self, embedding_generator):
        """Test embedding metadata generation."""
        text = "Test with metadata."
        result = embedding_generator.generate_embedding_with_metadata(text)
        
        assert isinstance(result, EmbeddingOutput)
        assert result.embedding is not None
        assert result.metadata['timestamp'] is not None
        assert result.metadata['model_name'] == embedding_generator.config.model_name
        
    def test_performance_metrics(self, embedding_generator, sample_texts):
        """Test embedding generation performance metrics."""
        with embedding_generator.track_performance() as metrics:
            embeddings = embedding_generator.generate_embeddings(sample_texts)
            
        assert metrics.total_time > 0
        assert metrics.texts_processed == len(sample_texts)
        assert metrics.avg_time_per_text > 0
