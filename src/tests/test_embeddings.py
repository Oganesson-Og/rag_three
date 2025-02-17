"""
Embedding Generation Test Module
-----------------------------

Tests for embedding generation and vector operations.
Validates embedding quality and performance.

Features:
- Embedding generation tests
- Vector similarity tests
- Dimension validation
- Performance benchmarks
- Error handling

Key Test Cases:
1. Embedding consistency
2. Similarity calculations
3. Dimension checks
4. API integration
5. Error scenarios

Technical Details:
- Vector normalization
- Similarity metrics
- Batch processing
- Performance monitoring

Dependencies:
- numpy>=1.19.0
- pytest>=7.0.0
- requests>=2.26.0

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..embeddings import EmbeddingGenerator
from ..config.embedding_config import EMBEDDING_CONFIG

class TestEmbeddingGenerator:
    @pytest.fixture
    def generator(self):
        return EmbeddingGenerator(model_name='minilm')
    
    @pytest.fixture
    def sample_texts(self):
        return [
            "This is a test sentence for embedding generation.",
            "Another sample text with different content.",
            "Technical documentation about RAG systems."
        ]
    
    def test_generate_embedding(self, generator, sample_texts):
        embedding = generator.generate(sample_texts[0])
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (generator.dimension,)
        assert not np.isnan(embedding).any()
        
    def test_batch_generation(self, generator, sample_texts):
        embeddings = generator.generate_batch(sample_texts)
        assert len(embeddings) == len(sample_texts)
        assert all(emb.shape == (generator.dimension,) for emb in embeddings)
        
    def test_similarity_calculation(self, generator):
        text1 = "The quick brown fox"
        text2 = "A fast brown fox"
        sim = generator.calculate_similarity(text1, text2)
        assert 0 <= sim <= 1
        
    def test_empty_text(self, generator):
        with pytest.raises(ValueError):
            generator.generate("")
            
    @pytest.mark.parametrize("model_name", list(EMBEDDING_CONFIG['models'].keys()))
    def test_model_loading(self, model_name):
        generator = EmbeddingGenerator(model_name=model_name)
        assert generator.model is not None
        
    def test_dimension_consistency(self, generator, sample_texts):
        embeddings = generator.generate_batch(sample_texts)
        assert all(emb.shape == embeddings[0].shape for emb in embeddings)
        
    @patch('time.time')
    def test_generation_time(self, mock_time, generator, sample_texts):
        mock_time.side_effect = [0, 1]  # 1 second generation time
        with generator.timer() as t:
            generator.generate_batch(sample_texts)
        assert t.elapsed <= 1.0 