"""
Test Vector Search Module
-----------------------

Tests for vector search functionality, including indexing, querying, and optimization.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import faiss
from src.retrieval.vector_search import VectorSearch
from src.embeddings.embedding_generator import EmbeddingGenerator

class TestVectorSearch:
    
    @pytest.fixture
    def vector_search(self, config_manager):
        """Fixture providing a vector search instance."""
        embedding_generator = EmbeddingGenerator(config_manager)
        return VectorSearch(embedding_generator)
    
    @pytest.fixture
    def sample_vectors(self):
        """Fixture providing sample vectors for testing."""
        # Generate 1000 random vectors of dimension 384 (typical embedding size)
        return np.random.randn(1000, 384).astype('float32')
    
    @pytest.fixture
    def sample_metadata(self):
        """Fixture providing sample metadata for vectors."""
        return [
            {
                'id': f'doc_{i}',
                'text': f'Sample document {i}',
                'metadata': {'category': f'category_{i % 5}'}
            }
            for i in range(1000)
        ]
    
    def test_index_creation(self, vector_search, sample_vectors):
        """Test vector index creation and configuration."""
        # Test with different index types
        index_configs = [
            {'index_type': 'flat'},
            {'index_type': 'ivf', 'nlist': 100},
            {'index_type': 'hnsw', 'M': 16, 'ef_construction': 200}
        ]
        
        for config in index_configs:
            index = vector_search.create_index(
                sample_vectors.shape[1],
                **config
            )
            assert isinstance(index, faiss.Index)
            
    def test_vector_addition(self, vector_search, sample_vectors, sample_metadata):
        """Test adding vectors to the index."""
        vector_search.create_index(sample_vectors.shape[1])
        
        # Add vectors in batches
        batch_size = 100
        for i in range(0, len(sample_vectors), batch_size):
            batch_vectors = sample_vectors[i:i + batch_size]
            batch_metadata = sample_metadata[i:i + batch_size]
            vector_search.add_vectors(batch_vectors, batch_metadata)
            
        assert vector_search.index_size == len(sample_vectors)
        
    def test_vector_search(self, vector_search, sample_vectors, sample_metadata):
        """Test vector search functionality."""
        # Initialize and add vectors
        vector_search.create_index(sample_vectors.shape[1])
        vector_search.add_vectors(sample_vectors, sample_metadata)
        
        # Test search with different k values
        query_vector = np.random.randn(384).astype('float32')
        for k in [1, 5, 10]:
            results = vector_search.search(query_vector, k=k)
            assert len(results) == k
            assert all('score' in result for result in results)
            assert all('metadata' in result for result in results)
            
    def test_batch_search(self, vector_search, sample_vectors, sample_metadata):
        """Test batch search operations."""
        vector_search.create_index(sample_vectors.shape[1])
        vector_search.add_vectors(sample_vectors, sample_metadata)
        
        # Create batch of query vectors
        query_vectors = np.random.randn(10, 384).astype('float32')
        results = vector_search.batch_search(query_vectors, k=5)
        
        assert len(results) == len(query_vectors)
        assert all(len(batch_results) == 5 for batch_results in results)
        
    def test_index_persistence(self, vector_search, sample_vectors):
        """Test index saving and loading."""
        vector_search.create_index(sample_vectors.shape[1])
        vector_search.add_vectors(sample_vectors[:100], None)
        
        # Save index
        with tempfile.NamedTemporaryFile(suffix='.index', delete=False) as tf:
            index_path = tf.name
            vector_search.save_index(index_path)
            
        # Create new instance and load index
        new_vector_search = VectorSearch(vector_search.embedding_generator)
        new_vector_search.load_index(index_path)
        
        assert new_vector_search.index_size == 100
        Path(index_path).unlink()
        
    def test_filtered_search(self, vector_search, sample_vectors, sample_metadata):
        """Test search with metadata filtering."""
        vector_search.create_index(sample_vectors.shape[1])
        vector_search.add_vectors(sample_vectors, sample_metadata)
        
        query_vector = np.random.randn(384).astype('float32')
        filter_condition = lambda meta: meta['metadata']['category'] == 'category_0'
        
        results = vector_search.search(
            query_vector,
            k=5,
            filter_condition=filter_condition
        )
        
        assert all(result['metadata']['category'] == 'category_0' 
                  for result in results)
        
    def test_index_optimization(self, vector_search, sample_vectors):
        """Test index optimization methods."""
        # Test IVF index training
        index = vector_search.create_index(
            sample_vectors.shape[1],
            index_type='ivf',
            nlist=100
        )
        
        vector_search.train_index(sample_vectors)
        assert index.is_trained
        
        # Test with different parameters
        index = vector_search.create_index(
            sample_vectors.shape[1],
            index_type='ivf',
            nlist=50,
            nprobe=10
        )
        
        vector_search.train_index(sample_vectors)
        assert index.is_trained
        
    def test_error_handling(self, vector_search):
        """Test error handling in vector search."""
        # Test search before index creation
        query_vector = np.random.randn(384).astype('float32')
        with pytest.raises(ValueError):
            vector_search.search(query_vector)
            
        # Test invalid vector dimension
        vector_search.create_index(384)
        with pytest.raises(ValueError):
            vector_search.add_vectors(np.random.randn(10, 256), None)
            
        # Test invalid k value
        with pytest.raises(ValueError):
            vector_search.search(query_vector, k=0)
