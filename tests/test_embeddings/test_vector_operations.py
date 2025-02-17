"""
Test Vector Operations Module
--------------------------

Tests for vector operations, similarity calculations, and vector manipulations.
"""

import pytest
import numpy as np
from src.embeddings.vector_store import VectorStore
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_ops import VectorOperations
from src.embeddings.models import SimilarityMetric

@pytest.fixture
def config_manager():
    """Fixture providing configuration for vector store."""
    from src.config.config_manager import ConfigManager
    return ConfigManager(
        config_path="config/test_config.yaml",
        embedding_dim=384,
        index_path="test_indices"
    )

class TestVectorOperations:
    
    @pytest.fixture
    def vector_store(self, config_manager):
        """Fixture providing a vector store instance."""
        return VectorStore(config_manager)
    
    @pytest.fixture
    def vector_ops(self, config_manager):
        """Fixture providing a vector operations instance."""
        return VectorOperations(config_manager)
    
    @pytest.fixture
    def sample_vectors(self):
        """Fixture providing sample vectors for testing."""
        # Generate normalized random vectors
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(100, 384))  # 100 vectors of dimension 384
        # Normalize vectors
        return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    @pytest.fixture
    def sample_metadata(self):
        """Fixture providing sample metadata for vectors."""
        return [
            {
                'id': f'doc_{i}',
                'text': f'Sample text {i}',
                'source': 'test'
            }
            for i in range(100)
        ]
    
    def test_vector_similarity(self, vector_store, sample_vectors):
        """Test vector similarity calculations."""
        query_vector = sample_vectors[0]
        
        # Test cosine similarity
        similarities = vector_store.calculate_similarity(
            query_vector,
            sample_vectors[1:],
            metric='cosine'
        )
        assert isinstance(similarities, np.ndarray)
        assert len(similarities) == len(sample_vectors) - 1
        assert all(0 <= sim <= 1 for sim in similarities)
        
        # Test euclidean distance
        distances = vector_store.calculate_similarity(
            query_vector,
            sample_vectors[1:],
            metric='euclidean'
        )
        assert isinstance(distances, np.ndarray)
        assert len(distances) == len(sample_vectors) - 1
        assert all(dist >= 0 for dist in distances)
        
    def test_nearest_neighbors(self, vector_store, sample_vectors, sample_metadata):
        """Test nearest neighbor search."""
        # Add vectors to store
        vector_store.add_vectors(sample_vectors, sample_metadata)
        
        query_vector = sample_vectors[0]
        k = 5
        
        results = vector_store.search_nearest(
            query_vector,
            k=k,
            return_metadata=True
        )
        
        assert len(results) == k
        assert all('score' in result for result in results)
        assert all('metadata' in result for result in results)
        
    def test_batch_search(self, vector_store, sample_vectors, sample_metadata):
        """Test batch nearest neighbor search."""
        vector_store.add_vectors(sample_vectors, sample_metadata)
        
        query_vectors = sample_vectors[:5]
        k = 3
        
        batch_results = vector_store.batch_search_nearest(
            query_vectors,
            k=k
        )
        
        assert len(batch_results) == len(query_vectors)
        assert all(len(results) == k for results in batch_results)
        
    def test_vector_indexing(self, vector_store, sample_vectors, sample_metadata):
        """Test vector indexing functionality."""
        # Test index building
        vector_store.build_index(sample_vectors, sample_metadata)
        
        assert vector_store.index is not None
        assert vector_store.index_size == len(sample_vectors)
        
        # Test index saving and loading
        vector_store.save_index('test_index')
        vector_store.load_index('test_index')
        
        # Clean up
        import os
        os.remove('test_index')
        
    def test_vector_filtering(self, vector_store, sample_vectors, sample_metadata):
        """Test vector filtering operations."""
        vector_store.add_vectors(sample_vectors, sample_metadata)
        
        # Filter by metadata
        filter_condition = lambda meta: meta['source'] == 'test'
        filtered_results = vector_store.search_nearest(
            sample_vectors[0],
            k=5,
            filter_condition=filter_condition
        )
        
        assert all(result['metadata']['source'] == 'test' 
                  for result in filtered_results)
        
    def test_vector_update(self, vector_store, sample_vectors, sample_metadata):
        """Test vector update operations."""
        # Add initial vectors
        vector_store.add_vectors(sample_vectors[:50], sample_metadata[:50])
        
        # Update vectors
        updated_vectors = np.random.randn(10, 384)
        updated_metadata = [
            {'id': f'doc_{i}', 'text': f'Updated text {i}'}
            for i in range(10)
        ]
        
        vector_store.update_vectors(
            list(range(10)),  # indices to update
            updated_vectors,
            updated_metadata
        )
        
        # Verify updates
        results = vector_store.search_nearest(updated_vectors[0], k=1)
        assert results[0]['metadata']['text'].startswith('Updated')
        
    def test_vector_deletion(self, vector_store, sample_vectors, sample_metadata):
        """Test vector deletion operations."""
        vector_store.add_vectors(sample_vectors, sample_metadata)
        
        # Delete some vectors
        indices_to_delete = [0, 1, 2]
        vector_store.delete_vectors(indices_to_delete)
        
        assert vector_store.index_size == len(sample_vectors) - len(indices_to_delete)
        
    def test_vector_persistence(self, vector_store, sample_vectors, sample_metadata):
        """Test vector persistence operations."""
        vector_store.add_vectors(sample_vectors, sample_metadata)
        
        # Save to disk
        vector_store.save('test_vectors.pkl')
        
        # Load new instance
        new_store = VectorStore(config_manager)
        new_store.load('test_vectors.pkl')
        
        # Verify loaded data
        assert new_store.index_size == vector_store.index_size
        
        # Clean up
        import os
        os.remove('test_vectors.pkl')

    def test_cosine_similarity(self, vector_ops):
        """Test cosine similarity calculation."""
        # Create two test vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([1, 0, 0])  # Same as v1
        
        # Test orthogonal vectors
        similarity = vector_ops.cosine_similarity(v1, v2)
        assert np.isclose(similarity, 0.0)
        
        # Test identical vectors
        similarity = vector_ops.cosine_similarity(v1, v3)
        assert np.isclose(similarity, 1.0)
        
        # Test with normalized vectors
        v4 = np.array([1, 1, 0]) / np.sqrt(2)
        similarity = vector_ops.cosine_similarity(v1, v4)
        assert np.isclose(similarity, 1/np.sqrt(2))
        
    def test_batch_similarity(self, vector_ops, sample_vectors):
        """Test batch similarity calculations."""
        query = sample_vectors[0]  # Use first vector as query
        
        # Calculate similarities with all vectors
        similarities = vector_ops.batch_similarity(query, sample_vectors)
        
        assert len(similarities) == len(sample_vectors)
        assert np.isclose(similarities[0], 1.0)  # Self-similarity
        assert all(0 <= s <= 1 for s in similarities)  # Valid similarity range
        
    def test_vector_normalization(self, vector_ops):
        """Test vector normalization."""
        # Test single vector
        vector = np.array([3, 4])
        normalized = vector_ops.normalize(vector)
        assert np.isclose(np.linalg.norm(normalized), 1.0)
        
        # Test batch of vectors
        vectors = np.array([[1, 1], [3, 4], [6, 8]])
        normalized = vector_ops.normalize(vectors)
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)
        
    def test_top_k_similar(self, vector_ops, sample_vectors):
        """Test top-k similar vectors retrieval."""
        query = sample_vectors[0]
        k = 5
        
        # Get top-k similar vectors
        indices, scores = vector_ops.top_k_similar(query, sample_vectors, k)
        
        assert len(indices) == k
        assert len(scores) == k
        assert np.all(scores[:-1] >= scores[1:])  # Check scores are sorted
        assert indices[0] == 0  # Most similar should be the query itself
        
    def test_similarity_metrics(self, vector_ops):
        """Test different similarity metrics."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        
        metrics = {
            SimilarityMetric.COSINE: 0.0,
            SimilarityMetric.DOT_PRODUCT: 0.0,
            SimilarityMetric.EUCLIDEAN: np.sqrt(2)
        }
        
        for metric, expected in metrics.items():
            similarity = vector_ops.calculate_similarity(v1, v2, metric)
            assert np.isclose(similarity, expected)
            
    def test_vector_aggregation(self, vector_ops, sample_vectors):
        """Test vector aggregation methods."""
        vectors = sample_vectors[:10]  # Use first 10 vectors
        
        # Test mean aggregation
        mean_vector = vector_ops.aggregate_vectors(vectors, method='mean')
        assert mean_vector.shape == (384,)
        assert np.isclose(np.linalg.norm(mean_vector), 1.0)  # Should be normalized
        
        # Test weighted aggregation
        weights = np.array([0.1] * 10)
        weighted_vector = vector_ops.aggregate_vectors(vectors, weights=weights)
        assert weighted_vector.shape == (384,)
        
    def test_vector_clustering(self, vector_ops, sample_vectors):
        """Test vector clustering functionality."""
        n_clusters = 5
        clusters = vector_ops.cluster_vectors(sample_vectors, n_clusters)
        
        assert len(np.unique(clusters)) == n_clusters
        assert len(clusters) == len(sample_vectors)
        
    def test_dimension_reduction(self, vector_ops, sample_vectors):
        """Test dimension reduction for visualization."""
        reduced_dims = 2
        reduced_vectors = vector_ops.reduce_dimensions(sample_vectors, reduced_dims)
        
        assert reduced_vectors.shape == (len(sample_vectors), reduced_dims)
        
    def test_error_handling(self, vector_ops):
        """Test error handling in vector operations."""
        # Test mismatched dimensions
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0])
        
        with pytest.raises(ValueError):
            vector_ops.cosine_similarity(v1, v2)
            
        # Test invalid input types
        with pytest.raises(TypeError):
            vector_ops.normalize("not a vector")
            
        # Test empty input
        with pytest.raises(ValueError):
            vector_ops.normalize(np.array([]))
            
    def test_batch_processing_efficiency(self, vector_ops, sample_vectors):
        """Test efficiency of batch processing."""
        query = sample_vectors[0]
        
        import time
        
        # Time batch processing
        start_time = time.time()
        batch_similarities = vector_ops.batch_similarity(query, sample_vectors)
        batch_time = time.time() - start_time
        
        # Time sequential processing
        start_time = time.time()
        sequential_similarities = [
            vector_ops.cosine_similarity(query, v)
            for v in sample_vectors
        ]
        sequential_time = time.time() - start_time
        
        assert batch_time < sequential_time
        assert np.allclose(batch_similarities, sequential_similarities)
        
    def test_numerical_stability(self, vector_ops):
        """Test numerical stability with very small/large values."""
        # Test with very small vectors
        small_v1 = np.array([1e-10, 2e-10, 3e-10])
        small_v2 = np.array([2e-10, 4e-10, 6e-10])
        
        similarity = vector_ops.cosine_similarity(small_v1, small_v2)
        assert not np.isnan(similarity)
        assert np.isclose(similarity, 1.0)
        
        # Test with very large vectors
        large_v1 = np.array([1e10, 2e10, 3e10])
        large_v2 = np.array([2e10, 4e10, 6e10])
        
        similarity = vector_ops.cosine_similarity(large_v1, large_v2)
        assert not np.isnan(similarity)
        assert np.isclose(similarity, 1.0)
