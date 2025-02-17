"""
Test Hybrid Search Module
-----------------------

Tests for hybrid search combining semantic and keyword-based search approaches.
"""

import pytest
from pathlib import Path
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.models import SearchConfig, SearchResult
import time

class TestHybridSearch:
    
    @pytest.fixture
    def hybrid_search(self, config_manager):
        """Fixture providing a hybrid search instance."""
        return HybridSearch(config_manager)
    
    @pytest.fixture
    def sample_corpus(self):
        """Fixture providing a test document corpus."""
        return [
            {
                'id': '1',
                'text': 'Machine learning is a subset of artificial intelligence.',
                'metadata': {'category': 'AI', 'source': 'textbook'}
            },
            {
                'id': '2',
                'text': 'Deep learning models require significant computational resources.',
                'metadata': {'category': 'AI', 'source': 'article'}
            },
            {
                'id': '3',
                'text': 'Natural Language Processing (NLP) helps computers understand human language.',
                'metadata': {'category': 'AI', 'source': 'blog'}
            },
            {
                'id': '4',
                'text': 'Python is widely used in data science and machine learning.',
                'metadata': {'category': 'Programming', 'source': 'tutorial'}
            },
            {
                'id': '5',
                'text': 'TensorFlow and PyTorch are popular deep learning frameworks.',
                'metadata': {'category': 'Programming', 'source': 'documentation'}
            }
        ]
        
    def test_hybrid_search_basic(self, hybrid_search, sample_corpus):
        """Test basic hybrid search functionality."""
        hybrid_search.index_documents(sample_corpus)
        
        query = "machine learning python"
        results = hybrid_search.search(query, k=2)
        
        assert len(results) == 2
        # Should find documents mentioning both ML and Python
        assert any('python' in result.text.lower() and 
                  'machine learning' in result.text.lower() 
                  for result in results)
        
    def test_weight_balancing(self, hybrid_search, sample_corpus):
        """Test different weight combinations for semantic and keyword search."""
        hybrid_search.index_documents(sample_corpus)
        
        query = "artificial intelligence"
        
        # Semantic-heavy search
        semantic_results = hybrid_search.search(
            query,
            weights={'semantic': 0.8, 'keyword': 0.2},
            k=2
        )
        
        # Keyword-heavy search
        keyword_results = hybrid_search.search(
            query,
            weights={'semantic': 0.2, 'keyword': 0.8},
            k=2
        )
        
        assert semantic_results != keyword_results
        
    def test_exact_matches(self, hybrid_search, sample_corpus):
        """Test handling of exact phrase matches."""
        hybrid_search.index_documents(sample_corpus)
        
        # Search for exact phrase
        query = '"deep learning"'
        results = hybrid_search.search(query, k=2)
        
        assert all('deep learning' in result.text.lower() 
                  for result in results)
        
    def test_filtered_search(self, hybrid_search, sample_corpus):
        """Test search with metadata filters."""
        hybrid_search.index_documents(sample_corpus)
        
        query = "learning"
        filters = {'category': 'Programming'}
        results = hybrid_search.search(query, filters=filters, k=2)
        
        assert all(result.metadata['category'] == 'Programming' 
                  for result in results)
        
    def test_boosted_fields(self, hybrid_search, sample_corpus):
        """Test search with field boosting."""
        hybrid_search.index_documents(sample_corpus)
        
        config = SearchConfig(
            field_weights={'title': 2.0, 'text': 1.0}
        )
        
        results = hybrid_search.search(
            "machine learning",
            search_config=config,
            k=3
        )
        
        assert isinstance(results[0], SearchResult)
        
    def test_result_scoring(self, hybrid_search, sample_corpus):
        """Test result scoring mechanism."""
        hybrid_search.index_documents(sample_corpus)
        
        query = "deep learning frameworks"
        results = hybrid_search.search(query, k=3)
        
        # Check score properties
        assert all(hasattr(result, 'score') for result in results)
        assert all(0 <= result.score <= 1 for result in results)
        
        # Verify score ordering
        scores = [result.score for result in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
    def test_query_preprocessing(self, hybrid_search, sample_corpus):
        """Test query preprocessing functionality."""
        hybrid_search.index_documents(sample_corpus)
        
        # Test different query forms
        queries = [
            "MACHINE LEARNING",  # uppercase
            "machine-learning",  # hyphenated
            "machine  learning"  # extra spaces
        ]
        
        base_results = hybrid_search.search("machine learning", k=1)
        
        for query in queries:
            results = hybrid_search.search(query, k=1)
            assert results[0].id == base_results[0].id
            
    def test_incremental_updates(self, hybrid_search, sample_corpus):
        """Test index updates during search."""
        # Initial indexing
        hybrid_search.index_documents(sample_corpus[:3])
        
        # Initial search
        query = "machine learning"
        initial_results = hybrid_search.search(query, k=2)
        
        # Add more documents
        hybrid_search.add_documents(sample_corpus[3:])
        
        # Search after update
        updated_results = hybrid_search.search(query, k=2)
        
        assert len(updated_results) == 2
        assert initial_results != updated_results
        
    def test_performance_optimization(self, hybrid_search, sample_corpus):
        """Test search performance optimization."""
        hybrid_search.index_documents(sample_corpus)
        
        import time
        
        # Measure search time
        query = "machine learning python"
        start_time = time.time()
        results = hybrid_search.search(query, k=3)
        search_time = time.time() - start_time
        
        assert search_time < 1.0  # Should be fast
        
    def test_error_handling(self, hybrid_search, sample_corpus):
        """Test error handling in hybrid search."""
        hybrid_search.index_documents(sample_corpus)
        
        # Test invalid weights
        with pytest.raises(ValueError):
            hybrid_search.search(
                "test",
                weights={'semantic': 1.5, 'keyword': 0.5}
            )
            
        # Test invalid k value
        with pytest.raises(ValueError):
            hybrid_search.search("test", k=-1)
            
        # Test empty query
        with pytest.raises(ValueError):
            hybrid_search.search("")
            
    def test_relevance_scoring(self, hybrid_search, sample_corpus):
        """Test relevance scoring with combined signals."""
        hybrid_search.index_documents(sample_corpus)
        
        query = "deep learning frameworks python"
        results = hybrid_search.search(query, k=len(sample_corpus))
        
        # Document with both deep learning and python should rank higher
        relevant_doc = next(r for r in results 
                          if 'python' in r.text.lower() and 
                          'deep learning' in r.text.lower())
        
        assert results.index(relevant_doc) < len(results) // 2
        
    def test_cache_management(self, hybrid_search, sample_corpus):
        """Test search cache management."""
        hybrid_search.index_documents(sample_corpus)
        
        # First search (cache miss)
        start_time = time.time()
        first_results = hybrid_search.search("machine learning", k=2)
        first_search_time = time.time() - start_time
        
        # Second search (cache hit)
        start_time = time.time()
        second_results = hybrid_search.search("machine learning", k=2)
        second_search_time = time.time() - start_time
        
        assert second_search_time < first_search_time
        assert first_results == second_results
