"""
Test Semantic Search Module
------------------------

Tests for semantic search functionality, including ranking and filtering.
"""

import pytest
import numpy as np
from pathlib import Path
from src.retrieval.semantic_search import SemanticSearch
from src.embeddings.embedding_generator import EmbeddingGenerator

class TestSemanticSearch:
    
    @pytest.fixture
    def semantic_search(self, config_manager):
        """Fixture providing a semantic search instance."""
        return SemanticSearch(config_manager)
    
    @pytest.fixture
    def embedding_generator(self, config_manager):
        """Fixture providing an embedding generator."""
        return EmbeddingGenerator(config_manager)
    
    @pytest.fixture
    def sample_documents(self):
        """Fixture providing sample documents with known semantic relationships."""
        return [
            {
                'id': 'doc1',
                'text': 'The quick brown fox jumps over the lazy dog.',
                'metadata': {'category': 'animals'}
            },
            {
                'id': 'doc2',
                'text': 'A fox is a cunning and agile animal.',
                'metadata': {'category': 'animals'}
            },
            {
                'id': 'doc3',
                'text': 'Python is a popular programming language.',
                'metadata': {'category': 'technology'}
            },
            {
                'id': 'doc4',
                'text': 'Programming languages are used to write software.',
                'metadata': {'category': 'technology'}
            },
            {
                'id': 'doc5',
                'text': 'Dogs are friendly and loyal pets.',
                'metadata': {'category': 'animals'}
            }
        ]
        
    def test_basic_search(self, semantic_search, embedding_generator, sample_documents):
        """Test basic semantic search functionality."""
        # Index documents
        semantic_search.index_documents(sample_documents)
        
        # Search query
        query = "Tell me about foxes"
        results = semantic_search.search(query, k=2)
        
        assert len(results) == 2
        # Top results should be about foxes
        assert any('fox' in result['text'].lower() for result in results)
        
    def test_search_with_filters(self, semantic_search, sample_documents):
        """Test semantic search with metadata filters."""
        semantic_search.index_documents(sample_documents)
        
        # Search with category filter
        query = "What animals are mentioned?"
        filters = {'category': 'animals'}
        results = semantic_search.search(query, filters=filters, k=3)
        
        assert len(results) == 3
        assert all(result['metadata']['category'] == 'animals' for result in results)
        
    def test_search_ranking(self, semantic_search, sample_documents):
        """Test search result ranking."""
        semantic_search.index_documents(sample_documents)
        
        query = "Programming in Python"
        results = semantic_search.search(query, k=2)
        
        # Check ranking order
        assert 'python' in results[0]['text'].lower()
        assert 'programming' in results[1]['text'].lower()
        
    def test_semantic_relevance(self, semantic_search, sample_documents):
        """Test semantic relevance of search results."""
        semantic_search.index_documents(sample_documents)
        
        # Test semantically related but lexically different queries
        queries = [
            "Computer programming languages",
            "Software development tools",
            "Coding languages"
        ]
        
        for query in queries:
            results = semantic_search.search(query, k=2)
            assert all('programming' in result['text'].lower() or 
                      'python' in result['text'].lower() 
                      for result in results)
            
    def test_batch_search(self, semantic_search, sample_documents):
        """Test batch search functionality."""
        semantic_search.index_documents(sample_documents)
        
        queries = [
            "Tell me about foxes",
            "Programming languages",
            "Pets and animals"
        ]
        
        results = semantic_search.batch_search(queries, k=2)
        
        assert len(results) == len(queries)
        assert all(len(result) == 2 for result in results)
        
    def test_search_with_scores(self, semantic_search, sample_documents):
        """Test search results with similarity scores."""
        semantic_search.index_documents(sample_documents)
        
        query = "Programming languages"
        results = semantic_search.search(query, k=3, include_scores=True)
        
        assert all('score' in result for result in results)
        assert all(0 <= result['score'] <= 1 for result in results)
        # Scores should be in descending order
        scores = [result['score'] for result in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
    def test_incremental_indexing(self, semantic_search, sample_documents):
        """Test incremental document indexing."""
        # Initial indexing
        initial_docs = sample_documents[:3]
        semantic_search.index_documents(initial_docs)
        
        # Initial search
        query = "fox"
        initial_results = semantic_search.search(query, k=1)
        
        # Add more documents
        additional_docs = sample_documents[3:]
        semantic_search.add_documents(additional_docs)
        
        # Search after adding documents
        updated_results = semantic_search.search(query, k=1)
        
        assert initial_results[0]['id'] == updated_results[0]['id']
        
    def test_document_deletion(self, semantic_search, sample_documents):
        """Test document deletion from index."""
        semantic_search.index_documents(sample_documents)
        
        # Delete a document
        doc_id = 'doc1'
        semantic_search.delete_document(doc_id)
        
        # Search and verify deleted document is not returned
        query = "fox"
        results = semantic_search.search(query)
        assert not any(result['id'] == doc_id for result in results)
        
    def test_empty_index_handling(self, semantic_search):
        """Test search behavior with empty index."""
        query = "test query"
        results = semantic_search.search(query)
        
        assert len(results) == 0
        
    def test_error_handling(self, semantic_search, sample_documents):
        """Test error handling in search operations."""
        semantic_search.index_documents(sample_documents)
        
        # Test invalid k value
        with pytest.raises(ValueError):
            semantic_search.search("test", k=0)
            
        # Test invalid query type
        with pytest.raises(TypeError):
            semantic_search.search(123)
            
        # Test invalid filters
        with pytest.raises(ValueError):
            semantic_search.search("test", filters="invalid")
            
    def test_search_performance(self, semantic_search, sample_documents):
        """Test search performance."""
        semantic_search.index_documents(sample_documents)
        
        import time
        
        # Measure search time
        query = "Programming languages"
        start_time = time.time()
        results = semantic_search.search(query)
        search_time = time.time() - start_time
        
        # Search should be reasonably fast
        assert search_time < 1.0  # Less than 1 second
        
    def test_multilingual_search(self, semantic_search):
        """Test multilingual search capabilities."""
        multilingual_docs = [
            {'id': 'en1', 'text': 'The cat is on the table', 'metadata': {'lang': 'en'}},
            {'id': 'es1', 'text': 'El gato está sobre la mesa', 'metadata': {'lang': 'es'}},
            {'id': 'fr1', 'text': 'Le chat est sur la table', 'metadata': {'lang': 'fr'}}
        ]
        
        semantic_search.index_documents(multilingual_docs)
        
        # Search in different languages
        queries = {
            'en': 'Where is the cat?',
            'es': '¿Dónde está el gato?',
            'fr': 'Où est le chat?'
        }
        
        for lang, query in queries.items():
            results = semantic_search.search(query, k=1)
            assert results[0]['metadata']['lang'] == lang
