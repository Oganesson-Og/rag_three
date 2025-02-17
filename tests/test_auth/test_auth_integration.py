"""
Test Auth Integration
------------------

Tests for RAG system integration with existing authentication system.
"""

import pytest
from src.auth.auth_integration import AuthIntegration
from src.rag import RAGSystem

class TestAuthIntegration:
    
    @pytest.fixture
    def auth_integration(self, config_manager):
        """Fixture providing auth integration instance."""
        return AuthIntegration(config_manager)
    
    @pytest.fixture
    def rag_system(self, config_manager):
        """Fixture providing RAG system instance."""
        return RAGSystem(config_manager)
    
    @pytest.fixture
    def mock_user_context(self):
        """Fixture providing mock user context."""
        return {
            'user_id': 'test_user_123',
            'access_level': 'standard',
            'department': 'engineering'
        }
        
    def test_authorized_query(self, auth_integration, rag_system, mock_user_context):
        """Test RAG query with authorized user context."""
        query = "What is machine learning?"
        
        results = rag_system.process_query(
            query,
            user_context=mock_user_context
        )
        
        assert len(results) > 0
        assert all(auth_integration.can_access_document(
            mock_user_context, result.metadata) 
            for result in results
        )
        
    def test_document_access_filtering(self, auth_integration, mock_user_context):
        """Test document access filtering based on user context."""
        documents = [
            {
                'text': 'Public document',
                'metadata': {'access_level': 'public'}
            },
            {
                'text': 'Engineering document',
                'metadata': {'access_level': 'restricted', 
                           'department': 'engineering'}
            },
            {
                'text': 'HR document',
                'metadata': {'access_level': 'restricted',
                           'department': 'hr'}
            }
        ]
        
        filtered_docs = auth_integration.filter_accessible_documents(
            documents,
            mock_user_context
        )
        
        assert len(filtered_docs) == 2  # Public + Engineering docs
        assert all(doc['metadata']['access_level'] == 'public' or
                  doc['metadata']['department'] == 'engineering'
                  for doc in filtered_docs)
        
    def test_metadata_filtering(self, auth_integration, mock_user_context):
        """Test metadata filtering for security."""
        document = {
            'text': 'Test document',
            'metadata': {
                'access_level': 'restricted',
                'department': 'engineering',
                'internal_id': 'sensitive_123',
                'created_by': 'admin'
            }
        }
        
        safe_metadata = auth_integration.filter_metadata(
            document['metadata'],
            mock_user_context
        )
        
        assert 'access_level' not in safe_metadata
        assert 'internal_id' not in safe_metadata
        assert 'department' in safe_metadata
        
    def test_query_logging(self, auth_integration, mock_user_context):
        """Test secure query logging."""
        query = "confidential project details"
        
        auth_integration.log_query(
            query=query,
            user_context=mock_user_context,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        logs = auth_integration.get_query_logs(mock_user_context['user_id'])
        assert len(logs) > 0
        assert logs[-1]['query'] == query
        
    def test_invalid_user_context(self, auth_integration, rag_system):
        """Test handling of invalid user context."""
        invalid_context = {'user_id': 'invalid'}
        
        with pytest.raises(ValueError):
            rag_system.process_query(
                "test query",
                user_context=invalid_context
            )
            
    def test_department_specific_search(self, auth_integration, rag_system, mock_user_context):
        """Test department-specific search results."""
        query = "project documentation"
        
        results = rag_system.process_query(
            query,
            user_context=mock_user_context
        )
        
        # Verify results are appropriate for user's department
        for result in results:
            assert (result.metadata.get('department') == 'engineering' or
                   result.metadata.get('access_level') == 'public') 