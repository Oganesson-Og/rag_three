"""
Test API Integration Module
-------------------------

Tests for API endpoints and integration with the RAG system.
"""

import pytest
import json
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
from src.api.main import app
from src.config.config_manager import ConfigManager

class TestAPIIntegration:
    
    @pytest.fixture
    def client(self):
        """Fixture providing a FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def test_file(self):
        """Fixture providing a temporary test file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tf:
            tf.write(b"Test document content for API testing.")
            temp_path = tf.name
        yield temp_path
        Path(temp_path).unlink()
        
    @pytest.fixture
    def auth_headers(self):
        """Fixture providing authentication headers."""
        return {"Authorization": "Bearer test-token"}
    
    def test_health_check(self, client):
        """Test API health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
    def test_document_upload(self, client, test_file, auth_headers):
        """Test document upload endpoint."""
        with open(test_file, 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            response = client.post(
                "/documents/upload",
                files=files,
                headers=auth_headers
            )
            
        assert response.status_code == 200
        assert "document_id" in response.json()
        
        # Store document_id for later tests
        return response.json()["document_id"]
        
    def test_document_processing_status(self, client, test_file, auth_headers):
        """Test document processing status endpoint."""
        # First upload a document
        doc_id = self.test_document_upload(client, test_file, auth_headers)
        
        response = client.get(
            f"/documents/{doc_id}/status",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] in ["pending", "processing", "completed", "failed"]
        
    def test_search_endpoint(self, client, auth_headers):
        """Test search endpoint."""
        query = "test content"
        response = client.post(
            "/search",
            json={"query": query, "k": 5},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "results" in response.json()
        assert len(response.json()["results"]) <= 5
        
    def test_batch_search_endpoint(self, client, auth_headers):
        """Test batch search endpoint."""
        queries = ["test content", "another query"]
        response = client.post(
            "/search/batch",
            json={"queries": queries, "k": 3},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "results" in response.json()
        assert len(response.json()["results"]) == len(queries)
        
    def test_document_metadata(self, client, test_file, auth_headers):
        """Test document metadata endpoints."""
        # First upload a document
        doc_id = self.test_document_upload(client, test_file, auth_headers)
        
        # Get metadata
        response = client.get(
            f"/documents/{doc_id}/metadata",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "metadata" in response.json()
        
        # Update metadata
        new_metadata = {"subject": "test", "tags": ["api", "testing"]}
        response = client.put(
            f"/documents/{doc_id}/metadata",
            json=new_metadata,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json()["metadata"] == new_metadata
        
    def test_error_handling(self, client, auth_headers):
        """Test API error handling."""
        # Test invalid document ID
        response = client.get(
            "/documents/invalid-id/status",
            headers=auth_headers
        )
        assert response.status_code == 404
        
        # Test invalid query
        response = client.post(
            "/search",
            json={"query": "", "k": 5},
            headers=auth_headers
        )
        assert response.status_code == 400
        
        # Test invalid authentication
        response = client.get(
            "/documents/test/status",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401
        
    def test_bulk_operations(self, client, auth_headers):
        """Test bulk operation endpoints."""
        # Bulk document upload
        docs = [
            {"content": "Test document 1", "metadata": {"type": "test"}},
            {"content": "Test document 2", "metadata": {"type": "test"}}
        ]
        
        response = client.post(
            "/documents/bulk-upload",
            json={"documents": docs},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "document_ids" in response.json()
        assert len(response.json()["document_ids"]) == len(docs)
        
    def test_system_configuration(self, client, auth_headers):
        """Test system configuration endpoints."""
        # Get current configuration
        response = client.get(
            "/system/config",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "config" in response.json()
        
        # Update configuration
        new_config = {
            "semantic_weight": 0.7,
            "keyword_weight": 0.3
        }
        
        response = client.put(
            "/system/config",
            json=new_config,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json()["config"]["semantic_weight"] == 0.7
        
    def test_api_rate_limiting(self, client, auth_headers):
        """Test API rate limiting."""
        # Make multiple rapid requests
        for _ in range(10):
            response = client.get("/health")
            
        # Should be rate limited
        response = client.get("/health")
        assert response.status_code == 429
