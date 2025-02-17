"""
Test API Endpoints
---------------

Tests for REST API endpoints and integration points.
"""

import pytest
import json
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.models import (
    QueryRequest,
    QueryResponse,
    DocumentRequest,
    BatchQueryRequest
)

class TestAPIEndpoints:
    
    @pytest.fixture
    def client(self):
        """Fixture providing a test client."""
        return TestClient(app)
        
    @pytest.fixture
    def sample_document(self):
        """Fixture providing a sample document for testing."""
        return {
            "text": "This is a test document for API testing.",
            "metadata": {
                "source": "test",
                "category": "documentation"
            }
        }
        
    @pytest.fixture
    def sample_query(self):
        """Fixture providing a sample query request."""
        return {
            "query": "What is the test about?",
            "filters": {"category": "documentation"},
            "top_k": 3
        }
        
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
    def test_query_endpoint(self, client, sample_query):
        """Test the query endpoint."""
        response = client.post("/query", json=sample_query)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "query_id" in data
        
    def test_batch_query_endpoint(self, client):
        """Test the batch query endpoint."""
        batch_request = {
            "queries": [
                {"query": "First test query"},
                {"query": "Second test query"}
            ]
        }
        
        response = client.post("/batch-query", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert all("results" in query_result for query_result in data["results"])
        
    def test_document_ingestion(self, client, sample_document):
        """Test document ingestion endpoint."""
        response = client.post("/documents", json=sample_document)
        
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        
    def test_batch_document_ingestion(self, client, sample_document):
        """Test batch document ingestion."""
        documents = [sample_document, sample_document]
        response = client.post("/documents/batch", json={"documents": documents})
        
        assert response.status_code == 201
        data = response.json()
        assert len(data["document_ids"]) == 2
        
    def test_document_deletion(self, client):
        """Test document deletion endpoint."""
        # First ingest a document
        doc_response = client.post("/documents", json={"text": "Test document"})
        document_id = doc_response.json()["document_id"]
        
        # Then delete it
        response = client.delete(f"/documents/{document_id}")
        assert response.status_code == 200
        
    def test_query_validation(self, client):
        """Test query parameter validation."""
        invalid_queries = [
            {},  # Empty query
            {"query": ""},  # Empty string
            {"query": "test", "top_k": 0},  # Invalid top_k
            {"query": "test", "filters": "invalid"}  # Invalid filters
        ]
        
        for query in invalid_queries:
            response = client.post("/query", json=query)
            assert response.status_code == 422
            
    def test_document_validation(self, client):
        """Test document validation."""
        invalid_documents = [
            {},  # Empty document
            {"metadata": {}},  # Missing text
            {"text": ""},  # Empty text
            {"text": 123}  # Invalid text type
        ]
        
        for doc in invalid_documents:
            response = client.post("/documents", json=doc)
            assert response.status_code == 422
            
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test non-existent endpoint
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test method not allowed
        response = client.put("/query")
        assert response.status_code == 405
        
    def test_query_with_parameters(self, client):
        """Test query endpoint with various parameters."""
        query_params = {
            "query": "test query",
            "filters": {"category": "test"},
            "top_k": 5,
            "threshold": 0.5,
            "include_metadata": True
        }
        
        response = client.post("/query", json=query_params)
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= query_params["top_k"]
        
    def test_document_update(self, client, sample_document):
        """Test document update endpoint."""
        # First ingest a document
        doc_response = client.post("/documents", json=sample_document)
        document_id = doc_response.json()["document_id"]
        
        # Update the document
        updated_doc = {
            "text": "Updated test document",
            "metadata": {"status": "updated"}
        }
        
        response = client.put(f"/documents/{document_id}", json=updated_doc)
        assert response.status_code == 200
        
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "query_metrics" in data
        assert "system_metrics" in data
        
    def test_authentication(self, client):
        """Test authentication requirements."""
        # Test without auth token
        response = client.post("/protected/query", json={"query": "test"})
        assert response.status_code == 401
        
        # Test with invalid token
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/protected/query", 
                             json={"query": "test"},
                             headers=headers)
        assert response.status_code == 401
        
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        for _ in range(10):
            response = client.post("/query", json={"query": "test"})
            
        # Should be rate limited
        response = client.post("/query", json={"query": "test"})
        assert response.status_code == 429
        
    def test_async_query(self, client):
        """Test asynchronous query endpoint."""
        response = client.post("/async-query", json={"query": "test"})
        
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        
        # Check task status
        task_id = data["task_id"]
        status_response = client.get(f"/tasks/{task_id}")
        assert status_response.status_code == 200
        
    def test_cors_headers(self, client):
        """Test CORS headers."""
        response = client.options("/query")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        
    def test_api_versioning(self, client):
        """Test API versioning."""
        # Test v1 endpoint
        response_v1 = client.post("/v1/query", json={"query": "test"})
        assert response_v1.status_code == 200
        
        # Test v2 endpoint (if exists)
        response_v2 = client.post("/v2/query", json={"query": "test"})
        assert response_v2.status_code in [200, 404]  # Depending on if v2 exists 