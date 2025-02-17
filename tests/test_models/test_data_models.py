"""
Test Data Models
-------------

Tests for data model validation, serialization, and behavior.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from src.models.document import Document, DocumentMetadata
from src.models.query import Query, QueryResult, SearchMetadata
from src.models.embedding import EmbeddingVector, EmbeddingMetadata
from src.models.config import ModelConfig, ProcessingConfig
from src.models.cache import CacheEntry, CacheMetadata

class TestDataModels:
    
    @pytest.fixture
    def sample_document(self):
        """Fixture providing a sample document."""
        return Document(
            id="doc123",
            text="Sample document text",
            metadata=DocumentMetadata(
                source="test",
                created_at=datetime.now(),
                language="en",
                category="test"
            )
        )
        
    @pytest.fixture
    def sample_query(self):
        """Fixture providing a sample query."""
        return Query(
            text="test query",
            filters={"category": "test"},
            top_k=5
        )
        
    def test_document_validation(self):
        """Test document model validation."""
        # Valid document
        doc = Document(
            id="test123",
            text="Valid document",
            metadata=DocumentMetadata(
                source="test",
                created_at=datetime.now()
            )
        )
        assert doc.id == "test123"
        
        # Invalid document
        with pytest.raises(ValidationError):
            Document(
                id="",  # Empty ID
                text="Invalid document"
            )
            
    def test_query_validation(self):
        """Test query model validation."""
        # Valid query
        query = Query(
            text="valid query",
            filters={"category": "test"},
            top_k=5
        )
        assert query.text == "valid query"
        
        # Invalid query
        with pytest.raises(ValidationError):
            Query(
                text="",  # Empty query
                top_k=-1  # Invalid top_k
            )
            
    def test_embedding_vector(self):
        """Test embedding vector model."""
        import numpy as np
        
        vector = np.random.rand(384).astype(np.float32)
        
        embedding = EmbeddingVector(
            vector=vector,
            metadata=EmbeddingMetadata(
                model_name="test-model",
                dimension=384
            )
        )
        
        assert embedding.vector.shape == (384,)
        assert embedding.metadata.dimension == 384
        
    def test_query_result(self):
        """Test query result model."""
        result = QueryResult(
            text="Result text",
            score=0.95,
            metadata=SearchMetadata(
                document_id="doc123",
                rank=1
            )
        )
        
        assert result.score >= 0 and result.score <= 1
        assert result.metadata.rank > 0
        
    def test_model_serialization(self, sample_document):
        """Test model serialization/deserialization."""
        # Serialize
        doc_dict = sample_document.model_dump()
        
        # Deserialize
        doc_restored = Document.model_validate(doc_dict)
        
        assert doc_restored == sample_document
        
    def test_config_validation(self):
        """Test configuration model validation."""
        # Valid config
        config = ModelConfig(
            model_name="test-model",
            dimension=384,
            batch_size=32
        )
        assert config.model_name == "test-model"
        
        # Invalid config
        with pytest.raises(ValidationError):
            ModelConfig(
                model_name="",  # Empty model name
                dimension=-1  # Invalid dimension
            )
            
    def test_cache_entry(self):
        """Test cache entry model."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            metadata=CacheMetadata(
                created_at=datetime.now(),
                expires_at=datetime.now(),
                size_bytes=100
            )
        )
        
        assert entry.key == "test_key"
        assert entry.metadata.size_bytes > 0
        
    def test_model_relationships(self, sample_document, sample_query):
        """Test relationships between models."""
        # Create query result referencing document
        result = QueryResult(
            text=sample_document.text,
            score=0.9,
            metadata=SearchMetadata(
                document_id=sample_document.id,
                rank=1
            )
        )
        
        assert result.metadata.document_id == sample_document.id
        
    def test_model_inheritance(self):
        """Test model inheritance behavior."""
        class ExtendedMetadata(DocumentMetadata):
            extra_field: str
            
        metadata = ExtendedMetadata(
            source="test",
            created_at=datetime.now(),
            extra_field="test_value"
        )
        
        assert metadata.extra_field == "test_value"
        
    def test_model_methods(self, sample_document):
        """Test custom model methods."""
        # Test document text truncation
        truncated = sample_document.get_truncated_text(max_length=10)
        assert len(truncated) <= 10
        
        # Test metadata formatting
        formatted = sample_document.metadata.format_created_at()
        assert isinstance(formatted, str)
        
    def test_model_validation_rules(self):
        """Test custom validation rules."""
        class ValidatedQuery(Query):
            @property
            def is_valid(self) -> bool:
                return len(self.text) >= 3 and self.top_k > 0
                
        # Valid query
        query = ValidatedQuery(text="valid", top_k=5)
        assert query.is_valid
        
        # Invalid query
        query = ValidatedQuery(text="ab", top_k=0)
        assert not query.is_valid
        
    def test_model_defaults(self):
        """Test model default values."""
        doc = Document(
            id="test",
            text="test"
            # metadata is optional with defaults
        )
        
        assert doc.metadata.language == "en"  # Default language
        assert doc.metadata.created_at is not None
        
    def test_model_constraints(self):
        """Test model field constraints."""
        with pytest.raises(ValidationError):
            Query(
                text="test",
                top_k=1000  # Exceeds maximum allowed value
            )
            
        with pytest.raises(ValidationError):
            Document(
                id="test",
                text="a" * 1000000  # Exceeds maximum length
            )
            
    def test_model_computed_fields(self, sample_document):
        """Test computed model fields."""
        assert sample_document.text_length == len(sample_document.text)
        assert sample_document.metadata.age_hours >= 0
        
    def test_model_updates(self, sample_document):
        """Test model update behavior."""
        # Update metadata
        sample_document.metadata.category = "new_category"
        assert sample_document.metadata.category == "new_category"
        
        # Update with new metadata
        new_metadata = DocumentMetadata(
            source="new_source",
            created_at=datetime.now()
        )
        sample_document.metadata = new_metadata
        assert sample_document.metadata.source == "new_source" 