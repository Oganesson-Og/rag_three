"""
Test End-to-End Integration Module
--------------------------------

Tests for complete RAG pipeline integration and workflow.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.document_processing.processor import DocumentProcessor
from src.retrieval.hybrid_search import HybridSearch
from src.config.config_manager import ConfigManager

class TestEndToEnd:
    
    @pytest.fixture
    def test_environment(self):
        """Fixture providing a complete test environment."""
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create necessary subdirectories
        (temp_dir / "documents").mkdir()
        (temp_dir / "processed").mkdir()
        (temp_dir / "embeddings").mkdir()
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def rag_components(self, config_manager):
        """Fixture providing initialized RAG components."""
        embedding_generator = EmbeddingGenerator(config_manager)
        document_processor = DocumentProcessor()
        hybrid_search = HybridSearch(embedding_generator)
        
        return {
            'embedding_generator': embedding_generator,
            'document_processor': document_processor,
            'hybrid_search': hybrid_search
        }
        
    def test_complete_workflow(self, test_environment, rag_components):
        """Test complete RAG workflow from document ingestion to search."""
        # Create test documents
        docs_dir = test_environment / "documents"
        test_doc1 = docs_dir / "test1.txt"
        test_doc1.write_text("""
        Neural networks are a fundamental concept in deep learning.
        They consist of layers of interconnected nodes that process information.
        """)
        
        test_doc2 = docs_dir / "test2.txt"
        test_doc2.write_text("""
        The Pythagorean theorem states that a² + b² = c² in a right triangle.
        This fundamental mathematical principle has many applications.
        """)
        
        # Process documents
        doc_processor = rag_components['document_processor']
        processed_docs = []
        
        for doc_path in docs_dir.glob("*.txt"):
            processed_doc = doc_processor.process_document(doc_path)
            processed_docs.append(processed_doc)
            
        assert len(processed_docs) == 2
        
        # Generate embeddings
        embedding_generator = rag_components['embedding_generator']
        embeddings = []
        
        for doc in processed_docs:
            embedding = embedding_generator.generate_embedding(doc['text'])
            embeddings.append(embedding)
            
        assert len(embeddings) == 2
        
        # Index documents for search
        hybrid_search = rag_components['hybrid_search']
        hybrid_search.index_documents(processed_docs)
        
        # Perform search queries
        results = hybrid_search.search(
            "neural networks deep learning",
            k=2
        )
        
        assert len(results) > 0
        assert "neural" in results[0]['text'].lower()
        
    def test_incremental_updates(self, test_environment, rag_components):
        """Test incremental updates to the RAG system."""
        # Initial setup
        doc_processor = rag_components['document_processor']
        hybrid_search = rag_components['hybrid_search']
        
        # Add initial document
        initial_doc = test_environment / "documents" / "initial.txt"
        initial_doc.write_text("Initial document about machine learning.")
        
        processed_doc = doc_processor.process_document(initial_doc)
        hybrid_search.index_documents([processed_doc])
        
        # Add new document
        new_doc = test_environment / "documents" / "new.txt"
        new_doc.write_text("New document about deep learning neural networks.")
        
        processed_new_doc = doc_processor.process_document(new_doc)
        hybrid_search.add_documents([processed_new_doc])
        
        # Search should find both documents
        results = hybrid_search.search("learning", k=2)
        assert len(results) == 2
        
    def test_error_recovery(self, test_environment, rag_components):
        """Test system recovery from various error conditions."""
        doc_processor = rag_components['document_processor']
        hybrid_search = rag_components['hybrid_search']
        
        # Test with corrupted document
        corrupt_doc = test_environment / "documents" / "corrupt.txt"
        corrupt_doc.write_bytes(b'\x00\x00\x00')
        
        try:
            doc_processor.process_document(corrupt_doc)
        except Exception as e:
            assert isinstance(e, Exception)
            
        # System should continue working
        valid_doc = test_environment / "documents" / "valid.txt"
        valid_doc.write_text("Valid document text.")
        
        processed_doc = doc_processor.process_document(valid_doc)
        hybrid_search.index_documents([processed_doc])
        
        results = hybrid_search.search("valid", k=1)
        assert len(results) == 1
        
    def test_large_scale_processing(self, test_environment, rag_components):
        """Test system performance with larger document sets."""
        doc_processor = rag_components['document_processor']
        hybrid_search = rag_components['hybrid_search']
        
        # Create multiple test documents
        docs_dir = test_environment / "documents"
        processed_docs = []
        
        for i in range(20):  # Create 20 test documents
            doc_path = docs_dir / f"doc_{i}.txt"
            doc_path.write_text(f"Document {i} content about topic {i % 5}")
            
            processed_doc = doc_processor.process_document(doc_path)
            processed_docs.append(processed_doc)
            
        # Batch index documents
        hybrid_search.index_documents(processed_docs)
        
        # Test search performance
        results = hybrid_search.search("topic 2", k=5)
        assert len(results) == 5
        
    def test_config_changes(self, test_environment, rag_components, config_manager):
        """Test system adaptation to configuration changes."""
        hybrid_search = rag_components['hybrid_search']
        
        # Create test document
        test_doc = test_environment / "documents" / "test.txt"
        test_doc.write_text("Test document content.")
        
        # Process with initial config
        initial_results = hybrid_search.search(
            "test content",
            semantic_weight=0.5,
            keyword_weight=0.5
        )
        
        # Modify search weights
        new_results = hybrid_search.search(
            "test content",
            semantic_weight=0.8,
            keyword_weight=0.2
        )
        
        # Results should differ
        assert initial_results[0]['combined_score'] != new_results[0]['combined_score']
