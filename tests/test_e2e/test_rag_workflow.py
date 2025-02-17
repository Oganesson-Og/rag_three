"""
Test End-to-End RAG Workflows
--------------------------

Tests for complete RAG workflows from document ingestion to query response.
"""

import pytest
import time
from pathlib import Path
from src.rag import RAGSystem
from src.document_processing import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.retrieval import SemanticSearch
from src.config import ConfigManager

class TestRAGWorkflow:
    
    @pytest.fixture
    def rag_system(self, config_manager):
        """Fixture providing a complete RAG system instance."""
        return RAGSystem(config_manager)
    
    @pytest.fixture
    def sample_documents(self):
        """Fixture providing a set of test documents."""
        return [
            {
                "text": """
                Python is a high-level programming language. It emphasizes code 
                readability with its notable use of significant indentation. Python's 
                dynamic typing and garbage collection support multiple programming 
                paradigms.
                """,
                "metadata": {"category": "programming", "source": "documentation"}
            },
            {
                "text": """
                Machine learning is a subset of artificial intelligence. It focuses 
                on developing systems that can learn from and make decisions based 
                on data. Popular machine learning frameworks include TensorFlow 
                and PyTorch.
                """,
                "metadata": {"category": "AI", "source": "textbook"}
            },
            {
                "text": """
                Data structures are ways of organizing and storing data. Common 
                data structures in Python include lists, dictionaries, sets, and 
                tuples. Each has its own use cases and performance characteristics.
                """,
                "metadata": {"category": "programming", "source": "tutorial"}
            }
        ]
        
    def test_complete_workflow(self, rag_system, sample_documents):
        """Test complete RAG workflow from ingestion to query."""
        # 1. Document Ingestion
        doc_ids = rag_system.ingest_documents(sample_documents)
        assert len(doc_ids) == len(sample_documents)
        
        # 2. Query Processing
        query = "What are Python data structures?"
        results = rag_system.process_query(query)
        
        assert len(results) > 0
        assert any("data structures" in result.text.lower() 
                  for result in results)
        assert all(hasattr(result, 'score') for result in results)
        
    def test_multi_step_workflow(self, rag_system, sample_documents):
        """Test multi-step workflow with intermediate verification."""
        # 1. Initial Setup
        rag_system.initialize()
        
        # 2. Document Processing
        processed_docs = rag_system.document_processor.process_documents(
            sample_documents
        )
        assert len(processed_docs) == len(sample_documents)
        
        # 3. Embedding Generation
        embeddings = rag_system.embedding_generator.generate_embeddings(
            [doc.text for doc in processed_docs]
        )
        assert len(embeddings) == len(processed_docs)
        
        # 4. Index Building
        rag_system.search_engine.build_index(processed_docs, embeddings)
        
        # 5. Query Processing
        query = "Explain machine learning"
        results = rag_system.process_query(query)
        
        assert len(results) > 0
        assert any("machine learning" in result.text.lower() 
                  for result in results)
        
    def test_incremental_updates(self, rag_system, sample_documents):
        """Test incremental document updates and reindexing."""
        # 1. Initial Ingestion
        initial_docs = sample_documents[:2]
        rag_system.ingest_documents(initial_docs)
        
        # 2. Initial Query
        query = "What is Python?"
        initial_results = rag_system.process_query(query)
        
        # 3. Add New Document
        new_doc = {
            "text": "Python was created by Guido van Rossum.",
            "metadata": {"category": "programming", "source": "history"}
        }
        rag_system.ingest_documents([new_doc])
        
        # 4. Query After Update
        updated_results = rag_system.process_query(query)
        
        assert len(updated_results) >= len(initial_results)
        assert any("Guido" in result.text for result in updated_results)
        
    def test_error_recovery(self, rag_system, sample_documents):
        """Test system recovery from various error conditions."""
        # 1. Simulate Failed Ingestion
        with pytest.raises(Exception):
            rag_system.ingest_documents([{"invalid": "document"}])
            
        # 2. Verify System State
        assert rag_system.is_healthy()
        
        # 3. Continue with Valid Documents
        doc_ids = rag_system.ingest_documents(sample_documents)
        assert len(doc_ids) == len(sample_documents)
        
    def test_performance_workflow(self, rag_system, sample_documents):
        """Test workflow performance and timing."""
        # 1. Measure Ingestion Time
        start_time = time.time()
        rag_system.ingest_documents(sample_documents)
        ingestion_time = time.time() - start_time
        
        # 2. Measure Query Time
        start_time = time.time()
        rag_system.process_query("Python programming")
        query_time = time.time() - start_time
        
        # Verify Performance
        assert ingestion_time < 5.0  # Should be fast
        assert query_time < 1.0  # Should be very fast
        
    def test_concurrent_operations(self, rag_system, sample_documents):
        """Test concurrent document operations and queries."""
        import threading
        
        def query_operation(system, query):
            results = system.process_query(query)
            assert len(results) > 0
            
        # 1. Initial Setup
        rag_system.ingest_documents(sample_documents)
        
        # 2. Run Concurrent Queries
        threads = []
        queries = ["Python", "machine learning", "data structures"]
        
        for query in queries:
            thread = threading.Thread(
                target=query_operation,
                args=(rag_system, query)
            )
            threads.append(thread)
            
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
    def test_filtered_workflow(self, rag_system, sample_documents):
        """Test workflow with metadata filtering."""
        # 1. Ingest Documents
        rag_system.ingest_documents(sample_documents)
        
        # 2. Query with Filters
        results = rag_system.process_query(
            "What is Python?",
            filters={"category": "programming"}
        )
        
        assert len(results) > 0
        assert all(result.metadata["category"] == "programming" 
                  for result in results)
        
    def test_persistence_workflow(self, rag_system, sample_documents, tmp_path):
        """Test workflow with system persistence."""
        # 1. Initial Setup and Ingestion
        rag_system.ingest_documents(sample_documents)
        
        # 2. Save System State
        save_path = tmp_path / "rag_state"
        rag_system.save_state(save_path)
        
        # 3. Create New System and Load State
        new_system = RAGSystem(rag_system.config_manager)
        new_system.load_state(save_path)
        
        # 4. Verify Loaded System
        results = new_system.process_query("Python")
        assert len(results) > 0
        
    def test_end_to_end_metrics(self, rag_system, sample_documents):
        """Test end-to-end workflow metrics collection."""
        # 1. Enable Metrics
        rag_system.enable_metrics()
        
        # 2. Perform Operations
        rag_system.ingest_documents(sample_documents)
        rag_system.process_query("Python programming")
        
        # 3. Check Metrics
        metrics = rag_system.get_metrics()
        assert metrics.total_documents == len(sample_documents)
        assert metrics.total_queries > 0
        assert metrics.avg_query_time > 0 