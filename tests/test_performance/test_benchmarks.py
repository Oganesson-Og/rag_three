"""
Test Performance Benchmarks
-------------------------

Tests for system performance metrics and benchmarks.
"""

import pytest
import time
import numpy as np
from pathlib import Path
import psutil
import torch
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.hybrid_search import HybridSearch
from src.document_processing.processor import DocumentProcessor

class TestPerformanceBenchmarks:
    
    @pytest.fixture
    def performance_data(self, tmp_path):
        """Fixture providing test data for performance testing."""
        # Generate synthetic documents
        docs = []
        for i in range(1000):  # 1000 test documents
            content = f"Document {i} content with enough text to make it meaningful. " * 50
            file_path = tmp_path / f"doc_{i}.txt"
            file_path.write_text(content)
            docs.append(file_path)
        return docs
        
    def test_embedding_generation_speed(self, config_manager, performance_data):
        """Test embedding generation performance."""
        embedding_generator = EmbeddingGenerator(config_manager)
        
        batch_sizes = [1, 8, 16, 32, 64]
        timings = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Process in batches
            for i in range(0, len(performance_data), batch_size):
                batch = performance_data[i:i + batch_size]
                texts = [doc.read_text() for doc in batch]
                embeddings = embedding_generator.generate_embeddings(
                    texts,
                    batch_size=batch_size
                )
                
            elapsed = time.time() - start_time
            timings[batch_size] = elapsed
            
        # Assert performance requirements
        assert timings[32] < timings[1]  # Batching should be faster
        assert min(timings.values()) < len(performance_data) * 0.1  # Less than 100ms per document
        
    def test_search_latency(self, config_manager, performance_data):
        """Test search operation latency."""
        hybrid_search = HybridSearch(config_manager)
        
        # Index documents first
        doc_processor = DocumentProcessor()
        processed_docs = []
        for doc in performance_data[:100]:  # Use subset for search testing
            processed = doc_processor.process_document(doc)
            processed_docs.append(processed)
            
        hybrid_search.index_documents(processed_docs)
        
        # Measure search latency
        query = "test query for performance measurement"
        latencies = []
        
        for _ in range(100):  # 100 search operations
            start_time = time.time()
            results = hybrid_search.search(query, k=5)
            latency = time.time() - start_time
            latencies.append(latency)
            
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 0.1  # Average latency under 100ms
        assert p95_latency < 0.2  # 95th percentile under 200ms
        
    def test_memory_usage(self, config_manager, performance_data):
        """Test memory usage during operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Load and process documents
        doc_processor = DocumentProcessor()
        processed_docs = []
        
        for doc in performance_data:
            processed = doc_processor.process_document(doc)
            processed_docs.append(processed)
            
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Assert memory growth is reasonable
            assert memory_increase < 1e9  # Less than 1GB increase
            
    def test_gpu_memory_usage(self, config_manager):
        """Test GPU memory usage if available."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
            
        initial_memory = torch.cuda.memory_allocated()
        
        embedding_generator = EmbeddingGenerator(config_manager)
        
        # Generate embeddings
        text = "Test text for GPU memory measurement." * 100
        embedding = embedding_generator.generate_embedding(text)
        
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 1e9  # Less than 1GB GPU memory increase
        
    def test_concurrent_operations(self, config_manager, performance_data):
        """Test system performance under concurrent operations."""
        import concurrent.futures
        
        hybrid_search = HybridSearch(config_manager)
        
        # Index some documents first
        doc_processor = DocumentProcessor()
        processed_docs = [doc_processor.process_document(doc) 
                         for doc in performance_data[:100]]
        hybrid_search.index_documents(processed_docs)
        
        # Test concurrent searches
        def search_operation():
            query = "test query for concurrent operation"
            return hybrid_search.search(query, k=5)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_operation) for _ in range(50)]
            results = [future.result() for future in futures]
            
        assert len(results) == 50
        assert all(len(result) == 5 for result in results)
        
    def test_index_update_performance(self, config_manager, performance_data):
        """Test performance of index updates."""
        hybrid_search = HybridSearch(config_manager)
        doc_processor = DocumentProcessor()
        
        # Measure initial indexing time
        start_time = time.time()
        processed_docs = [doc_processor.process_document(doc) 
                         for doc in performance_data[:500]]
        hybrid_search.index_documents(processed_docs)
        initial_index_time = time.time() - start_time
        
        # Measure update time
        start_time = time.time()
        update_docs = [doc_processor.process_document(doc) 
                      for doc in performance_data[500:600]]
        hybrid_search.add_documents(update_docs)
        update_time = time.time() - start_time
        
        assert update_time < initial_index_time / 5  # Updates should be faster
        
    def test_system_load(self, config_manager, performance_data):
        """Test system load during intensive operations."""
        start_cpu_percent = psutil.cpu_percent(interval=1)
        start_memory_percent = psutil.virtual_memory().percent
        
        # Perform intensive operations
        embedding_generator = EmbeddingGenerator(config_manager)
        doc_processor = DocumentProcessor()
        
        for doc in performance_data[:100]:
            processed = doc_processor.process_document(doc)
            text = processed['text']
            embedding = embedding_generator.generate_embedding(text)
            
        end_cpu_percent = psutil.cpu_percent(interval=1)
        end_memory_percent = psutil.virtual_memory().percent
        
        cpu_increase = end_cpu_percent - start_cpu_percent
        memory_increase = end_memory_percent - start_memory_percent
        
        assert cpu_increase < 80  # CPU usage shouldn't spike too high
        assert memory_increase < 50  # Memory usage shouldn't increase too much 