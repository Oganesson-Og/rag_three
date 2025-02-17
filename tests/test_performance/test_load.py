"""
Test Load and Performance
----------------------

Tests for system behavior under load and stress conditions.
"""

import pytest
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from src.rag import RAGSystem
from src.monitoring.metrics import MetricsCollector

class TestLoadPerformance:
    
    @pytest.fixture
    def rag_system(self, config_manager):
        """Fixture providing a RAG system instance."""
        return RAGSystem(config_manager)
        
    @pytest.fixture
    def large_document_set(self):
        """Fixture providing a large set of test documents."""
        documents = []
        for i in range(1000):  # 1000 documents
            documents.append({
                "text": f"Document {i} with content for testing performance. " * 20,
                "metadata": {
                    "id": str(i),
                    "category": f"category_{i % 5}"
                }
            })
        return documents
        
    def test_concurrent_queries(self, rag_system, large_document_set):
        """Test system performance under concurrent queries."""
        # Setup
        rag_system.ingest_documents(large_document_set[:100])  # Use subset for quick setup
        
        # Test queries
        queries = [
            "performance testing",
            "system analysis",
            "load testing",
            "stress testing",
            "concurrent operations"
        ] * 20  # 100 total queries
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(rag_system.process_query, queries))
            
        total_time = time.time() - start_time
        
        # Verify performance
        assert len(results) == len(queries)
        assert total_time < 30.0  # Should complete within 30 seconds
        
    def test_bulk_ingestion(self, rag_system, large_document_set):
        """Test performance of bulk document ingestion."""
        start_time = time.time()
        
        doc_ids = rag_system.ingest_documents(large_document_set)
        
        total_time = time.time() - start_time
        
        # Verify performance
        assert len(doc_ids) == len(large_document_set)
        assert total_time < 60.0  # Should complete within 60 seconds
        
        # Check memory usage
        memory_usage = rag_system.get_memory_usage()
        assert memory_usage < 1024 * 1024 * 1024  # Should use less than 1GB
        
    def test_query_latency(self, rag_system, large_document_set):
        """Test query latency under different loads."""
        # Setup
        rag_system.ingest_documents(large_document_set)
        
        latencies = []
        
        # Test different load levels
        for concurrent_queries in [1, 5, 10, 20]:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrent_queries) as executor:
                queries = ["test query"] * concurrent_queries
                results = list(executor.map(rag_system.process_query, queries))
                
            duration = time.time() - start_time
            latency = duration / concurrent_queries
            latencies.append(latency)
            
        # Verify latencies
        assert all(lat < 1.0 for lat in latencies)  # Each query should be under 1 second
        assert latencies[0] < latencies[-1]  # Latency should increase with load
        
    def test_memory_usage(self, rag_system, large_document_set):
        """Test memory usage under load."""
        import psutil
        process = psutil.Process()
        
        # Measure initial memory
        initial_memory = process.memory_info().rss
        
        # Load documents in batches
        batch_size = 100
        memory_measurements = []
        
        for i in range(0, len(large_document_set), batch_size):
            batch = large_document_set[i:i + batch_size]
            rag_system.ingest_documents(batch)
            memory_measurements.append(process.memory_info().rss)
            
        # Verify memory usage
        max_memory = max(memory_measurements)
        assert max_memory - initial_memory < 2 * 1024 * 1024 * 1024  # Less than 2GB increase
        
    def test_cpu_utilization(self, rag_system, large_document_set):
        """Test CPU utilization under load."""
        import psutil
        
        # Setup
        rag_system.ingest_documents(large_document_set[:500])
        
        cpu_measurements = []
        
        def measure_cpu():
            while len(cpu_measurements) < 10:
                cpu_measurements.append(psutil.cpu_percent(interval=1))
                
        # Start CPU measurement
        cpu_thread = threading.Thread(target=measure_cpu)
        cpu_thread.start()
        
        # Generate load
        with ThreadPoolExecutor(max_workers=5) as executor:
            queries = ["test query"] * 50
            list(executor.map(rag_system.process_query, queries))
            
        cpu_thread.join()
        
        # Verify CPU usage
        avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
        assert avg_cpu < 80  # Should not max out CPU
        
    def test_concurrent_mixed_operations(self, rag_system, large_document_set):
        """Test performance with mixed read/write operations."""
        def query_operation():
            for _ in range(10):
                rag_system.process_query("test query")
                time.sleep(0.1)
                
        def ingest_operation():
            for i in range(10):
                doc = {
                    "text": f"New document {i} for testing",
                    "metadata": {"type": "new"}
                }
                rag_system.ingest_documents([doc])
                time.sleep(0.2)
                
        # Start mixed operations
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=query_operation))
            threads.append(threading.Thread(target=ingest_operation))
            
        start_time = time.time()
        
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        total_time = time.time() - start_time
        
        # Verify performance
        assert total_time < 30.0  # Should complete within 30 seconds
        
    def test_system_stability(self, rag_system, large_document_set):
        """Test system stability under extended load."""
        # Setup
        rag_system.ingest_documents(large_document_set)
        
        error_count = 0
        total_queries = 1000
        
        # Run extended test
        for i in range(total_queries):
            try:
                results = rag_system.process_query(f"test query {i}")
                assert len(results) > 0
            except Exception:
                error_count += 1
                
            if i % 100 == 0:
                time.sleep(1)  # Brief pause every 100 queries
                
        # Verify stability
        assert error_count == 0  # No errors should occur
        assert rag_system.is_healthy()  # System should remain healthy
        
    def test_recovery_performance(self, rag_system, large_document_set):
        """Test system recovery performance after high load."""
        # Generate high load
        with ThreadPoolExecutor(max_workers=20) as executor:
            queries = ["stress test"] * 200
            list(executor.map(rag_system.process_query, queries))
            
        # Measure recovery
        start_time = time.time()
        results = rag_system.process_query("test query")
        recovery_time = time.time() - start_time
        
        # Verify recovery
        assert recovery_time < 1.0  # Should recover quickly
        assert len(results) > 0
        
    def test_cache_performance(self, rag_system, large_document_set):
        """Test caching performance under load."""
        # Setup
        rag_system.ingest_documents(large_document_set)
        
        # First query (cache miss)
        start_time = time.time()
        first_results = rag_system.process_query("cache test")
        first_query_time = time.time() - start_time
        
        # Second query (cache hit)
        start_time = time.time()
        second_results = rag_system.process_query("cache test")
        second_query_time = time.time() - start_time
        
        # Verify cache performance
        assert second_query_time < first_query_time * 0.5  # Cache should be significantly faster
        assert first_results == second_results  # Results should be consistent 