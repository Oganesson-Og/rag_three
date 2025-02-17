"""
Test System Scalability
---------------------

Tests for system scalability and resource management.
"""

import pytest
import numpy as np
from pathlib import Path
import time
import multiprocessing
from src.retrieval.vector_store import VectorStore
from src.document_processing.processor import DocumentProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator

class TestScalability:
    
    @pytest.fixture
    def large_dataset(self, tmp_path):
        """Fixture providing a large dataset for scalability testing."""
        # Generate large synthetic dataset
        docs = []
        for i in range(10000):  # 10K documents
            content = f"Large document {i} content for scalability testing. " * 100
            file_path = tmp_path / f"large_doc_{i}.txt"
            file_path.write_text(content)
            docs.append(file_path)
        return docs
        
    def test_large_scale_indexing(self, config_manager, large_dataset):
        """Test indexing performance with large datasets."""
        vector_store = VectorStore(config_manager)
        doc_processor = DocumentProcessor()
        embedding_generator = EmbeddingGenerator(config_manager)
        
        batch_sizes = [100, 500, 1000]
        indexing_times = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            for i in range(0, len(large_dataset), batch_size):
                batch = large_dataset[i:i + batch_size]
                processed_docs = [doc_processor.process_document(doc) for doc in batch]
                texts = [doc['text'] for doc in processed_docs]
                embeddings = embedding_generator.generate_embeddings(texts)
                vector_store.add_vectors(embeddings, processed_docs)
                
            indexing_times[batch_size] = time.time() - start_time
            
        # Verify scaling efficiency
        assert indexing_times[1000] < indexing_times[100] * 10
        
    def test_parallel_processing(self, config_manager, large_dataset):
        """Test parallel processing capabilities."""
        doc_processor = DocumentProcessor()
        
        def process_chunk(docs):
            return [doc_processor.process_document(doc) for doc in docs]
            
        # Split data into chunks for parallel processing
        num_cores = multiprocessing.cpu_count()
        chunk_size = len(large_dataset) // num_cores
        chunks = [large_dataset[i:i + chunk_size] 
                 for i in range(0, len(large_dataset), chunk_size)]
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            start_time = time.time()
            results = pool.map(process_chunk, chunks)
            parallel_time = time.time() - start_time
            
        # Sequential processing for comparison
        start_time = time.time()
        sequential_results = process_chunk(large_dataset)
        sequential_time = time.time() - start_time
        
        assert parallel_time < sequential_time / 2
        
    def test_memory_scaling(self, config_manager, large_dataset):
        """Test memory usage scaling with dataset size."""
        import psutil
        
        vector_store = VectorStore(config_manager)
        doc_processor = DocumentProcessor()
        embedding_generator = EmbeddingGenerator(config_manager)
        
        process = psutil.Process()
        memory_usage = []
        
        for i in range(0, len(large_dataset), 1000):
            batch = large_dataset[i:i + 1000]
            processed_docs = [doc_processor.process_document(doc) for doc in batch]
            texts = [doc['text'] for doc in processed_docs]
            embeddings = embedding_generator.generate_embeddings(texts)
            vector_store.add_vectors(embeddings, processed_docs)
            
            memory_usage.append(process.memory_info().rss)
            
        # Check memory growth rate
        memory_growth = np.diff(memory_usage)
        assert np.mean(memory_growth) < 1e8  # Average growth under 100MB per batch
        
    def test_query_scaling(self, config_manager, large_dataset):
        """Test query performance scaling with index size."""
        vector_store = VectorStore(config_manager)
        doc_processor = DocumentProcessor()
        embedding_generator = EmbeddingGenerator(config_manager)
        
        index_sizes = [1000, 5000, 10000]
        query_times = {}
        
        for size in index_sizes:
            # Index documents
            batch = large_dataset[:size]
            processed_docs = [doc_processor.process_document(doc) for doc in batch]
            texts = [doc['text'] for doc in processed_docs]
            embeddings = embedding_generator.generate_embeddings(texts)
            vector_store.add_vectors(embeddings, processed_docs)
            
            # Measure query time
            query = "test query for scaling measurement"
            query_embedding = embedding_generator.generate_embedding(query)
            
            start_time = time.time()
            for _ in range(100):  # 100 queries
                results = vector_store.search(query_embedding, k=10)
            query_times[size] = (time.time() - start_time) / 100
            
        # Verify sub-linear scaling
        assert query_times[10000] < query_times[1000] * 10
        
    def test_distributed_operations(self, config_manager, large_dataset):
        """Test distributed processing capabilities."""
        try:
            import ray
            ray.init()
            
            @ray.remote
            def process_chunk(chunk):
                doc_processor = DocumentProcessor()
                return [doc_processor.process_document(doc) for doc in chunk]
                
            # Split dataset into chunks
            chunks = np.array_split(large_dataset, 10)
            
            # Distributed processing
            start_time = time.time()
            futures = [process_chunk.remote(chunk) for chunk in chunks]
            results = ray.get(futures)
            distributed_time = time.time() - start_time
            
            # Local processing
            start_time = time.time()
            local_results = [process_chunk(chunk) for chunk in chunks]
            local_time = time.time() - start_time
            
            assert distributed_time < local_time
            
        except ImportError:
            pytest.skip("Ray not installed")
            
    def test_index_optimization(self, config_manager, large_dataset):
        """Test index optimization for large scales."""
        vector_store = VectorStore(config_manager)
        doc_processor = DocumentProcessor()
        embedding_generator = EmbeddingGenerator(config_manager)
        
        # Index with different configurations
        index_configs = [
            {'index_type': 'flat'},
            {'index_type': 'ivf', 'nlist': 100},
            {'index_type': 'hnsw', 'M': 16}
        ]
        
        query = "test query for optimization"
        query_embedding = embedding_generator.generate_embedding(query)
        
        for config in index_configs:
            vector_store.create_index(**config)
            
            # Add vectors
            processed_docs = [doc_processor.process_document(doc) 
                            for doc in large_dataset[:5000]]
            texts = [doc['text'] for doc in processed_docs]
            embeddings = embedding_generator.generate_embeddings(texts)
            vector_store.add_vectors(embeddings, processed_docs)
            
            # Measure search time
            start_time = time.time()
            for _ in range(100):
                results = vector_store.search(query_embedding, k=10)
            search_time = time.time() - start_time
            
            assert search_time < 10  # Less than 10 seconds for 100 searches 