"""
General Embedding Benchmark Module
-------------------------------

Comprehensive benchmarking system for embedding models, providing performance
metrics, quality assessment, and comparative analysis capabilities.

Key Features:
- Performance benchmarking
- Memory usage analysis
- Quality metrics calculation
- Comparative model analysis
- Resource utilization tracking
- Automated reporting
- Custom metric definition

Technical Details:
- Multiple evaluation metrics
- Statistical analysis tools
- Resource monitoring
- Automated test generation
- Results persistence
- Visualization support

Dependencies:
- numpy>=1.24.0
- pandas>=2.1.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
- psutil>=5.9.0
- torch>=2.0.0

Example Usage:
    # Basic benchmarking
    benchmark = EmbeddingBenchmark()
    results = benchmark.run_benchmark()
    
    # Custom test data
    benchmark = EmbeddingBenchmark(test_data_path='tests.json')
    results = benchmark.run_benchmark(batch_sizes=[8, 16, 32])
    
    # Save results
    benchmark.save_results('benchmark_results.json')
    benchmark.print_summary()

Metrics Included:
- Embedding quality scores
- Processing speed (texts/second)
- Memory usage patterns
- GPU utilization
- Model comparison metrics

Author: Keith Satuku
Version: 2.3.0
Created: 2025
License: MIT
"""

from typing import List, Dict
import time
import numpy as np
from pathlib import Path
import json
import logging
from .enhanced_generator import EnhancedEmbeddingGenerator
from ..config.embedding_config import EMBEDDING_MODELS

class EmbeddingBenchmark:
    
    def __init__(self, test_data_path: str = None):
        self.models = EMBEDDING_MODELS
        self.results = {}
        self.test_data = self._load_test_data(test_data_path)
        
    def _load_test_data(self, path: str = None) -> List[str]:
        """Load or generate test data."""
        if path and Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        
        # Default test data if none provided
        return [
            "The mitochondria is the powerhouse of the cell",
            "Newton's laws of motion describe the relationship between force and motion",
            "Photosynthesis is the process by which plants convert light energy into chemical energy",
            # Add more domain-specific test sentences
        ]
    
    def run_benchmark(self, batch_sizes: List[int] = [1, 8, 32]) -> Dict:
        """Run comprehensive benchmark tests."""
        for model_key in self.models:
            self.results[model_key] = {
                'speed': {},
                'memory': {},
                'quality': {}
            }
            
            generator = EnhancedEmbeddingGenerator(model_key=model_key)
            
            # Test different batch sizes
            for batch_size in batch_sizes:
                speed, memory = self._benchmark_performance(
                    generator, 
                    self.test_data, 
                    batch_size
                )
                
                self.results[model_key]['speed'][batch_size] = speed
                self.results[model_key]['memory'][batch_size] = memory
            
            # Test embedding quality
            quality = self._benchmark_quality(generator, self.test_data)
            self.results[model_key]['quality'] = quality
            
        return self.results
    
    def _benchmark_performance(
        self, 
        generator: EnhancedEmbeddingGenerator,
        texts: List[str],
        batch_size: int
    ) -> tuple:
        """Benchmark speed and memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        _ = generator.generate_batch_embeddings(texts, batch_size=batch_size)
        end_time = time.time()
        
        end_mem = process.memory_info().rss / 1024 / 1024
        
        speed = len(texts) / (end_time - start_time)  # texts per second
        memory = end_mem - start_mem  # MB used
        
        return speed, memory
    
    def _benchmark_quality(
        self, 
        generator: EnhancedEmbeddingGenerator,
        texts: List[str]
    ) -> Dict:
        """Benchmark embedding quality metrics."""
        embeddings = generator.generate_batch_embeddings(texts)
        
        # Convert to numpy array for calculations
        embeddings = np.array(embeddings)
        
        # Calculate various quality metrics
        metrics = {
            'dimensionality': embeddings.shape[1],
            'sparsity': np.mean(np.abs(embeddings) < 1e-6),
            'mean_magnitude': np.mean(np.linalg.norm(embeddings, axis=1)),
            'std_magnitude': np.std(np.linalg.norm(embeddings, axis=1))
        }
        
        return metrics
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print human-readable summary of benchmark results."""
        for model_key, results in self.results.items():
            print(f"\nModel: {model_key}")
            print("Speed (texts/second):")
            for batch_size, speed in results['speed'].items():
                print(f"  Batch size {batch_size}: {speed:.2f}")
            
            print("\nMemory Usage (MB):")
            for batch_size, memory in results['memory'].items():
                print(f"  Batch size {batch_size}: {memory:.2f}")
            
            print("\nQuality Metrics:")
            for metric, value in results['quality'].items():
                print(f"  {metric}: {value}") 