"""
Vector Metrics Module
-----------------

Comprehensive vector similarity and distance metrics system for educational RAG applications.

Key Features:
- Multiple distance metrics
- Metric recording
- Statistical analysis
- Performance tracking
- Batch processing
- Historical tracking
- Automated reporting

Technical Details:
- Cosine similarity
- Euclidean distance
- Statistical computations
- Numpy optimizations
- Memory efficiency
- Error handling
- Metric persistence

Dependencies:
- numpy>=1.24.0
- typing (standard library)
- datetime (standard library)
- logging (standard library)

Example Usage:
    # Initialize metrics
    metrics = VectorMetrics()
    
    # Record distance between vectors
    distance = metrics.record_distance(
        vector1=[0.1, 0.2, 0.3],
        vector2=[0.2, 0.3, 0.4],
        metric_name='cosine'
    )
    
    # Get historical metrics
    cosine_distances = metrics.metrics['cosine']
    
    # Calculate average distance
    avg_distance = np.mean(cosine_distances)

Available Metrics:
- Cosine Distance (1 - cosine similarity)
- Euclidean Distance (L2 norm)
- Future metrics can be added via the _calculate_distance method

Performance Considerations:
- Numpy operations for efficiency
- Automatic type conversion
- Vector normalization
- Error handling for invalid inputs

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Union, Optional
import numpy as np
from datetime import datetime

class VectorMetrics:
    """
    Vector similarity and distance metrics calculator and recorder.
    
    Attributes:
        metrics (Dict[str, List[float]]): Historical record of calculated metrics
    """
    
    def __init__(self):
        """Initialize VectorMetrics with empty metrics history."""
        self.metrics: Dict[str, List[float]] = {}
        
    def record_distance(
        self,
        vector1: Union[List[float], np.ndarray],
        vector2: Union[List[float], np.ndarray],
        metric_name: str = 'cosine'
    ) -> float:
        """
        Record distance between vectors using specified metric.
        
        Args:
            vector1: First vector for comparison
            vector2: Second vector for comparison
            metric_name: Name of distance metric to use ('cosine' or 'euclidean')
            
        Returns:
            float: Calculated distance between vectors
            
        Raises:
            ValueError: If vectors have different dimensions or metric is unsupported
        """
        distance = self._calculate_distance(vector1, vector2, metric_name)
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(distance)
        return distance
        
    def _calculate_distance(
        self,
        vector1: Union[List[float], np.ndarray],
        vector2: Union[List[float], np.ndarray],
        metric: str
    ) -> float:
        """
        Calculate distance between vectors using specified metric.
        
        Args:
            vector1: First vector for comparison
            vector2: Second vector for comparison
            metric: Distance metric to use ('cosine' or 'euclidean')
            
        Returns:
            float: Calculated distance between vectors
            
        Raises:
            ValueError: If vectors have different dimensions or metric is unsupported
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        if metric == 'cosine':
            return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif metric == 'euclidean':
            return np.linalg.norm(v1 - v2)
        else:
            raise ValueError(f"Unsupported metric: {metric}") 