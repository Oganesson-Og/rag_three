"""
Metrics Collection Module
-----------------------

Handles metrics collection and tracking for document processing pipeline.

Key Features:
- Performance metrics tracking
- Processing time measurements
- Document statistics
- Error rate monitoring
- Resource usage tracking
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages processing metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.logger = logging.getLogger(__name__)

    def record(
        self,
        document_id: str,
        extractor: str,
        timestamp: datetime,
        **kwargs
    ) -> None:
        """Record processing metrics.
        
        Args:
            document_id: ID of processed document
            extractor: Name of extractor used
            timestamp: Time of processing
            **kwargs: Additional metrics to record
        """
        try:
            if document_id not in self.metrics:
                self.metrics[document_id] = []
                
            self.metrics[document_id].append({
                'extractor': extractor,
                'timestamp': timestamp.isoformat(),
                **kwargs
            })
            
        except Exception as e:
            self.logger.error(f"Failed to record metrics: {str(e)}")

    @contextmanager
    def measure_time(self, operation: str = "processing"):
        """Context manager to measure operation time.
        
        Args:
            operation: Name of operation being timed
        """
        start = time.time()
        try:
            yield self
        finally:
            elapsed = time.time() - start
            self.logger.debug(f"{operation} took {elapsed:.2f} seconds")

    def get_metrics(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Get recorded metrics.
        
        Args:
            document_id: Optional document ID to filter metrics
            
        Returns:
            Dictionary of recorded metrics
        """
        if document_id:
            return self.metrics.get(document_id, {})
        return self.metrics

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        try:
            total_documents = len(self.metrics)
            total_operations = sum(len(ops) for ops in self.metrics.values())
            
            return {
                'total_documents': total_documents,
                'total_operations': total_operations,
                'documents_processed': list(self.metrics.keys()),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate metrics summary: {str(e)}")
            return {} 