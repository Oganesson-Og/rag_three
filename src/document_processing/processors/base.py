"""
Base Processor Module
------------------

Abstract base class for document processors with shared functionality.

Key Features:
- Common processor interfaces
- Error handling
- Status tracking
- Resource management
- Configuration handling
- Progress monitoring
- Validation utilities

Technical Details:
- Abstract methods
- Type annotations
- Error management
- Resource cleanup
- Configuration validation
- Progress tracking
- Status reporting

Dependencies:
- abc (standard library)
- typing (standard library)
- logging (standard library)
- dataclasses (standard library)

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
from PIL import Image

@dataclass
class ProcessorConfig:
    """Base configuration for processors."""
    batch_size: int = 32
    timeout: int = 300
    max_retries: int = 3
    device: str = "cpu"
    debug: bool = False
    cache_enabled: bool = True
    cache_dir: Optional[str] = None

@dataclass
class ProcessingResult:
    """Standard result structure for processors."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    errors: List[str] = None
    warnings: List[str] = None

class BaseProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_processor()
        
    def _initialize_processor(self):
        """Initialize processor resources."""
        try:
            self._validate_config()
            self._setup_resources()
            self._initialized = True
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self._initialized = False
            raise

    def _validate_config(self):
        """Validate processor configuration."""
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.config.timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self.config.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    @abstractmethod
    def _setup_resources(self):
        """Set up processor-specific resources."""
        pass

    @abstractmethod
    def process(
        self,
        content: Union[str, Path, bytes, Image.Image, np.ndarray],
        options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process content with specified options."""
        pass

    def batch_process(
        self,
        contents: List[Union[str, Path, bytes, Image.Image, np.ndarray]],
        options: Optional[Dict[str, Any]] = None
    ) -> List[ProcessingResult]:
        """Process multiple items in batches."""
        results = []
        for i in range(0, len(contents), self.config.batch_size):
            batch = contents[i:i + self.config.batch_size]
            batch_results = []
            for content in batch:
                try:
                    result = self.process(content, options)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {str(e)}")
                    batch_results.append(ProcessingResult(
                        success=False,
                        data=None,
                        metadata={},
                        errors=[str(e)]
                    ))
            results.extend(batch_results)
        return results

    def cleanup(self):
        """Clean up processor resources."""
        try:
            self._cleanup_resources()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise

    @abstractmethod
    def _cleanup_resources(self):
        """Clean up processor-specific resources."""
        pass

    def validate_input(
        self,
        content: Union[str, Path, bytes, Image.Image, np.ndarray]
    ) -> bool:
        """Validate input content."""
        if content is None:
            return False
            
        if isinstance(content, (str, Path)):
            return Path(content).exists()
            
        if isinstance(content, bytes):
            return len(content) > 0
            
        if isinstance(content, Image.Image):
            return content.size[0] > 0 and content.size[1] > 0
            
        if isinstance(content, np.ndarray):
            return content.size > 0
            
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            'initialized': self._initialized,
            'config': self.config.__dict__,
            'device': self.config.device,
            'cache_enabled': self.config.cache_enabled,
            'cache_dir': self.config.cache_dir
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup() 