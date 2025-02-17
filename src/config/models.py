"""
Configuration Models Module
-------------------------

Pydantic models for managing system-wide configuration settings in the
educational RAG system.

Key Features:
- Model configuration management
- Processing settings control
- Retrieval parameter handling
- Storage path configuration
- Fallback model support
- Resource optimization
- Configuration validation

Technical Details:
- Dataclass-based models
- Type validation
- Default value handling
- Configuration inheritance
- Dynamic settings
- Resource management
- Path handling

Dependencies:
- dataclasses (standard library)
- typing (standard library)
- pathlib (standard library)

Example Usage:
    # Basic configuration
    model_config = ModelConfig(
        primary_model="gpt-3.5-turbo",
        fallback_model="gpt-3.5-turbo-instruct",
        device="cuda"
    )
    
    # Complete RAG configuration
    rag_config = RAGConfig(
        model=model_config,
        processing=ProcessingConfig(chunk_size=1000),
        retrieval=RetrievalConfig(top_k=5),
        storage=StorageConfig()
    )

Configuration Categories:
- Model Settings
- Processing Parameters
- Retrieval Options
- Storage Locations

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """
    Enhanced model configuration with fallbacks.
    
    Attributes:
        primary_model (str): Main model identifier
        fallback_model (Optional[str]): Backup model identifier
        device (str): Processing device (cpu/cuda)
        batch_size (int): Processing batch size
        max_length (int): Maximum sequence length
        use_fallback (bool): Whether to use fallback model
    """
    primary_model: str
    fallback_model: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    use_fallback: bool = False
    
    def get_active_model(self) -> str:
        """Get the currently active model name."""
        return self.fallback_model if self.use_fallback else self.primary_model

@dataclass
class ProcessingConfig:
    """
    Document processing configuration.
    
    Attributes:
        chunk_size (int): Size of document chunks
        chunk_overlap (int): Overlap between chunks
        min_chunk_size (int): Minimum chunk size
        max_chunks_per_doc (int): Maximum chunks per document
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunks_per_doc: int = 50

@dataclass
class RetrievalConfig:
    """
    Retrieval configuration settings.
    
    Attributes:
        top_k (int): Number of top results to retrieve
        score_threshold (float): Minimum similarity score
        semantic_weight (float): Weight for semantic search
        keyword_weight (float): Weight for keyword search
    """
    top_k: int = 5
    score_threshold: float = 0.7
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

@dataclass
class StorageConfig:
    """
    Storage path configuration.
    
    Attributes:
        vector_store_path (str): Path to vector storage
        document_store_path (str): Path to document storage
        cache_dir (str): Path to cache directory
    """
    vector_store_path: str = "vectors"
    document_store_path: str = "documents"
    cache_dir: str = "cache"

@dataclass
class RAGConfig:
    """
    Complete RAG system configuration.
    
    Attributes:
        model (ModelConfig): Model configuration
        processing (ProcessingConfig): Processing settings
        retrieval (RetrievalConfig): Retrieval parameters
        storage (StorageConfig): Storage locations
    """
    model: ModelConfig
    processing: ProcessingConfig
    retrieval: RetrievalConfig
    storage: StorageConfig 