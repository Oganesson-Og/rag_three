"""
RAG (Retrieval-Augmented Generation) Module
------------------------------------------

A comprehensive RAG implementation with educational content focus, providing
document processing, embedding management, and response generation capabilities.

Key Components:
- Pipeline: Main RAG pipeline implementation
- Document/Chunk Models: Core data structures
- Prompt Engineering: Advanced prompt management
- Processing Stages: Modular document processing

Example Usage:
    from rag import Pipeline, Document, ContentModality
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Process document
    document = await pipeline.process_document(
        source="path/to/doc",
        modality=ContentModality.TEXT
    )
    
    # Generate response
    result = await pipeline.generate_response(
        query="Sample query",
        context={"subject": "math"}
    )

Author: Keith Satuku
Version: 1.0.0
License: MIT
"""

from .pipeline import Pipeline
from .models import (
    Document,
    Chunk,
    ProcessingStage,
    ContentModality,
    ProcessingEvent,
    ProcessingMetrics,
    SearchResult,
    GenerationResult,
    EmbeddingVector
)
from .prompt_engineering import PromptGenerator, PromptTemplate
from .utils import num_tokens_from_string, rmSpace

__version__ = "1.0.0"
__author__ = "Keith Satuku"

__all__ = [
    # Main pipeline
    "Pipeline",
    
    # Core models
    "Document",
    "Chunk",
    "ProcessingStage",
    "ContentModality",
    "ProcessingEvent",
    "ProcessingMetrics",
    "SearchResult",
    "GenerationResult",
    "EmbeddingVector",
    
    # Prompt engineering
    "PromptGenerator",
    "PromptTemplate",
    
    # Utilities
    "num_tokens_from_string",
    "rmSpace",
]
