"""
Document Processing Processors Package
-----------------------------------

Collection of document processing modules for the RAG pipeline.

Available Processors:
- OCRProcessor: Text extraction and OCR processing
- DiagramAnalyzer: Scientific diagram analysis
- TableStructureRecognizer: Table detection and structure analysis
- BaseProcessor: Abstract base class for processors

Key Features:
- Modular processor architecture
- Standardized interfaces
- Error handling
- Resource management
- Progress tracking
- Batch processing
- Configuration management

Usage:
    from rag_pipeline.src.document_processing.processors import (
        OCRProcessor,
        DiagramAnalyzer,
        TableStructureRecognizer
    )

    # Initialize processor
    ocr = OCRProcessor()
    
    # Process document
    result = ocr.process("document.pdf")

Version: 1.0.0
Author: Keith Satuku
License: MIT
"""

from .base import BaseProcessor, ProcessorConfig, ProcessingResult
from .ocr import OCRProcessor, OCRConfig
from .math_processor import MathProcessor, MathProcessorConfig
from .diagram_analyzer import DiagramAnalyzer, DiagramConfig
from .diagram_archive import DiagramAnalyzer as DiagramAnalyzerLegacy

__all__ = [
    # Base classes
    'BaseProcessor',
    'ProcessorConfig',
    'ProcessingResult',
    
    # OCR processor
    'OCRProcessor',
    'OCRConfig',
    'MathProcessor',
    # Diagram processors
    'DiagramAnalyzer',
    'DiagramConfig',
    'DiagramAnalyzerLegacy',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Keith Satuku'
__license__ = 'MIT'
