"""
Document Processing Package
------------------------

Comprehensive document processing system with extraction, analysis, and OCR capabilities.

Components:
- Extractors: Document content extraction
- Analyzers: Diagram and content analysis
- Processors: OCR and preprocessing
- Models: Shared data structures

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from .extractors import PDFExtractor, DocxExtractor, ExcelExtractor, CSVExtractor
from .analyzers import DiagramAnalyzerV2, ScientificDiagramAnalyzer
from .processors import OCRProcessor
from .models import ChunkingConfig, Chunk, PreprocessingConfig

__all__ = [
    # Extractors
    'PDFExtractor',
    'DocxExtractor',
    'ExcelExtractor',
    'CSVExtractor',
    # Analyzers
    'DiagramAnalyzerV2',
    'ScientificDiagramAnalyzer',
    # Processors
    'OCRProcessor',
    # Models
    'ChunkingConfig',
    'Chunk',
    'PreprocessingConfig'
]
