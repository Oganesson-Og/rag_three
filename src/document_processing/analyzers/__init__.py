"""
Document Analyzers Package
-----------------------

Collection of specialized document content analyzers.

Available Analyzers:
- DiagramAnalyzerV2: Enhanced diagram analysis
- ScientificDiagramAnalyzer: Scientific diagram processing
- BaseDiagramAnalyzer: Abstract base class

Usage:
    from document_processing.analyzers import DiagramAnalyzerV2
    
    analyzer = DiagramAnalyzerV2()
    result = analyzer.process_diagram('diagram.png')

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from .base import BaseDiagramAnalyzer, DiagramElement
from .diagram_analyzer_v2 import DiagramAnalyzerV2, DiagramType
from .scientific_diagram_analyzer import ScientificDiagramAnalyzer, ScientificAnalysisResult

__all__ = [
    'BaseDiagramAnalyzer',
    'DiagramAnalyzerV2',
    'ScientificDiagramAnalyzer',
    'DiagramElement',
    'DiagramType',
    'ScientificAnalysisResult'
] 