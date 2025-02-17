from .base import BaseDiagramAnalyzer
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ScientificAnalysisResult:
    """Structure for scientific analysis results"""
    notation: List[str]
    relationships: List[Dict]
    labels: List[Dict]
    chemical_structures: List[Dict]
    equations: List[Dict]
    confidence: float
    metadata: Dict

class ScientificDiagramAnalyzer(BaseDiagramAnalyzer):
    """Specialized analyzer for scientific diagrams."""
    
    def __init__(self):
        super().__init__()
        self._init_scientific_models()

    def _init_scientific_models(self):
        """Initialize scientific analysis models."""
        self._load_chemical_model()
        self._load_equation_model()
        self._load_notation_model() 