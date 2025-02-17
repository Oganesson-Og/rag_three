from .base import BaseDiagramAnalyzer, DiagramElement
from enum import Enum
from typing import Optional
from pathlib import Path

class DiagramType(Enum):
    """Types of diagrams supported by the analyzer."""
    FLOWCHART = "flowchart"
    TECHNICAL = "technical"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    UNKNOWN = "unknown"

class DiagramAnalyzerV2(BaseDiagramAnalyzer):
    """Enhanced diagram analyzer with multiple processing options."""
    
    def __init__(self, config_path: Optional[Path] = None, use_basic: bool = False):
        super().__init__()
        self.use_basic = use_basic
        
        # Initialize components
        self._init_ocr()
        self._init_models()

    def _init_ocr(self):
        """Initialize OCR component."""
        try:
            import pytesseract
            self.ocr = pytesseract
        except ImportError:
            self.logger.warning("Tesseract OCR not available")
            self.ocr = None

    def _init_models(self):
        """Initialize detection models."""
        if not self.use_basic:
            try:
                from transformers import DetrImageProcessor, DetrForObjectDetection
                self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            except Exception as e:
                self.logger.warning(f"Advanced models not available: {str(e)}")
                self.use_basic = True 