from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import numpy as np
from PIL import Image
import logging
import cv2

@dataclass
class DiagramElement:
    """Base class for diagram elements."""
    element_type: str
    confidence: float
    bbox: List[float]
    text: Optional[str] = None
    relationships: List[str] = None

class BaseDiagramAnalyzer(ABC):
    """Base class for diagram analyzers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def process_diagram(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """Process diagram with available methods."""
        pass

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Common image preprocessing."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return binary
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise 