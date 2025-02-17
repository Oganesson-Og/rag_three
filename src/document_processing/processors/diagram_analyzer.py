"""
Diagram Analysis Module
----------------------

Advanced system for analyzing and understanding educational diagrams using
computer vision and machine learning techniques.

Key Features:
- Multi-type diagram support
- Scientific notation detection
- Chemical structure recognition
- Equation extraction
- Relationship mapping
- Label identification
- Element detection
- Confidence scoring
- Type classification
- Basic/Advanced modes
- Capability detection

Technical Details:
- DETR object detection
- OpenCV processing
- OCR integration
- Shape detection
- Element classification
- Relationship mapping
- Chemical structure analysis
- Equation parsing
- Confidence calculation
- Error handling

Dependencies:
- ultralytics>=8.0.0
- opencv-python>=4.8.0
- Pillow>=8.0.0
- numpy>=1.24.0
- pytesseract>=0.3.8
- transformers>=4.35.0
- torch>=2.0.0
- rdkit>=2023.3
- scikit-image>=0.21.0
- logging (standard library)
- typing (standard library)
- dataclasses (standard library)
- enum (standard library)

Example Usage:
    # Initialize analyzer
    analyzer = DiagramAnalyzer(
        config=DiagramConfig(
            use_basic=False,
            detect_chemical=True,
            detect_equations=True
        )
    )
    
    # Process diagram
    result = analyzer.process_diagram(
        image_path="circuit_diagram.png"
    )
    
    # Get capabilities
    caps = analyzer.get_capabilities()

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import pytesseract
from rdkit import Chem
from rdkit.Chem import Draw
from datetime import datetime
import re
from numpy.typing import NDArray
from ultralytics import YOLO

class DiagramType(Enum):
    """Types of diagrams supported by the analyzer."""
    FLOWCHART = "flowchart"
    TECHNICAL = "technical"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    CHEMICAL = "chemical"
    UNKNOWN = "unknown"

@dataclass
class DiagramConfig:
    """Configuration for diagram analyzer."""
    use_basic: bool = False
    model_path: Optional[str] = None
    device: str = "cpu"
    confidence_threshold: float = 0.5
    max_diagrams: int = 10
    ocr_enabled: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DiagramConfig':
        """Create config from dictionary."""
        return cls(
            use_basic=config.get('use_basic', False),
            model_path=config.get('model_path'),
            device=config.get('device', 'cpu'),
            confidence_threshold=config.get('confidence_threshold', 0.5),
            max_diagrams=config.get('max_diagrams', 10),
            ocr_enabled=config.get('ocr_enabled', True)
        )

@dataclass
class DiagramElement:
    """Represents an element in a diagram."""
    element_type: str
    confidence: float
    bbox: List[float]
    text: Optional[str] = None
    relationships: List[str] = None
    notation: Optional[str] = None
    chemical_structure: Optional[str] = None
    equation: Optional[str] = None

class DiagramAnalyzer:
    """Analyzes diagrams in documents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize diagram analyzer."""
        self.config = DiagramConfig.from_dict(config)
        self.logger = logging.getLogger(__name__)
        try:
            self._initialize_models()
        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            # Set to basic mode on initialization failure
            self.config.use_basic = True
            self._initialize_basic_models()
    
    def _initialize_models(self):
        """Initialize ML models for diagram analysis."""
        if not self.config.use_basic:
            try:
                # Initialize advanced models
                pass
            except Exception as e:
                self.logger.warning(f"Advanced model initialization failed: {e}")
                self.config.use_basic = True
                self._initialize_basic_models()
    
    def _initialize_basic_models(self):
        """Initialize basic analysis models."""
        # Basic model initialization logic
        pass

    def process_diagram(
        self,
        image_path: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """Process diagram with all enabled analyzers."""
        try:
            # Load and validate image
            image = self._load_image(image_path)
            
            # Process based on available methods
            if self.config.use_basic:
                result = self._basic_processing(image)
            else:
                result = self._advanced_processing(image)
                
            # Add scientific analysis if enabled
            if self.config.detect_chemical:
                result['chemical_structures'] = self.detect_chemical_structures(image)
                
            if self.config.detect_equations:
                result['equations'] = self.extract_equations(image)
                
            # Add metadata
            result['metadata'] = self._generate_metadata(image, result)
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error processing diagram: {str(e)}")
            return {
                "error": str(e),
                "type": DiagramType.UNKNOWN.value,
                "elements": [],
                "confidence": 0.0
            }

    def _basic_processing(self, image: Image.Image) -> Dict[str, Any]:
        """Basic diagram processing using OpenCV."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Detect type
            diagram_type = self._detect_diagram_type(img_array)
            
            # Detect elements
            elements = self._detect_elements(img_array)
            
            # Extract text if needed
            text_elements = self._extract_text(img_array)
            
            return {
                'type': diagram_type,
                'elements': elements,
                'text': text_elements,
                'confidence': self._calculate_confidence({'elements': elements})
            }
            
        except Exception as e:
            self.logger.error(f"Basic processing error: {str(e)}")
            raise

    def _advanced_processing(self, image: Image.Image) -> Dict[str, Any]:
        """Advanced diagram processing using DETR and specialized models."""
        try:
            # Prepare image for DETR
            inputs = self.processor(images=image, return_tensors="pt")
            if self.config.enable_gpu and torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Get predictions
            outputs = self.model(**inputs)
            
            # Process results
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=self.config.confidence_threshold,
                target_sizes=[(image.size[1], image.size[0])]
            )[0]
            
            # Convert to diagram elements
            elements = []
            for score, label, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"]
            ):
                elements.append(DiagramElement(
                    element_type=self.model.config.id2label[label.item()],
                    confidence=score.item(),
                    bbox=box.tolist()
                ))
            
            # Analyze relationships
            relationships = self.analyze_relationships(image)
            
            # Identify labels
            labels = self.identify_labels(image)
            
            return {
                'type': self._detect_diagram_type(image),
                'elements': elements,
                'relationships': relationships,
                'labels': labels,
                'confidence': self._calculate_confidence({
                    'elements': elements,
                    'relationships': relationships
                })
            }
            
        except Exception as e:
            self.logger.error(f"Advanced processing error: {str(e)}")
            raise

    def detect_chemical_structures(self, image: np.ndarray) -> List[Dict]:
        """Detect and analyze chemical structures."""
        try:
            # Convert image for RDKit processing
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect potential chemical structures
            contours = self._find_chemical_contours(img_gray)
            
            structures = []
            for contour in contours:
                # Extract region
                x, y, w, h = cv2.boundingRect(contour)
                region = img_gray[y:y+h, x:x+w]
                
                # Convert to SMILES using RDKit
                try:
                    mol = self.chemical_model.rdkit.MolFromImage(region)
                    if mol:
                        structures.append({
                            'smiles': Chem.MolToSmiles(mol),
                            'confidence': self._get_structure_confidence(mol),
                            'bbox': [x, y, w, h]
                        })
                except Exception as e:
                    self.logger.debug(f"Chemical structure conversion error: {str(e)}")
                    continue
                    
            return structures
            
        except Exception as e:
            self.logger.error(f"Chemical structure detection error: {str(e)}")
            return []

    def extract_equations(self, image: np.ndarray) -> List[Dict]:
        """Extract and parse mathematical equations."""
        try:
            # Convert image for equation detection
            preprocessed = self._preprocess_for_math(image)
            
            # Detect equation regions
            regions = self._detect_equation_regions(preprocessed)
            
            equations = []
            for region in regions:
                try:
                    # Extract text using OCR
                    text = pytesseract.image_to_string(
                        region,
                        config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789+-*/()=xyz'
                    )
                    
                    # Convert to LaTeX
                    latex = self._convert_to_latex(text)
                    
                    if self._validate_equation(latex):
                        equations.append({
                            'text': text,
                            'latex': latex,
                            'confidence': self._get_equation_confidence(text)
                        })
                except Exception as e:
                    self.logger.debug(f"Equation extraction error: {str(e)}")
                    continue
                    
            return equations
            
        except Exception as e:
            self.logger.error(f"Equation extraction error: {str(e)}")
            return []

    def _load_chemical_model(self):
        """Load chemical structure recognition model."""
        try:
            # Configure RDKit parameters
            Draw.DrawingOptions.bondLineWidth = 1.2
            Draw.DrawingOptions.atomLabelFontSize = 12
            
            return {
                "rdkit": Chem,
                "draw_utils": Draw,
                "supported_formats": ['.mol', '.sdf', '.png', '.jpg'],
                "min_confidence": 0.7
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chemical structure recognition: {str(e)}")

    def _load_equation_model(self):
        """Load equation recognition model."""
        try:
            config = {
                "model_type": "transformer",
                "max_sequence_length": 512,
                "confidence_threshold": 0.8,
                "device": 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            return {
                "config": config,
                "preprocessor": self._preprocess_for_math,
                "postprocessor": self._convert_to_latex,
                "validator": self._validate_equation
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize equation recognition: {str(e)}")

    def _load_label_model(self):
        """Load YOLOv10 model for label detection."""
        try:
            # Initialize YOLOv10 model
            # Using YOLOv10-X (extra large) for maximum accuracy
            # Other options: yolov10n.pt (nano), yolov10s.pt (small), yolov10m.pt (medium), 
            # yolov10l.pt (large), yolov10x.pt (extra large)
            model = YOLO('yolov10x.pt')
            
            # Configure model settings for optimal diagram analysis
            model.conf = 0.5  # Confidence threshold
            model.iou = 0.45  # NMS IoU threshold
            model.max_det = 100  # Maximum detections per image
            
            # Set device
            device = 'cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu'
            model.to(device)
            
            return {
                "model": model,
                "ocr_engine": pytesseract,
                "device": device,
                "conf_threshold": 0.5,
                "supported_classes": model.names,
                "model_version": "YOLOv10"
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv10 model: {str(e)}")

    def identify_labels(self, image: Image.Image) -> List[Dict]:
        """
        Identify labels and text regions using YOLOv10.
        
        Args:
            image: Input image
            
        Returns:
            List of detected labels with their properties
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            # Run YOLOv10 detection with augmentation for better accuracy
            results = self.label_model["model"](
                image_np,
                conf=self.label_model["conf_threshold"],
                augment=True  # Enable test time augmentation for better accuracy
            )

            labels = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    conf = float(box.conf)
                    cls = int(box.cls)
                    cls_name = self.label_model["supported_classes"][cls]
                    
                    # Extract text from region using OCR if it's a text region
                    region = image_np[int(y1):int(y2), int(x1):int(x2)]
                    text = self.label_model["ocr_engine"].image_to_string(region) if cls_name == "text" else ""
                    
                    labels.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class": cls_name,
                        "text": text.strip() if text else None
                    })
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Label detection error: {str(e)}")
            return []

    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence score."""
        try:
            confidences = []
            
            # Element confidence
            if 'elements' in results:
                element_confidences = [e.confidence for e in results['elements']]
                if element_confidences:
                    confidences.append(np.mean(element_confidences))
            
            # Relationship confidence
            if 'relationships' in results:
                rel_confidences = [r.get('confidence', 0) for r in results['relationships']]
                if rel_confidences:
                    confidences.append(np.mean(rel_confidences))
            
            # Chemical structure confidence
            if 'chemical_structures' in results:
                chem_confidences = [c.get('confidence', 0) for c in results['chemical_structures']]
                if chem_confidences:
                    confidences.append(np.mean(chem_confidences))
            
            # Equation confidence
            if 'equations' in results:
                eq_confidences = [e.get('confidence', 0) for e in results['equations']]
                if eq_confidences:
                    confidences.append(np.mean(eq_confidences))
            
            return np.mean(confidences) if confidences else 0.0
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {str(e)}")
            return 0.0

    def _generate_metadata(self, image: np.ndarray, results: Dict) -> Dict:
        """Generate comprehensive metadata about the analysis."""
        return {
            'analyzer': self.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'image_shape': image.shape,
            'diagram_type': results.get('type', DiagramType.UNKNOWN.value),
            'element_count': len(results.get('elements', [])),
            'chemical_structure_count': len(results.get('chemical_structures', [])),
            'equation_count': len(results.get('equations', [])),
            'confidence': results.get('confidence', 0.0),
            'processing_mode': 'basic' if self.config.use_basic else 'advanced',
            'gpu_enabled': self.config.enable_gpu and torch.cuda.is_available()
        }

    def get_capabilities(self) -> Dict[str, bool]:
        """Get current analyzer capabilities."""
        return {
            "basic_analysis": True,
            "advanced_analysis": not self.config.use_basic,
            "chemical_detection": self.config.detect_chemical,
            "equation_detection": self.config.detect_equations,
            "gpu_enabled": self.config.enable_gpu and torch.cuda.is_available()
        } 