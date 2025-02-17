"""
Diagram Analysis Module
---------------------

Advanced diagram analysis system for scientific and educational content.
Processes technical diagrams, charts, and scientific illustrations.

Features:
- Scientific notation detection
- Relationship mapping
- Label identification
- Chemical structure recognition
- Equation extraction
- Component analysis
- Spatial relationship detection

Key Components:
1. Scientific Notation: Mathematical and technical symbols
2. Relationship Analysis: Component connections and flows
3. Label Detection: Text and annotation processing
4. Chemical Structure: Molecular and compound recognition
5. Equation Processing: Mathematical formula extraction

Technical Details:
- Computer vision algorithms
- OCR integration
- Graph-based analysis
- Pattern recognition
- Spatial analysis
- Symbol classification
- Neural network models

Dependencies:
- opencv-python>=4.8.0
- numpy>=1.24.0
- torch>=2.0.0
- scikit-image>=0.21.0

Author: Keith Satuku
Version: 1.5.0
Created: 2025
License: MIT
"""

import cv2
import numpy as np
import torch
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection, AutoConfig
from PIL import Image
import pytesseract
from rdkit import Chem
from rdkit.Chem import Draw
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import re
from numpy.typing import NDArray
from datetime import datetime
import io
from ultralytics import YOLO

# Type aliases
ImageArray = NDArray[np.uint8]
AnalysisResult = Dict[str, Any]

@dataclass
class AnalysisResult:
    """Structure for analysis results"""
    notation: List[str]
    relationships: List[Dict]
    labels: List[Dict]
    chemical_structures: List[Dict]
    equations: List[Dict]
    confidence: float
    metadata: Dict

class DiagramAnalyzer:
    """Handles diagram analysis with fallback options."""
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.processor = None
        
        try:
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForObjectDetection.from_pretrained(model_name)
        except Exception as e:
            self.logger.warning(f"Failed to initialize DETR model: {str(e)}")
            self.logger.info("Falling back to basic diagram processing")
            
    def process(self, image: Any) -> Dict[str, Any]:
        """Process diagram with fallback to basic analysis if DETR unavailable."""
        if self.model and self.processor:
            return self._process_with_detr(image)
        return self._process_basic(image)

    def analyze_image(
        self,
        image: Union[ImageArray, bytes],
        options: Optional[Dict[str, bool]] = None
    ) -> AnalysisResult:
        """Analyze diagram image."""
        try:
            options = options or {}
            
            # Convert bytes to image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
                image = np.array(image)
            
            # Ensure correct type
            if not isinstance(image, np.ndarray):
                raise TypeError("Image must be numpy array or bytes")
            
            result = {
                'type': self._detect_diagram_type(image),
                'elements': self._detect_elements(image),
                'metadata': {
                    'analyzer': self.__class__.__name__,
                    'timestamp': datetime.now().isoformat(),
                    'image_shape': image.shape,
                    'options': options
                }
            }
            
            if options.get('detect_text', True):
                result['text'] = self._extract_text(image)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Diagram analysis error: {str(e)}")
            raise

    def _detect_diagram_type(self, image: ImageArray) -> str:
        """Detect type of diagram."""
        # Implement diagram type detection
        return "unknown"

    def _detect_elements(self, image: ImageArray) -> List[Dict[str, Any]]:
        """Detect diagram elements."""
        # Implement element detection
        return []

    def _extract_text(self, image: ImageArray) -> List[Dict[str, Any]]:
        """Extract text from diagram."""
        # Implement text extraction
        return []

    def analyze_diagram(self, image: np.ndarray) -> AnalysisResult:
        """Main entry point for diagram analysis."""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Perform analysis
            results = {
                'notation': self.detect_scientific_notation(processed_image),
                'relationships': self.analyze_relationships(processed_image),
                'labels': self.identify_labels(processed_image),
                'chemical_structures': self.detect_chemical_structures(processed_image),
                'equations': self.extract_equations(processed_image)
            }
            
            # Calculate confidence
            confidence = self._calculate_confidence(results)
            
            # Add metadata
            metadata = self._generate_metadata(image, results)
            
            return AnalysisResult(
                **results,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing diagram: {str(e)}")
            raise

    def _load_notation_model(self):
        """
        Loads and configures Tesseract OCR with specific settings for scientific notation
        """
        try:
            # Configure Tesseract path if needed (especially on Windows)
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            # Test if Tesseract is working
            pytesseract.get_tesseract_version()
            
            # Configure custom parameters for scientific notation
            custom_config = {
                'oem': 3,  # OCR Engine Mode
                'psm': 6,  # Page Segmentation Mode (assume uniform block of text)
                'char_whitelist': '0123456789.×+-Ee',
                'language': 'eng'
            }
            
            return {
                'engine': pytesseract,
                'config': custom_config
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tesseract OCR: {str(e)}")

    def _load_relationship_model(self):
        """
        Loads DETR model for detecting arrows and relationships
        """
        try:
            # Move model to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            
            # Define relationship categories
            categories = {
                0: "arrow",
                1: "double_arrow",
                2: "dashed_arrow",
                3: "connection_line"
            }
            
            return {
                "device": device,
                "categories": categories
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load DETR model: {str(e)}")

    def _load_label_model(self):
        """
        Loads and configures YOLOv10 model for label detection
        """
        try:
            # Initialize YOLO model
            model = YOLO('yolov10.pt')  # or specific path to your model
            
            return {
                "model": model,
                "conf_threshold": 0.5,
                "ocr_engine": pytesseract
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def _load_chemical_model(self):
        """
        Loads and configures RDKit for chemical structure recognition
        """
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
        """
        Loads models for mathematical equation recognition
        """
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
            raise RuntimeError(f"Failed to load equation recognition model: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis."""
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

    def detect_scientific_notation(self, image: np.ndarray) -> List[str]:
        """Handles scientific notation recognition"""
        try:
            # Convert image to format suitable for Tesseract
            img_pil = Image.fromarray(image)
            
            # Configure Tesseract for scientific notation
            config = self._load_notation_model()['config']
            custom_config = f'--oem {config["oem"]} --psm {config["psm"]}'
            
            # Extract text
            text = self._load_notation_model()['engine'].image_to_string(
                img_pil, config=custom_config)
            
            # Process and validate scientific notation
            notations = []
            for potential_notation in text.split('\n'):
                if self._validate_scientific_notation(potential_notation):
                    notations.append(potential_notation)
            
            return notations
        except Exception as e:
            self.logger.error(f"Error detecting scientific notation: {str(e)}")
            return []

    def analyze_relationships(self, image: np.ndarray) -> List[Dict]:
        """Analyzes arrows and relationships between components"""
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            
            # Post-process to detect arrows and connections
            target_sizes = torch.tensor([image.shape[:2]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes)[0]
            
            relationships = []
            categories = self._load_relationship_model()["categories"]
            
            for score, label, box in zip(results["scores"], results["labels"], 
                                       results["boxes"]):
                if score > 0.7:  # Confidence threshold
                    relationships.append({
                        "type": categories.get(label.item(), "unknown"),
                        "box": box.tolist(),
                        "confidence": score.item()
                    })
            
            return relationships
        except Exception as e:
            self.logger.error(f"Error analyzing relationships: {str(e)}")
            return []

    def identify_labels(self, image: np.ndarray) -> List[Dict]:
        """Detects and extracts labeled parts using YOLOv10"""
        try:
            # Use YOLO for label detection
            model = self._load_label_model()["model"]
            results = model(image)[0]
            
            labels = []
            
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                
                if score > self._load_label_model()["conf_threshold"]:
                    # Extract text within the detected box
                    cropped = image[int(y1):int(y2), int(x1):int(x2)]
                    text = self._load_label_model()["ocr_engine"].image_to_string(cropped)
                    
                    labels.append({
                        "text": text.strip(),
                        "box": [x1, y1, x2, y2],
                        "confidence": score
                    })
            
            return labels
        except Exception as e:
            self.logger.error(f"Error identifying labels: {str(e)}")
            return []

    def detect_chemical_structures(self, image: np.ndarray) -> List[Dict]:
        """Recognizes chemical structures and formulas"""
        try:
            # Preprocess image for chemical structure detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours that might represent chemical structures
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            chemical_structures = []
            rdkit = self._load_chemical_model()["rdkit"]
            min_confidence = self._load_chemical_model()["min_confidence"]
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Size threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    structure_img = image[y:y+h, x:x+w]
                    
                    # Try to convert to SMILES using RDKit
                    try:
                        mol = rdkit.MolFromImage(structure_img)
                        if mol:
                            smiles = rdkit.MolToSmiles(mol)
                            chemical_structures.append({
                                "smiles": smiles,
                                "box": [x, y, w, h],
                                "confidence": min_confidence
                            })
                    except:
                        continue
            
            return chemical_structures
        except Exception as e:
            self.logger.error(f"Error detecting chemical structures: {str(e)}")
            return []

    def extract_equations(self, image: np.ndarray) -> List[Dict]:
        """Extracts mathematical equations from the diagram"""
        try:
            # Preprocess image for equation detection
            processed_image = self._load_equation_model()["preprocessor"](image)
            
            # Use specialized math OCR model
            config = self._load_equation_model()["config"]
            validator = self._load_equation_model()["validator"]
            postprocessor = self._load_equation_model()["postprocessor"]
            
            # Placeholder for actual equation recognition
            # You would need to implement or import actual model processing here
            equations = []
            
            # Post-process and validate equations
            for eq in equations:
                if validator(eq):
                    latex = postprocessor(eq)
                    equations.append({
                        "text": eq,
                        "latex": latex,
                        "confidence": config["confidence_threshold"]
                    })
            
            return equations
        except Exception as e:
            self.logger.error(f"Error extracting equations: {str(e)}")
            return []

    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence score."""
        confidence_weights = {
            'notation': 0.2,
            'relationships': 0.25,
            'labels': 0.25,
            'chemical_structures': 0.15,
            'equations': 0.15
        }
        
        confidence_scores = []
        for key, weight in confidence_weights.items():
            if results[key]:
                if isinstance(results[key][0], dict) and 'confidence' in results[key][0]:
                    scores = [r['confidence'] for r in results[key]]
                    confidence_scores.append(np.mean(scores) * weight)
                else:
                    confidence_scores.append(weight)
                    
        return sum(confidence_scores)

    def _generate_metadata(self, image: np.ndarray, results: Dict) -> Dict:
        """Generate metadata about the analysis."""
        return {
            'image_size': image.shape,
            'component_counts': {
                k: len(v) for k, v in results.items()
            },
            'processing_info': {
                'models_used': [
                    'scientific_notation',
                    'relationship_analyzer',
                    'label_detector',
                    'chemical_structure',
                    'equation_extractor'
                ],
                'device': str(next(self.model.parameters()).device)
            }
        }

    def _validate_scientific_notation(self, text: str) -> bool:
        """Helper method to validate scientific notation"""
        pattern = r'^[+-]?\d*\.?\d+[Ee][+-]?\d+$'
        return bool(re.match(pattern, text.strip()))

    def _preprocess_for_math(self, image: np.ndarray) -> np.ndarray:
        """Helper method to preprocess image for math detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        return binary

    def _convert_to_latex(self, text: str) -> str:
        """Helper method to convert extracted text to LaTeX format"""
        # Implement LaTeX conversion logic here
        return text

    def process_chemical_diagram(self, image: NDArray) -> Optional[str]:
        """Process chemical diagram to SMILES."""
        try:
            mol = Chem.MolFromImage(image)
            if mol is not None:
                return Chem.MolToSmiles(mol)
            return None
        except Exception as e:
            self.logger.error("Error processing chemical diagram: %s", str(e))
            return None

    def process_equations(self, equations: List[str]) -> List[str]:
        """Process mathematical equations."""
        try:
            equations_copy = equations.copy()
            for i, eq in enumerate(equations_copy):
                # Process equation
                pass
        except Exception as e:
            self.logger.error("Error processing equations: %s", str(e))
            return equations

 
    
