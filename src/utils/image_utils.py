"""
Image Utils Module
--------------
"""

from typing import Dict, Optional, Union, Any
import numpy as np
from numpy.typing import NDArray
import cv2
from PIL import Image
import logging
from datetime import datetime

# Type aliases
ImageType = Union[NDArray[np.uint8], Image.Image]
ProcessingResult = Dict[str, Any]

class ImageUtils:
    """Image processing utilities."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def convert_to_grayscale(
        self,
        image: ImageType,
        options: Optional[Dict[str, bool]] = None
    ) -> ImageType:
        """Convert image to grayscale."""
        try:
            options = options or {}
            
            if isinstance(image, Image.Image):
                return image.convert('L')
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        except Exception as e:
            self.logger.error(f"Grayscale conversion error: {str(e)}")
            raise

    def resize_image(
        self,
        image: NDArray[np.uint8],
        scale_factor: float,
        options: Optional[Dict[str, bool]] = None
    ) -> NDArray[np.uint8]:
        """Resize image by scale factor."""
        try:
            options = options or {}
            
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            return cv2.resize(image, (width, height))
            
        except Exception as e:
            self.logger.error(f"Image resize error: {str(e)}")
            raise

    def enhance_contrast(
        self,
        image: NDArray[np.uint8],
        options: Optional[Dict[str, bool]] = None
    ) -> NDArray[np.uint8]:
        """Enhance image contrast."""
        try:
            options = options or {}
            
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=options.get('clip_limit', 3.0),
                tileGridSize=options.get('tile_size', (8,8))
            )
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            self.logger.error(f"Contrast enhancement error: {str(e)}")
            raise

    def remove_noise(
        self,
        image: NDArray[np.uint8],
        strength: int = 7,
        options: Optional[Dict[str, bool]] = None
    ) -> NDArray[np.uint8]:
        """Remove image noise."""
        try:
            options = options or {}
            return cv2.fastNlMeansDenoising(image, None, strength)
            
        except Exception as e:
            self.logger.error(f"Noise removal error: {str(e)}")
            raise

    def process_image(
        self,
        image: ImageType,
        options: Optional[Dict[str, bool]] = None
    ) -> ProcessingResult:
        """Process image with multiple operations."""
        try:
            options = options or {}
            
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            result = {
                'original_shape': image.shape,
                'metadata': {
                    'processor': self.__class__.__name__,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Apply processing steps
            if options.get('grayscale', False):
                image = self.convert_to_grayscale(image)
                result['grayscale'] = True
                
            if options.get('resize', False):
                scale = options.get('scale_factor', 1.0)
                image = self.resize_image(image, scale)
                result['resized'] = True
                
            if options.get('enhance_contrast', False):
                image = self.enhance_contrast(image)
                result['contrast_enhanced'] = True
                
            if options.get('remove_noise', False):
                strength = options.get('noise_strength', 7)
                image = self.remove_noise(image, strength)
                result['noise_removed'] = True
                
            result['processed_image'] = image
            result['final_shape'] = image.shape
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            raise