from typing import Optional, Dict
from PIL import Image
import io
import logging
from ...database.models import DocumentDB
from ...rag.models import Document as PydanticDocument
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)

class ImageExtractor(BaseExtractor):
    """Handles image document extraction with OCR capabilities."""
    
    async def extract(self, document: 'DocumentDB') -> 'PydanticDocument':
        """Extract content from image document.
        
        Args:
            document: Document instance containing image data
            
        Returns:
            Updated document with extracted text and metadata
        """
        try:
            # Get content bytes from document
            content = document.content
            if not content:
                raise ValueError("Document has no content")

            # Convert bytes to image
            image = Image.open(io.BytesIO(content))
            
            # Use OCR to extract text
            from ..core.vision.ocr import OCR
            ocr = OCR()
            extracted_text = ocr.process_image(image)
            
            # Update document metadata
            document.doc_info.update({
                'extracted_text': extracted_text,
                'image_metadata': {
                    'width': image.width,
                    'height': image.height,
                    'format': image.format,
                    'mode': image.mode
                },
                'extraction_method': 'ocr'
            })
            
            return document
            
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            raise