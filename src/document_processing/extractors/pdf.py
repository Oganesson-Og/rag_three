"""
Enhanced PDF Document Extractor Module
----------------------------------

Advanced PDF processing system with layout recognition and structural analysis.

Key Features:
- Layout-aware extraction
- Hierarchical content detection
- Cross-page content handling
- Smart text merging
- Title/subtitle recognition
- Table structure detection
- Image extraction with context
- Zoom-based processing

Technical Details:
- PyMuPDF integration
- Layout recognition
- Structure preservation
- Content hierarchy
- Smart merging
- OCR capabilities
- Error handling
- Performance optimization

Dependencies:
- PyMuPDF>=1.18.0
- pytesseract>=0.3.8
- numpy>=1.24.0
- Pillow>=8.0.0
- xgboost>=1.7.0
- torch>=2.0.0

Author: Keith Satuku
Version: 2.1.0
Created: 2025
License: MIT
"""

import fitz
from PIL import Image
import pytesseract
import numpy as np
import torch
import re
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass
from .models import Document
from .base import BaseExtractor, ExtractorResult

@dataclass
class LayoutElement:
    """Structure for layout elements."""
    type: str
    text: str
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    is_bold: bool
    in_row: int = 1
    row_height: float = 0
    is_row_header: bool = False
    confidence: float = 1.0

class PDFExtractor(BaseExtractor):
    """Enhanced PDF extractor with layout awareness."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PDF extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        default_config = {
            'zoom_factor': 2.0,
            'title_font_scale': 1.2,
            'merge_tolerance': 0.5,
            'min_confidence': 0.7,
            'line_spacing': 1.2,
            'row_tolerance': 5,
            'char_spacing': 0.1
        }
        
        # Merge provided config with defaults
        if config:
            default_config.update(config)
            
        super().__init__(config=default_config)
        self._init_components()
        self.base_font_size = 12.0  # Will be updated during processing

    def _init_components(self):
        """Initialize processing components."""
        try:
            self._init_ocr()
            self._init_layout_recognizer()
            self._init_device()
        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise

    def _init_layout_recognizer(self):
        """Initialize layout recognition capabilities."""
        try:
            from ..core.vision.layout_recognizer import LayoutRecognizer
            
            # Get layout config
            layout_config = self.config.get('layout', {})
            
            self.layout_recognizer = LayoutRecognizer(
                model_dir=layout_config.get('model_dir', ""),
                device=layout_config.get('device', "cuda"),
                batch_size=layout_config.get('batch_size', 32),
                cache_dir=layout_config.get('cache_dir'),
                model_type=layout_config.get('model_type', 'yolov10'),
                confidence=layout_config.get('confidence', 0.5),
                merge_boxes=layout_config.get('merge_boxes', True),
                label_list=layout_config.get('label_list', [
                    "title", "text", "list", "table", "figure", 
                    "header", "footer", "sidebar", "caption"
                ]),
                task_name=layout_config.get('task_name', 'document_layout')
            )
            self.has_layout_recognition = True
            self.logger.info("Layout recognition initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Layout recognition not available: {str(e)}")
            self.has_layout_recognition = False

    def _init_device(self):
        """Initialize device settings."""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except Exception:
            self.device = torch.device('cpu')

    def _init_ocr(self):
        """Initialize OCR capabilities."""
        try:
            from ..core.vision.ocr import OCR
            self.ocr = OCR()
            self.has_ocr = True
            self.logger.info("OCR initialized successfully")
        except Exception as e:
            self.logger.warning(f"OCR initialization failed: {str(e)}")
            self.has_ocr = False

    def _match_proj(self, text_block: Dict[str, Any]) -> bool:
        """Check if text block matches projection patterns for structure."""
        proj_patterns = [
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[A-Z]\.",
            r"[0-9]+\.",
            r"[\(（][0-9]+[）\)]",
            r"[•⚫➢①②③④⑤⑥⑦⑧⑨⑩]",
            r"[IVX]+\.",
            r"[a-z]\)",
            r"[A-Z]\)",
        ]
        return any(re.match(p, text_block["text"].strip()) for p in proj_patterns)

    def _updown_concat_features(self, up: Dict, down: Dict) -> List[float]:
        """Extract features for content concatenation decision."""
        features = [
            # Layout matching
            float(up.get("layout_type") == down.get("layout_type")),
            float(up.get("layout_type") == "text"),
            float(down.get("layout_type") == "text"),
            
            # Punctuation analysis
            float(bool(re.search(r"([.?!;+)）]|[a-z]\.)$", up["text"]))),
            float(bool(re.search(r"[,:'\"(+-]$", up["text"]))),
            float(bool(re.search(r"(^.?[/,?;:\],.;:'\"])", down["text"]))),
            
            # Special cases
            float(bool(re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]))),
            float(bool(re.search(r"[,][^.]+$", up["text"]))),
            
            # Parentheses matching
            float(bool(re.search(r"[\(（][^\)）]+$", up["text"]) and 
                 re.search(r"[\)）]", down["text"]))),
                 
            # Character type analysis
            float(bool(re.match(r"[A-Z]", down["text"]))),
            float(bool(re.match(r"[A-Z]", up["text"][-1:]))),
            float(bool(re.match(r"[a-z0-9]", up["text"][-1:]))),
            
            # Distance metrics
            self._x_distance(up, down) / max(self._char_width(up), 0.000001),
            abs(self._block_height(up) - self._block_height(down)) / 
                max(min(self._block_height(up), self._block_height(down)), 0.000001)
        ]
        return features

    def _analyze_layout_with_rows(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Analyze page layout with row awareness."""
        layout_elements = []
        current_row = []
        last_y = 0
        
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:  # text block
                y_mid = (block["bbox"][1] + block["bbox"][3]) / 2
                
                # Check if new row
                if abs(y_mid - last_y) > self.config['row_tolerance'] and current_row:
                    # Process current row
                    layout_elements.extend(self._process_row(current_row))
                    current_row = []
                
                current_row.append(block)
                last_y = y_mid
        
        # Process final row
        if current_row:
            layout_elements.extend(self._process_row(current_row))
            
        return layout_elements

    def _process_row(self, row: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a row of text blocks."""
        # Sort by x coordinate
        row = sorted(row, key=lambda b: b["bbox"][0])
        
        # Analyze row characteristics
        row_height = max(b["bbox"][3] - b["bbox"][1] for b in row)
        row_font_sizes = [b["spans"][0]["size"] for b in row if b["spans"]]
        is_header = any(size > self.base_font_size * 1.2 for size in row_font_sizes)
        
        processed = []
        for block in row:
            block["in_row"] = len(row)
            block["is_row_header"] = is_header
            block["row_height"] = row_height
            processed.append(block)
            
        return processed

    def _concat_text_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Concatenate text blocks intelligently."""
        result = []
        for i, block in enumerate(blocks):
            text = block["text"].strip()
            
            # Skip if empty
            if not text:
                continue
                
            # Check for concatenation with previous block
            if result and i > 0 and self._should_concat_with_previous(block, blocks[i-1]):
                # Add appropriate spacing/joining
                if re.match(r"[a-zA-Z0-9]", text[0]) and \
                   re.match(r"[a-zA-Z0-9]", result[-1][-1]):
                    result[-1] += " " + text
                else:
                    result[-1] += text
            else:
                result.append(text)
                
        return "\n".join(result)

    def _should_concat_with_previous(
        self, 
        current: Dict[str, Any], 
        previous: Dict[str, Any]
    ) -> bool:
        """Determine if blocks should be concatenated."""
        # Get coordinates
        prev_end = previous["bbox"][3]
        curr_start = current["bbox"][1]
        
        # Check vertical distance
        if curr_start - prev_end > self.config['line_spacing'] * 1.5:
            return False
            
        # Check if same paragraph
        if current.get("in_row", 1) > 1 or previous.get("in_row", 1) > 1:
            return False
            
        # Check for sentence completion
        if not re.search(r"[.!?]$", previous["text"]):
            return True
            
        return False

    def _char_width(self, block: Dict[str, Any]) -> float:
        """Calculate average character width in block."""
        width = block["bbox"][2] - block["bbox"][0]
        text_length = len(block["text"].strip())
        return width / max(text_length, 1)

    def _block_height(self, block: Dict[str, Any]) -> float:
        """Calculate block height."""
        return block["bbox"][3] - block["bbox"][1]

    def _x_distance(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> float:
        """Calculate horizontal distance between blocks."""
        return abs(block1["bbox"][0] - block2["bbox"][0])

    async def extract(self, document: 'Document') -> 'Document':
        """Extract content from PDF document with layout awareness."""
        try:
            content = document.content
            if isinstance(content, str):
                # If it's a file path, read the file
                with open(content, 'rb') as f:
                    content = f.read()
                    
            # Open PDF from binary content
            doc = fitz.open(stream=content, filetype="pdf")
            
            # Process document structure
            structure = self._process_document_structure(doc)
            
            # Extract content with layout awareness
            extracted_text = self._extract_structured_text(doc, structure)
            
            # Update document with extracted content
            document.doc_info.update({
                'extracted_text': extracted_text,
                'structure': structure['hierarchy'],
                'metadata': self._extract_pdf_metadata(doc)
            })
            
            # Perform OCR if needed
            if self.has_ocr and self.config.get('use_ocr', False):
                ocr_text = await self._perform_ocr(doc)
                document.doc_info['ocr_text'] = ocr_text
            
            doc.close()
            return document
            
        except Exception as e:
            self.logger.error(f"PDF extraction error: {str(e)}")
            raise

    def _process_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        """Process document structure and hierarchy."""
        structure = {
            'hierarchy': [],
            'page_layouts': [],
            'content_map': {}
        }
        
        # Process each page
        for page_num, page in enumerate(doc):
            layout = self._analyze_page_layout(page)
            structure['page_layouts'].append(layout)
            
            # Build content hierarchy
            for element in layout:
                if element.type in ['title', 'subtitle', 'heading']:
                    structure['hierarchy'].append({
                        'text': element.text,
                        'level': self._determine_heading_level(element),
                        'page': page_num + 1
                    })
                    
        return structure

    def _analyze_page_layout(self, page: fitz.Page) -> List[LayoutElement]:
        """Analyze page layout with enhanced recognition."""
        elements = []
        
        # Get raw layout information
        layout = page.get_text("dict")
        base_font_size = self._get_base_font_size(layout)
        
        for block in layout['blocks']:
            if block.get('type') == 0:  # Text block
                element = self._process_text_block(block, base_font_size)
                if element:
                    elements.append(element)
                    
        # Merge related elements
        elements = self._merge_related_elements(elements)
        
        return elements

    def _process_text_block(
        self,
        block: Dict[str, Any],
        base_font_size: float
    ) -> Optional[LayoutElement]:
        """Process text block with layout analysis."""
        try:
            # Extract text properties
            text = ' '.join(span['text'] for span in block['spans'])
            font_info = block['spans'][0]  # Use first span for font info
            
            # Calculate properties
            font_size = font_info.get('size', 0)
            is_bold = 'bold' in font_info.get('font', '').lower()
            
            # Determine element type
            element_type = self._determine_element_type(
                font_size,
                base_font_size,
                is_bold,
                block['bbox']
            )
            
            return LayoutElement(
                type=element_type,
                text=text,
                bbox=block['bbox'],
                font_size=font_size,
                font_name=font_info.get('font', ''),
                is_bold=is_bold
            )
            
        except Exception as e:
            self.logger.warning(f"Error processing text block: {str(e)}")
            return None

    def _determine_element_type(
        self,
        font_size: float,
        base_font_size: float,
        is_bold: bool,
        bbox: Tuple[float, float, float, float]
    ) -> str:
        """Determine element type based on properties."""
        # Title detection
        if font_size >= base_font_size * 1.5:
            return 'title'
            
        # Subtitle detection
        if font_size >= base_font_size * 1.2 or (
            font_size >= base_font_size * 1.1 and is_bold
        ):
            return 'subtitle'
            
        # Heading detection
        if is_bold or font_size > base_font_size:
            return 'heading'
            
        return 'text'

    def _merge_related_elements(
        self,
        elements: List[LayoutElement]
    ) -> List[LayoutElement]:
        """Merge related elements based on layout."""
        merged = []
        current = None
        
        for element in elements:
            if not current:
                current = element
                continue
                
            # Check if elements should be merged
            if self._should_merge_elements(current, element):
                current = self._merge_elements(current, element)
            else:
                merged.append(current)
                current = element
                
        if current:
            merged.append(current)
            
        return merged

    def _should_merge_elements(
        self,
        elem1: LayoutElement,
        elem2: LayoutElement
    ) -> bool:
        """Determine if elements should be merged."""
        # Check vertical distance
        vertical_gap = elem2.bbox[1] - elem1.bbox[3]
        
        # Check horizontal overlap
        horizontal_overlap = (
            min(elem1.bbox[2], elem2.bbox[2]) -
            max(elem1.bbox[0], elem2.bbox[0])
        )
        
        return (
            elem1.type == elem2.type and
            vertical_gap <= self.config['merge_tolerance'] and
            horizontal_overlap > 0
        )

    def _extract_tables_with_structure(
        self,
        doc: fitz.Document
    ) -> List[Dict[str, Any]]:
        """Extract tables with structural context."""
        tables = []
        
        for page_num, page in enumerate(doc):
            # Find table regions
            table_regions = self._find_table_regions(page)
            
            for region in table_regions:
                table_data = self._extract_table_data(page, region)
                if table_data:
                    tables.append({
                        'data': table_data,
                        'page': page_num + 1,
                        'bbox': region,
                        'context': self._get_table_context(page, region)
                    })
                    
        return tables

    def _extract_images_with_context(
        self,
        doc: fitz.Document
    ) -> List[Dict[str, Any]]:
        """Extract images with surrounding context."""
        images = []
        
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                if base_image:
                    # Get image context
                    context = self._get_image_context(page, img)
                    
                    images.append({
                        'data': base_image["image"],
                        'metadata': {
                            'page': page_num + 1,
                            'index': img_index,
                            'size': base_image.get("size", 0),
                            'format': base_image.get("ext", "unknown"),
                            'context': context
                        }
                    })
                    
        return images

    def _get_base_font_size(self, layout: Dict[str, Any]) -> float:
        """Determine base font size for the page."""
        font_sizes = []
        
        for block in layout['blocks']:
            if block.get('type') == 0:  # Text block
                for span in block.get('spans', []):
                    if size := span.get('size'):
                        font_sizes.append(size)
                        
        return np.median(font_sizes) if font_sizes else 12.0

    def _extract_pdf_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract comprehensive PDF metadata."""
        metadata = doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'keywords': metadata.get('keywords', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'page_count': doc.page_count,
            'file_size': doc.stream_length
        } 

    async def _perform_ocr(self, doc: fitz.Document) -> str:
        """Perform OCR on document pages."""
        if not self.has_ocr:
            return ""

        ocr_text = []
        for page in doc:
            try:
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Perform OCR
                page_text = await self.ocr.process_image(img)
                ocr_text.append(page_text)
                
            except Exception as e:
                self.logger.warning(f"OCR failed for page {page.number}: {str(e)}")
                continue

        return "\n\n".join(ocr_text) 