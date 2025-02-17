"""
Document Processor Module
----------------------

Comprehensive document processing system that integrates all processing capabilities
into a unified pipeline.

Key Features:
- Multi-format processing (PDF, DOCX, TXT)
- Text cleaning and normalization
- Mathematical content processing
- Cross-modal content handling
- Metadata extraction
- Content chunking
- Language detection
- Error handling

Technical Details:
- Modular processing pipeline
- Configurable processors
- Resource management
- Cache handling
- Performance optimization
- Error recovery
- Progress tracking

Dependencies:
- numpy>=1.24.0
- spacy>=3.7.0
- pdfminer>=20.0.0
- python-docx>=0.8.11
- sympy>=1.12
- latex2sympy2>=1.0.0
- nltk>=3.8.0

Example Usage:
    # Initialize processor
    processor = DocumentProcessor(
        config=ProcessorConfig(
            clean_text=True,
            extract_math=True,
            chunk_size=512
        )
    )
    
    # Process document
    result = processor.process(
        document_path="path/to/document.pdf",
        options={
            'extract_images': True,
            'ocr_enabled': True
        }
    )

Processing Pipeline:
1. Document Loading
2. Text Extraction
3. Content Cleaning
4. Special Content Processing
5. Chunking
6. Metadata Extraction
7. Result Generation

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import datetime
from .processors.math_processor import MathProcessor, MathProcessorConfig

@dataclass
class ProcessorConfig:
    """Enhanced processor configuration."""
    # Document handling
    supported_formats: List[str] = field(default_factory=lambda: [
        'pdf', 'docx', 'txt', 'md'
    ])
    
    # Text processing
    clean_text: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True
    preserve_line_breaks: bool = False
    
    # Math processing
    extract_math: bool = True
    math_complexity_threshold: float = 0.7
    parse_equations: bool = True
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    
    # Language
    default_language: str = "en"
    detect_language: bool = True
    
    # Performance
    max_workers: int = 4
    timeout: int = 300
    cache_enabled: bool = True

@dataclass
class TextProcessorConfig:
    """Text processor configuration."""
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True

@dataclass
class ProcessingResult:
    """Result of document processing."""
    success: bool
    content: Optional[str] = None
    chunks: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    math_content: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)

class TextProcessor:
    """Handles text processing tasks."""
    def __init__(self, config: TextProcessorConfig):
        self.config = config
        
    def process(self, text: str) -> str:
        """Process text according to configuration."""
        if self.config.normalize_whitespace:
            text = ' '.join(text.split())
        # Add other processing steps based on config
        return text

class DocumentProcessor:
    """Enhanced document processor with integrated features."""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self.logger = logging.getLogger(__name__)
        self._initialize_processors()
        
    def _initialize_processors(self):
        """Initialize all necessary processors."""
        # Text processor for cleaning and normalization
        self.text_processor = TextProcessor(
            TextProcessorConfig(
                remove_urls=self.config.remove_urls,
                remove_emails=self.config.remove_emails,
                normalize_whitespace=self.config.normalize_whitespace
            )
        )
        
        # Math content processor
        if self.config.extract_math:
            self.math_processor = MathProcessor(
                MathProcessorConfig(
                    complexity_threshold=self.config.math_complexity_threshold
                )
            )
            
        # Initialize other processors as needed
        
    def process(
        self,
        document_path: Union[str, Path],
        options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process document with all enabled processors.
        
        Args:
            document_path: Path to document
            options: Additional processing options
            
        Returns:
            ProcessingResult containing processed content and metadata
        """
        try:
            # Load document
            content = self._load_document(document_path)
            
            # Clean and normalize text
            if self.config.clean_text:
                content = self.text_processor.process(content)
            
            # Extract and process mathematical content
            if self.config.extract_math:
                math_content = self.math_processor.process(content)
                
            # Split into chunks
            chunks = self._create_chunks(content)
            
            # Extract metadata
            metadata = self._extract_metadata(content)
            
            return ProcessingResult(
                success=True,
                content=content,
                chunks=chunks,
                metadata=metadata,
                math_content=math_content if self.config.extract_math else None
            )
            
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[str(e)]
            )
            
    def _load_document(self, path: Union[str, Path]) -> str:
        """Load document content based on file type."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
            
        # Handle different file types
        if path.suffix.lower() == '.pdf':
            return self._extract_pdf_content(path)
        elif path.suffix.lower() == '.docx':
            return self._extract_docx_content(path)
        elif path.suffix.lower() in ['.txt', '.md']:
            return path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
    def _create_chunks(self, content: str) -> List[str]:
        """Create overlapping chunks from content."""
        return self.text_processor._split_into_chunks(
            content,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            min_size=self.config.min_chunk_size
        )
        
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from content."""
        metadata = {
            'length': len(content),
            'word_count': len(content.split()),
            'created_at': datetime.now().isoformat()
        }
        
        if self.config.detect_language:
            metadata['language'] = self._detect_language(content)
            
        return metadata 