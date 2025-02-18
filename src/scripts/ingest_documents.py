"""
Document Ingestion Script
-----------------------

Script for ingesting educational content into the RAG system.
Handles syllabus parsing, content organization, and vector store population.

Features:
- Directory traversal
- File type detection
- Content categorization
- Syllabus parsing
- Batch processing
- Progress tracking
- Error handling

Usage:
    python -m src.scripts.ingest_documents --input_dir /path/to/content --config path/to/config.yaml

    # First, ingest your content
    python -m src.scripts.ingest_documents --input_dir /path/to/content --config config.yaml
    python -m src.scripts.ingest_documents --input_dir "knowledge_bank/Form 4/mathematics/mathematics" --config config/rag_config.yaml

    # Then use the pipeline for queries (in your application)
    from src.rag.pipeline import Pipeline

    pipeline = Pipeline("config.yaml")
    response = await pipeline.generate_response("What is...")
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Union, Optional
import argparse
from datetime import datetime
from tqdm import tqdm
import sys
import time
import warnings
import yaml
import traceback

from src.rag.pipeline import Pipeline
from src.rag.models import Document, ContentModality, ProcessingStage
from src.config.rag_config import ConfigManager
from ..document_processing.extractors.text import TextExtractor
from ..document_processing.extractors.pdf import PDFExtractor
from ..document_processing.extractors.docx import DocxExtractor

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Filter specific warnings
warnings.filterwarnings('ignore', message='could not convert string to float: .*')
warnings.filterwarnings('ignore', message='Trie initialization warning:.*')

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks and displays ingestion progress."""
    
    def __init__(self, total_files: int, log_interval: int = 10):
        self.total_files = total_files
        self.processed_files = 0
        self.failed_files = 0
        self.start_time = time.time()
        
        # Initialize progress bar
        self.pbar = tqdm(
            total=total_files,
            desc="Ingesting documents",
            unit="file",
            file=sys.stdout
        )
        
        self.log_interval = log_interval
        
    def update(self, success: bool = True):
        """Update progress tracking."""
        self.processed_files += 1
        if not success:
            self.failed_files += 1
        self.pbar.update(1)
        
        # Calculate and display statistics
        elapsed_time = time.time() - self.start_time
        avg_time_per_file = elapsed_time / self.processed_files if self.processed_files > 0 else 0
        remaining_files = self.total_files - self.processed_files
        estimated_time_remaining = remaining_files * avg_time_per_file
        
        # Update progress bar description
        self.pbar.set_postfix({
            'success_rate': f"{((self.processed_files - self.failed_files) / self.processed_files * 100):.1f}%" if self.processed_files > 0 else "N/A",
            'eta': f"{estimated_time_remaining/60:.1f}min"
        })
    
    def finalize(self):
        """Display final statistics."""
        self.pbar.close()
        total_time = time.time() - self.start_time
        
        print("\nIngestion Summary:")
        print(f"Total files processed: {self.processed_files}/{self.total_files}")
        print(f"Successfully processed: {self.processed_files - self.failed_files}")
        print(f"Failed: {self.failed_files}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per file: {total_time/self.processed_files:.1f} seconds")
        print(f"Success rate: {((self.processed_files - self.failed_files) / self.processed_files * 100):.1f}%")

class ErrorHandler:
    """Handles error handling for the ingestion process."""
    
    def __init__(self):
        self.errors = []
        
    def log_error(self, message: str, exc_info: bool = False):
        """Log an error message."""
        if exc_info:
            self.errors.append(f"{message}: {traceback.format_exc()}")
        else:
            self.errors.append(message)

class ContentIngester:
    """Handles content ingestion into the RAG system."""
    
    # Add class-level constants
    SYLLABUS_KEYWORDS = {
        'syllabus',
        'curriculum',
        'course outline',
        'learning objectives',
        'course content',
        'scheme of work',
        'lesson plan',
        'teaching plan',
        'course schedule',
        'learning outcomes'
    }
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': ContentModality.PDF,
        '.txt': ContentModality.TEXT,
        '.docx': ContentModality.DOCX,
        '.xlsx': ContentModality.EXCEL,
        '.xls': ContentModality.EXCEL,
        '.csv': ContentModality.CSV
    }
    
    def __init__(self, config: Union[str, Dict[str, Any], Path]):
        """Initialize content ingester.
        
        Args:
            config: Either a path to config file or config dict
        """
        try:
            # Add error handler initialization
            self.error_handler = ErrorHandler()
            
            # Handle config input
            if isinstance(config, (str, Path)):
                with open(config, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif isinstance(config, dict):
                self.config = config
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
            
            # Initialize logger
            self.logger = logging.getLogger(self.__class__.__name__)
            
            # Initialize pipeline with the loaded config dict
            self.pipeline = Pipeline(self.config)
            
            # Initialize tracking
            self.processed_files = set()
            self.progress_tracker = None
            
        except Exception as e:
            logger.error(f"Failed to initialize ContentIngester: {str(e)}")
            raise
        
    async def ingest_directory(self, directory: Path) -> None:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to content directory
        """
        try:
            # Add detailed progress tracking
            self.progress_tracker = ProgressTracker(
                total_files=sum(1 for _ in directory.rglob('*') if _.is_file()),
                log_interval=10  # Log progress every 10%
            )
            
            # First, find and process syllabus files
            syllabus_files = await self._find_syllabus_files(directory)
            syllabus_content = await self._process_syllabus_files(syllabus_files)
            
            # Then process all other content files
            content_files = self._get_content_files(directory)
            
            # Group files by subject/topic based on syllabus
            organized_content = self._organize_content(content_files, syllabus_content)
            
            # Process each group
            for subject, files in organized_content.items():
                logger.info(f"Processing content for subject: {subject}")
                await self._process_content_group(subject, files)
                
            # Display final statistics
            self.progress_tracker.finalize()
                
        except Exception as e:
            self.logger.error(f"Error processing directory: {str(e)}")
            raise
            
    async def _find_syllabus_files(self, directory: Path) -> List[Path]:
        """Find syllabus files in directory based on keywords."""
        syllabus_files = []
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    # Check if filename contains syllabus keywords
                    if any(keyword in file_path.stem.lower() 
                          for keyword in self.SYLLABUS_KEYWORDS):
                        syllabus_files.append(file_path)
                        
            return syllabus_files
            
        except Exception as e:
            self.logger.error(f"Error finding syllabus files: {str(e)}")
            return []
        
    async def _process_syllabus_files(self, files: List[Path]) -> Dict[str, Any]:
        """
        Process syllabus files to extract course structure.
        
        Args:
            files: List of syllabus file paths
            
        Returns:
            Dict containing course structure and topics
        """
        syllabus_content = {}
        
        for file_path in files:
            try:
                # Process through pipeline
                document = await self.pipeline.process_document(
                    source=str(file_path),
                    modality=ContentModality.TEXT,
                    options={'is_syllabus': True}
                )
                
                # Extract course structure
                structure = self._extract_course_structure(document)
                syllabus_content.update(structure)
                
            except Exception as e:
                logger.error(f"Error processing syllabus {file_path}: {str(e)}")
                
        return syllabus_content
        
    def _get_content_files(self, directory: Path) -> List[Path]:
        """Get all supported content files."""
        content_files = []
        
        for file_path in directory.rglob('*'):
            if (file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS and 
                str(file_path) not in self.processed_files):
                content_files.append(file_path)
                
        return content_files
        
    def _organize_content(
        self,
        files: List[Path],
        syllabus_content: Dict[str, Any]
    ) -> Dict[str, List[Path]]:
        """
        Organize content files by subject/topic.
        
        Args:
            files: List of content files
            syllabus_content: Extracted syllabus information
            
        Returns:
            Dict mapping subjects to their content files
        """
        organized = {}
        
        for file_path in files:
            # Match file to subject based on filename and location
            subject = self._match_content_to_subject(file_path, syllabus_content)
            
            if subject not in organized:
                organized[subject] = []
            organized[subject].append(file_path)
            
        return organized
        
    async def _process_content_group(
        self,
        subject: str,
        files: List[Path]
    ) -> None:
        """
        Process a group of related content files.
        
        Args:
            subject: Subject/topic name
            files: List of content files
        """
        for file_path in files:
            try:
                # Determine modality
                modality = self._detect_modality(file_path)
                
                # Read file content
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Create document object
                document = Document(
                    content=content,
                    source=str(file_path),
                    modality=modality,
                    doc_info={
                        'subject': subject,
                        'file_path': str(file_path)
                    }
                )
                
                # Process through pipeline
                processed_doc = await self.pipeline.process_document(document)
                
                # Mark as processed
                self.processed_files.add(str(file_path))
                self.progress_tracker.update(success=True)
                
                self.logger.info(f"Processed {file_path} for subject {subject}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                self.progress_tracker.update(success=False)
                
    def _extract_course_structure(self, document: Document) -> Dict[str, Any]:
        """
        Extract course structure from syllabus document.
        
        Args:
            document: Processed syllabus document
            
        Returns:
            Dict containing course structure with topics and subtopics
        """
        try:
            # Use OpenAI to extract structured information
            prompt = f"""
            Analyze this syllabus content and extract the course structure.
            Format the response as a JSON structure with:
            - Course name
            - Topics/units
            - Subtopics for each unit
            - Key terms and concepts
            - Learning objectives
            
            Syllabus content:
            {document.content}
            """
            
            response = self.pipeline._generate_completion(prompt)
            
            # Parse JSON response
            import json
            structure = json.loads(response)
            
            # Add metadata
            structure['source_document'] = document.metadata.get('filename')
            structure['processed_date'] = datetime.now().isoformat()
            
            return structure
            
        except Exception as e:
            logger.error(f"Course structure extraction failed: {str(e)}")
            return {
                'course_name': 'Unknown',
                'topics': [],
                'metadata': {
                    'extraction_error': str(e),
                    'source_document': document.metadata.get('filename')
                }
            }

    def _match_content_to_subject(
        self,
        file_path: Path,
        syllabus_content: Dict[str, Any]
    ) -> str:
        """
        Match content file to subject based on syllabus structure.
        
        Args:
            file_path: Content file path
            syllabus_content: Extracted syllabus information
            
        Returns:
            Matched subject/topic name
        """
        try:
            # Extract potential identifiers from file path
            identifiers = self._extract_identifiers(file_path)
            
            # Get topics and their keywords from syllabus
            topics = self._get_topic_keywords(syllabus_content)
            
            # Score each topic
            scores = {}
            for topic, keywords in topics.items():
                score = self._calculate_topic_match_score(identifiers, keywords)
                scores[topic] = score
            
            # Get best matching topic
            if scores:
                best_match = max(scores.items(), key=lambda x: x[1])
                if best_match[1] > 0.3:  # Confidence threshold
                    return best_match[0]
            
            # Fall back to directory name if no good match
            return file_path.parent.name
            
        except Exception as e:
            logger.error(f"Content matching failed: {str(e)}")
            return "unclassified"

    def _detect_modality(self, file_path: Path) -> ContentModality:
        """Detect content modality from file extension."""
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        modality = self.SUPPORTED_EXTENSIONS[ext]
        self.logger.debug(f"Detected modality {modality} for file {file_path}")
        return modality

    def _extract_identifiers(self, file_path: Path) -> Set[str]:
        """Extract identifying terms from file path."""
        identifiers = set()
        
        # Add terms from file name
        name_parts = file_path.stem.lower().replace('_', ' ').replace('-', ' ').split()
        identifiers.update(name_parts)
        
        # Add terms from directory names
        for parent in file_path.parents:
            if parent.name:
                parent_terms = parent.name.lower().replace('_', ' ').split()
                identifiers.update(parent_terms)
        
        return identifiers

    def _get_topic_keywords(self, syllabus_content: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Extract keywords for each topic from syllabus content."""
        keywords = {}
        
        try:
            for topic in syllabus_content.get('topics', []):
                topic_name = topic.get('name', '')
                if not topic_name:
                    continue
                
                # Collect keywords from topic name
                topic_keywords = set(topic_name.lower().split())
                
                # Add keywords from subtopics
                for subtopic in topic.get('subtopics', []):
                    topic_keywords.update(subtopic.lower().split())
                
                # Add keywords from key terms
                topic_keywords.update(
                    term.lower() for term in topic.get('key_terms', [])
                )
                
                keywords[topic_name] = topic_keywords
                
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            
        return keywords

    def _calculate_topic_match_score(
        self,
        identifiers: Set[str],
        topic_keywords: Set[str]
    ) -> float:
        """Calculate matching score between identifiers and topic keywords."""
        try:
            # Get overlapping terms
            common_terms = identifiers & topic_keywords
            
            if not common_terms:
                return 0.0
            
            # Calculate Jaccard similarity
            similarity = len(common_terms) / len(identifiers | topic_keywords)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Match score calculation failed: {str(e)}")
            return 0.0

    def _analyze_pdf_modality(self, file_path: Path) -> ContentModality:
        """Analyze PDF content to determine primary modality."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            
            # Count different content types
            text_count = 0
            image_count = 0
            
            for page in doc:
                # Check for text
                if page.get_text().strip():
                    text_count += 1
                    
                # Check for images
                if page.get_images():
                    image_count += 1
            
            doc.close()
            
            # Determine primary modality
            if image_count > text_count:
                return ContentModality.IMAGE
            return ContentModality.TEXT
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {str(e)}")
            return ContentModality.TEXT

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ingest documents into RAG pipeline')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing documents (enclose in quotes if path contains spaces)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    return parser.parse_args()

async def main():
    """Main ingestion function."""
    args = parse_args()
    
    # Convert to Path object to handle spaces properly
    input_dir = Path(args.input_dir)
    config_path = Path(args.config)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file does not exist: {config_path}")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        ingester = ContentIngester(str(config_path))  # Convert Path to string
        await ingester.ingest_directory(input_dir)
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 