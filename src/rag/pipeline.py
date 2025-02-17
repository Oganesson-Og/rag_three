"""
Enhanced RAG Pipeline
--------------------------------

Comprehensive pipeline implementation for processing documents through multiple
stages with robust error handling and metric tracking.

Key Features:
- Modular processing stages
- Multi-modal document support
- Comprehensive error handling
- Metric collection and tracking
- Caching integration
- Asynchronous processing
- Flexible configuration

Technical Details:
- Async/await pattern implementation
- Stage-based processing architecture
- Integrated caching system
- Vector storage capabilities
- Comprehensive logging
- Metric collection at each stage

Dependencies:
- asyncio>=3.4.3
- logging>=2.0.0
- pydantic>=2.5.0
- numpy>=1.24.0

Example Usage:
    # Initialize pipeline
    pipeline = Pipeline(config={})
    
    # Process document
    document = await pipeline.process_document(
        source="path/to/doc",
        modality=ContentModality.TEXT
    )
    
    # Generate response
    result = await pipeline.generate_response(
        query="Sample query"
    )

Performance Considerations:
- Asynchronous processing for better throughput
- Efficient caching mechanisms
- Optimized error handling
- Configurable processing stages

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
from datetime import datetime
import yaml
import torch
import ollama
from uuid import uuid4

from .models import (
    Document,
    Chunk,
    ProcessingStage,
    ContentModality,
    ProcessingEvent,
    ProcessingMetrics,
    SearchResult,
    GenerationResult
)
from ..document_processing.extractors.base import BaseExtractor
from ..document_processing.extractors.audio import AudioExtractor
from ..document_processing.extractors.text import TextExtractor
from .prompt_engineering import PromptGenerator
from ..cache.advanced_cache import MultiModalCache, CacheConfig
from ..cache.vector_cache import VectorCache
from ..utils.metrics import MetricsCollector
from ..chunking.utils.text import clean_text, split_into_sentences
from ..chunking.utils.validation import validate_chunk, is_complete_sentence
from ..config.config_manager import ConfigManager
from ..config.embedding_config import EMBEDDING_CONFIG
from ..config.domain_config import get_domain_config
from ..document_processing.processors.diagram_analyzer import DiagramAnalyzer
from ..standards.education_standards_manager import StandardsManager
from ..feedback.feedback_processor import FeedbackProcessor
from ..document_processing.processors.math_processor import MathProcessor
from ..nlp.cross_modal_processor import CrossModalProcessor
from ..database.models import (
    DocumentDB,
    Chunk as DBChunk,
    QdrantVectorStore,
    Base,
    DatabaseConnection
)
from ..llm.model_manager import LLMManager, ModelError, ModelErrorType
from ..document_processing.extractors.image import ImageExtractor
from ..document_processing.extractors.pdf import PDFExtractor
from ..document_processing.extractors.docx import DocxExtractor
from ..document_processing.extractors.spreadsheet import ExcelExtractor, CSVExtractor

logger = logging.getLogger(__name__)

class PipelineStage:
    """Base class for pipeline stages."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsCollector()
    
    async def process(self, document: Document) -> Document:
        """Process document and update its state."""
        raise NotImplementedError
    
    def _record_metrics(self, document: Document, metrics: ProcessingMetrics):
        """Record processing metrics."""
        event = ProcessingEvent(
            stage=self.stage,
            processor=self.__class__.__name__,
            metrics=metrics,
            config_snapshot=self.config
        )
        document.add_processing_event(event)

class ExtractionStage(PipelineStage):
    """Document extraction stage."""
    
    stage = ProcessingStage.EXTRACTED
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize extraction stage."""
        super().__init__(config)
        
        # Get extractor-specific configs
        pdf_config = config.get('pdf', {})
        audio_config = config.get('audio', {})
        text_config = config.get('text', {})
        image_config = config.get('image', {})
        docx_config = config.get('docx', {})
        excel_config = config.get('excel', {})
        csv_config = config.get('csv', {})
        
        # Initialize extractors with their configs
        self.extractors = {
            ContentModality.PDF: PDFExtractor(config=pdf_config),
            ContentModality.AUDIO: AudioExtractor(config=audio_config),
            ContentModality.TEXT: TextExtractor(config=text_config),
            ContentModality.IMAGE: ImageExtractor(config=image_config),
            ContentModality.DOCX: DocxExtractor(config=docx_config),
            ContentModality.EXCEL: ExcelExtractor(config=excel_config),
            ContentModality.CSV: CSVExtractor(config=csv_config)
        }
    
    def _detect_modality(self, document: Document) -> ContentModality:
        """Detect document modality from content type or file extension."""
        # First check content type if available
        if document.content_type:
            content_type = document.content_type.lower()
            content_type_mapping = {
                'pdf': ContentModality.PDF,
                'image': ContentModality.IMAGE,
                'audio': ContentModality.AUDIO,
                'text': ContentModality.TEXT,
                'docx': ContentModality.DOCX,
                'excel': ContentModality.EXCEL,
                'csv': ContentModality.CSV
            }
            
            for key, modality in content_type_mapping.items():
                if key in content_type:
                    return modality
        
        # Fallback to file extension detection
        if isinstance(document.source, str):
            ext = document.source.lower().split('.')[-1]
            extension_mapping = {
                'pdf': ContentModality.PDF,
                'jpg': ContentModality.IMAGE,
                'jpeg': ContentModality.IMAGE,
                'png': ContentModality.IMAGE,
                'gif': ContentModality.IMAGE,
                'bmp': ContentModality.IMAGE,
                'mp3': ContentModality.AUDIO,
                'wav': ContentModality.AUDIO,
                'ogg': ContentModality.AUDIO,
                'txt': ContentModality.TEXT,
                'md': ContentModality.TEXT,
                'rst': ContentModality.TEXT,
                'docx': ContentModality.DOCX,
                'doc': ContentModality.DOCX,
                'xlsx': ContentModality.EXCEL,
                'xls': ContentModality.EXCEL,
                'csv': ContentModality.CSV
            }
            
            if ext in extension_mapping:
                return extension_mapping[ext]
        
        raise ValueError(f"Could not detect modality for document: {document.source}")
    
    async def process(self, document: Document) -> Document:
        """Process document through appropriate extractor."""
        try:
            # Detect modality if not set
            if not document.modality:
                modality = self._detect_modality(document)
                document.doc_info['modality'] = modality.value
            else:
                modality = document.modality
            
            # Get appropriate extractor
            extractor = self.extractors.get(modality)
            if not extractor:
                raise ValueError(f"No extractor found for modality: {modality}")
            
            # Process with metrics collection
            with self.metrics.measure_time() as timer:
                processed_document = await extractor.extract(document)
                
                # Record metrics
                self._record_metrics(processed_document, ProcessingMetrics(
                    processing_time=timer.elapsed,
                    token_count=len(str(processed_document.content)) if processed_document.content else 0
                ))
                
                return processed_document
                
        except Exception as e:
            logger.error(f"Extraction stage failed: {str(e)}")
            raise

class ChunkingStage(PipelineStage):
    """Document chunking stage."""
    
    stage = ProcessingStage.CHUNKED
    
    async def process(self, document: Document) -> Document:
        with self.metrics.measure_time() as timer:
            # Clean text
            cleaned_text = clean_text(document.content)
            
            # Split into sentences
            sentences = split_into_sentences(cleaned_text)
            
            # Create chunks
            chunks = []
            current_chunk = []
            
            for sentence in sentences:
                current_chunk.append(sentence)
                chunk_text = " ".join(current_chunk)
                
                if validate_chunk(Chunk(text=chunk_text)):
                    chunks.append(chunk_text)
                    current_chunk = []
            
            document.chunks = [Chunk(text=chunk) for chunk in chunks]
            
        self._record_metrics(document, ProcessingMetrics(
            processing_time=timer.elapsed,
            chunk_count=len(chunks)
        ))
        return document

class DiagramAnalysisStage(PipelineStage):
    """Diagram analysis stage."""
    
    stage = ProcessingStage.ANALYZED
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.analyzer = DiagramAnalyzer(config)
    
    async def process(self, document: Document) -> Document:
        if document.has_diagrams:
            with self.metrics.measure_time() as timer:
                document.diagram_analysis = await self.analyzer.analyze(document.diagrams)
            
            self._record_metrics(document, ProcessingMetrics(
                processing_time=timer.elapsed,
                diagram_count=len(document.diagrams)
            ))
        return document

class EducationalProcessingStage(PipelineStage):
    """Educational content processing stage."""
    
    stage = ProcessingStage.EDUCATIONAL_PROCESSED
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.math_processor = MathProcessor(config.get('math', {}))
        self.standards_manager = StandardsManager(config.get('standards', {}))
        self.cross_modal_processor = CrossModalProcessor(config.get('cross_modal', {}))
    
    async def process(self, document: Document) -> Document:
        """Process document with educational enhancements."""
        with self.metrics.measure_time() as timer:
            # Process mathematical content if present
            if self.math_processor.has_math_content(document.content):
                document.content = await self.math_processor.process(document.content)
            
            # Map educational standards
            standards_mapping = await self.standards_manager.map_content(document.content)
            document.metadata['standards'] = standards_mapping
            
            # Cross-modal educational analysis
            if document.has_multiple_modalities():
                document = await self.cross_modal_processor.process(document)
            
        self._record_metrics(document, ProcessingMetrics(
            processing_time=timer.elapsed,
            math_content_processed=bool(document.metadata.get('math_processed')),
            standards_mapped=len(standards_mapping),
            modalities_processed=len(document.processed_modalities)
        ))
        return document

class FeedbackProcessingStage(PipelineStage):
    """Educational feedback processing stage."""
    
    stage = ProcessingStage.FEEDBACK_PROCESSED
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feedback_processor = FeedbackProcessor(config.get('feedback', {}))
    
    async def process(self, document: Document) -> Document:
        """Process educational feedback."""
        with self.metrics.measure_time() as timer:
            feedback = await self.feedback_processor.process_document(
                document.content,
                document.metadata
            )
            document.metadata['educational_feedback'] = feedback
            
            # Update learning progress if student ID is present
            if student_id := document.metadata.get('student_id'):
                await self.feedback_processor.update_learning_progress(
                    student_id=student_id,
                    topic=document.metadata.get('topic'),
                    performance=feedback.get('performance_metrics')
                )
        
        self._record_metrics(document, ProcessingMetrics(
            processing_time=timer.elapsed,
            feedback_generated=bool(feedback),
            learning_progress_updated=bool(student_id)
        ))
        return document

class Pipeline:
    """RAG pipeline implementation."""
    
    def __init__(self, config: Union[str, Dict[str, Any], Path]):
        """Initialize pipeline with configuration.
        
        Args:
            config: Either a path to config file or config dict
        """
        try:
            # Handle config input
            if isinstance(config, (str, Path)):
                with open(config, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif isinstance(config, dict):
                self.config = config
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
            
            # Initialize logger
            self.logger = logging.getLogger(__name__)
            
            # Initialize components
            self._init_stages()
            self._init_cache()
            self._init_vector_store()
            self._init_other_components()
            
        except Exception as e:
            logger.error(f"Failed to initialize Pipeline: {str(e)}")
            logger.error(f"Config that failed: {self.config}")
            raise

    def _init_stages(self):
        """Initialize pipeline stages."""
        self.stages = {
            ProcessingStage.EXTRACTED: ExtractionStage(self.config.get('extraction', {})),
            ProcessingStage.CHUNKED: ChunkingStage(self.config.get('chunking', {})),
            ProcessingStage.ANALYZED: DiagramAnalysisStage(self.config.get('diagram', {})),
            ProcessingStage.EDUCATIONAL_PROCESSED: EducationalProcessingStage(self.config.get('educational', {})),
            ProcessingStage.FEEDBACK_PROCESSED: FeedbackProcessingStage(self.config.get('feedback', {}))
        }

    def _init_cache(self):
        """Initialize caching system."""
        try:
            cache_config = self.config.get('cache', {})
            if cache_config.get('enabled', False):
                self.cache = MultiModalCache(config=cache_config)
                self.logger.info("Cache initialized successfully")
            else:
                self.cache = None
                self.logger.info("Cache disabled in configuration")
        except Exception as e:
            self.logger.error(f"Cache initialization failed: {str(e)}")
            self.cache = None

    def _init_vector_store(self):
        """Initialize vector store."""
        try:
            # Get vector store config from dict
            vector_store_config = self.config.get('vector_store', {})
            
            # Initialize QdrantVectorStore with config
            self.vector_store = QdrantVectorStore(
                collection_name=vector_store_config.get('collection_name', 'default'),
                dimension=vector_store_config.get('dimension', 1536),
                similarity_threshold=vector_store_config.get('similarity_threshold', 0.8),
                host=vector_store_config.get('host', 'localhost'),
                port=vector_store_config.get('port', 6333)
            )
            
            self.logger.info(f"Vector store initialized with collection: {vector_store_config.get('collection_name')}")
            
        except Exception as e:
            self.logger.error(f"Vector store initialization failed: {str(e)}")
            raise

    def _init_other_components(self):
        """Initialize other pipeline components."""
        try:
            # Initialize prompt generator
            prompt_config = self.config.get('components', {}).get('prompts', {})
            self.prompt_generator = PromptGenerator(prompt_config)
            
            # Initialize database connection
            db_config = self.config.get('database', {})
            self.db_connection = DatabaseConnection(
                connection_string=db_config.get('url', 'sqlite:///rag.db')
            )
            self.db_connection.connect()
            
            # Initialize Ollama client
            model_config = self.config.get('model', {})
            self.llm = ollama.Client(
                host=model_config.get('model_endpoint', 'http://localhost:11434')
            )
            
            # Test model availability
            try:
                model_name = model_config.get('model_name', 'qwen2.5')  # Default to qwen2.5 if not specified
                self.llm.chat(
                    model=model_name,
                    messages=[{'role': 'system', 'content': 'Test connection'}]
                )
                self.logger.info(f"Successfully connected to Ollama model: {model_name}")
            except Exception as e:
                raise ModelError(
                    message=f"Failed to connect to Ollama model: {str(e)}",
                    error_type=ModelErrorType.INITIALIZATION_ERROR,
                    provider="ollama"
                )
                
        except Exception as e:
            raise ModelError(
                message=f"Component initialization failed: {str(e)}",
                error_type=ModelErrorType.INITIALIZATION_ERROR,
                provider="pipeline"
            )

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
        
    def _create_document(self, source: Union[str, Path, bytes], modality: ContentModality, options: Optional[Dict[str, Any]] = None) -> Document:
        """Create a document instance with proper schema mapping."""
        options = options or {}
        
        # Generate title from source if it's a path
        title = str(Path(source).stem) if isinstance(source, (str, Path)) else "Untitled Document"
        
        return Document(
            title=title,
            source=str(source) if isinstance(source, Path) else None,
            content_type=modality.value,  # Assuming ContentModality is an Enum
            doc_info={
                "content": source if isinstance(source, (str, bytes)) else None,
                "modality": modality.value,
                "processing_options": options,
                "status": "pending",
                **options  # Include any additional metadata
            },
        )

    async def process_document(
        self,
        source: Union[str, bytes],
        modality: Optional[ContentModality] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Process document through pipeline stages."""
        try:
            # Debug: Print input parameters
            self.logger.debug(f"Processing document with parameters:")
            self.logger.debug(f"Source type: {type(source)}")
            self.logger.debug(f"Modality: {modality}")
            self.logger.debug(f"Options: {options}")

            # Read file content if source is a file path
            if isinstance(source, str) and Path(source).exists():
                try:
                    with open(source, 'rb') as f:
                        content = f.read()
                    self.logger.debug(f"Successfully read file: {source}")
                    self.logger.debug(f"Content type: {type(content)}")
                    self.logger.debug(f"Content size: {len(content)} bytes")
                except Exception as e:
                    self.logger.error(f"Failed to read file {source}: {str(e)}")
                    raise
            else:
                content = source
                self.logger.debug(f"Using provided content directly")

            # Debug: Print document creation parameters
            self.logger.debug("Creating document with parameters:")
            self.logger.debug(f"Content type: {type(content)}")
            self.logger.debug(f"Source: {source}")
            self.logger.debug(f"Modality: {modality}")
            self.logger.debug(f"Doc info: {{'options': {options or {}}}}")

            try:
                # Create document object with required fields
                document = Document(
                    content=content,
                    source=str(source),
                    modality=modality,
                    doc_info={'options': options or {}}
                )
                self.logger.debug("Successfully created Document object")
            except Exception as e:
                self.logger.error(f"Failed to create Document object: {str(e)}")
                self.logger.error(f"Document parameters that failed:")
                self.logger.error(f"content type: {type(content)}")
                self.logger.error(f"source: {source}")
                self.logger.error(f"modality: {modality}")
                self.logger.error(f"doc_info: {{'options': {options or {}}}}")
                raise
            
            # Debug: Print document object details
            self.logger.debug(f"Created document with ID: {document.id}")
            
            # Process through stages
            for stage in ProcessingStage:
                if stage in self.stages:
                    try:
                        self.logger.debug(f"Starting stage: {stage}")
                        document = await self.stages[stage].process(document)
                        self.logger.debug(f"Completed stage: {stage}")
                    except Exception as e:
                        self.logger.error(f"Stage {stage} failed: {str(e)}")
                        self.logger.error(f"Document state at failure:")
                        self.logger.error(f"Document ID: {document.id}")
                        self.logger.error(f"Current stage: {stage}")
                        raise
                        
            return document
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            self.logger.error(f"Full error details:", exc_info=True)
            raise

    async def generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            context: Additional context
            options: Generation options
            
        Returns:
            Generation result
        """
        try:
            model_config = self.config.get('model', {})
            model_name = model_config.get('model_name', 'qwen2.5')  # Default to qwen2.5 if not specified
            
            response = self.llm.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': self.prompt_generator.generate_system_prompt(context)
                    },
                    {
                        'role': 'user',
                        'content': query
                    }
                ]
            )
            return GenerationResult(
                text=response.message.content,
                chunks_used=[],
                confidence_score=0.9,  # TODO: Implement confidence scoring
                metadata={
                    "query": query,
                    "context": context,
                    "options": options,
                    "educational_feedback": None,
                    "standards_alignment": self._get_standards_alignment(
                        response.message.content,
                        context.get('grade_level') if context else None
                    )
                }
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise
    
    async def _cache_document(self, document: Document):
        """Cache document state."""
        await self.cache.set(f"doc:{document.id}", document)
        if document.embeddings:
            await self.vector_cache.set(document.id, document.embeddings)
    
    async def _retrieve_chunks(
        self,
        query: str,
        max_chunks: int = 5,
        similarity_threshold: float = 0.7,
        retrieval_strategy: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks using multiple retrieval strategies.
        
        Args:
            query: Search query
            max_chunks: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score threshold
            retrieval_strategy: Strategy to use ('semantic', 'keyword', 'hybrid')
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Check cache first
            cache_key = f"search:{hash(query)}"
            cached_results = await self.cache.get(cache_key)
            if cached_results:
                return cached_results

            results = []
            
            if retrieval_strategy in ['semantic', 'hybrid']:
                # Get query embedding
                query_embedding = await self._get_embedding(query)
                
                # Semantic search using vector store
                semantic_results = await self.vector_cache.search(
                    query_embedding,
                    k=max_chunks,
                    threshold=similarity_threshold
                )
                results.extend(semantic_results)

            if retrieval_strategy in ['keyword', 'hybrid']:
                # Keyword-based search
                keyword_results = await self._keyword_search(
                    query,
                    max_chunks=max_chunks
                )
                results.extend(keyword_results)

            # Deduplicate and rank results
            final_results = self._rank_and_deduplicate_results(
                results,
                max_chunks=max_chunks,
                strategy=retrieval_strategy
            )

            # Cache results
            await self.cache.set(cache_key, final_results)
            
            return final_results

        except Exception as e:
            logger.error(f"Chunk retrieval failed: {str(e)}", exc_info=True)
            raise

    async def _keyword_search(
        self,
        query: str,
        max_chunks: int = 5
    ) -> List[SearchResult]:
        """Perform keyword-based search."""
        try:
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Get all chunks from cache
            chunks = await self.cache.get_all_chunks()
            
            # Calculate BM25 scores
            scores = []
            for chunk in chunks:
                score = self._calculate_bm25_score(processed_query, chunk.text)
                scores.append((chunk, score))
            
            # Sort by score and return top results
            scores.sort(key=lambda x: x[1], reverse=True)
            return [
                SearchResult(
                    chunk=chunk,
                    score=score,
                    strategy="keyword"
                ) for chunk, score in scores[:max_chunks]
            ]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}", exc_info=True)
            return []

    def _rank_and_deduplicate_results(
        self,
        results: List[SearchResult],
        max_chunks: int = 5,
        strategy: str = "hybrid"
    ) -> List[SearchResult]:
        """Rank and deduplicate search results."""
        try:
            # Remove duplicates based on chunk ID
            seen_chunks = set()
            unique_results = []
            
            for result in results:
                if result.chunk.id not in seen_chunks:
                    seen_chunks.add(result.chunk.id)
                    unique_results.append(result)

            # Adjust scores based on strategy
            if strategy == "hybrid":
                for result in unique_results:
                    if result.strategy == "semantic":
                        result.score *= 0.7  # Weight for semantic search
                    else:
                        result.score *= 0.3  # Weight for keyword search

            # Sort by score and return top results
            unique_results.sort(key=lambda x: x.score, reverse=True)
            return unique_results[:max_chunks]
            
        except Exception as e:
            logger.error(f"Result ranking failed: {str(e)}", exc_info=True)
            return results[:max_chunks]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using configured model."""
        try:
            # Check embedding cache
            cache_key = f"emb:{hash(text)}"
            cached_embedding = await self.vector_cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

            # Generate embedding using configured model
            embedding = await self.embedding_model.embed_text(text)
            
            # Cache embedding
            await self.vector_cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            raise

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess search query with tokenization, stopword removal, and normalization.
        
        Args:
            query: Raw query string
            
        Returns:
            Preprocessed query string
        """
        try:
            # Import NLTK components
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Download required NLTK data (if not already downloaded)
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Convert to lowercase
            query = query.lower()
            
            # Tokenize
            tokens = word_tokenize(query)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            # Remove special characters and numbers
            tokens = [token for token in tokens if token.isalnum()]
            
            return ' '.join(tokens)
            
        except Exception as e:
            self.logger.error(f"Query preprocessing failed: {str(e)}")
            return query.lower()  # Fallback to basic preprocessing

    def _calculate_bm25_score(self, query: str, text: str) -> float:
        """
        Calculate BM25 similarity score between query and text.
        
        Args:
            query: Preprocessed query string
            text: Document text to compare against
            
        Returns:
            BM25 similarity score
        """
        try:
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize
            
            # Tokenize query and text
            query_tokens = word_tokenize(query)
            text_tokens = word_tokenize(text)
            
            # Create BM25 object with single document
            bm25 = BM25Okapi([text_tokens])
            
            # Calculate score
            score = bm25.get_scores(query_tokens)[0]
            
            return float(score)
            
        except Exception as e:
            self.logger.error(f"BM25 scoring failed: {str(e)}")
            return 0.0

    def _get_standards_alignment(
        self,
        text: str,
        grade_level: Optional[str]
    ) -> Dict[str, Any]:
        """Get educational standards alignment for generated response."""
        try:
            standards_stage = self.stages[ProcessingStage.EDUCATIONAL_PROCESSED]
            return standards_stage.standards_manager.get_alignment(
                text, grade_level
            )
        except Exception as e:
            logger.warning(f"Standards alignment failed: {str(e)}")
            return {}

    async def process_educational_session(
        self,
        student_id: str,
        content: str,
        session_id: Optional[str] = None
    ):
        """Process educational content with both vector and relational storage."""
        try:
            # Store vectors in Qdrant
            vector = await self.generate_embedding(content)
            qdrant_id = await self.vector_store.add_vector(
                vector=vector,
                metadata={"student_id": student_id}
            )
            
            # Store educational metadata in PostgreSQL
            with self.db_connection.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO content_metadata 
                    (id, qdrant_id, content_type, educational_metadata)
                    VALUES (%s, %s, %s, %s)
                """, (
                    uuid4(),
                    qdrant_id,
                    "educational_content",
                    {
                        "student_id": student_id,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                ))
                
            return qdrant_id
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise 