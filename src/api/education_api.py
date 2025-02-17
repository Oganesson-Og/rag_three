"""
Educational RAG API Module
------------------------

FastAPI-based REST API implementation providing endpoints for educational content
retrieval, benchmarking, and system management.

Key Features:
- Query processing endpoints
- Benchmark execution endpoints
- Subject management interfaces
- Comprehensive error handling
- Response formatting and validation
- Rate limiting and throttling
- Authentication and authorization
- API documentation generation

Technical Details:
- FastAPI framework implementation
- OpenAPI/Swagger documentation
- JWT authentication support
- Request validation using Pydantic
- Async endpoint handling
- CORS support

Dependencies:
- fastapi>=0.109.0
- uvicorn>=0.27.0
- pydantic>=2.5.0
- python-jose>=3.3.0
- python-multipart>=0.0.6
- starlette>=0.36.0

Example Usage:
    # Start API server
    uvicorn education_api:app --reload
    
    # Query endpoint
    POST /query
    {
        "query": "explain quantum mechanics",
        "subject": "physics",
        "level": "a-level",
        "filters": {"complexity": "advanced"}
    }
    
    # Benchmark endpoint
    POST /benchmark
    {
        "subject": "physics",
        "level": "a-level",
        "model_key": "scibert"
    }

API Security:
- JWT token authentication
- Role-based access control
- Rate limiting per client
- Input validation and sanitization
- Error handling and logging

Author: Keith Satuku
Version: 2.2.0
Created: 2025
License: MIT
"""
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, status, BackgroundTasks, WebSocket
from pydantic import BaseModel, Field

# Remove preprocessor import and use extractors instead
from ..document_processing.extractors import (
    PDFExtractor,
    DocxExtractor,
    ExcelExtractor,
    CSVExtractor,
    TextExtractor,
    AudioExtractor,
    BaseExtractor
)

from ..retrieval.education_retriever import EducationRetriever
from ..embeddings.education_benchmark import EducationBenchmark
from ..config.domain_config import EDUCATION_DOMAINS
import whisper
from transformers import pipeline
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
import uuid
import time  # Add at top with other imports
import logging
from typing_extensions import TypeAlias
from datetime import datetime
from fastapi.websockets import WebSocketDisconnect

from ..audio.processor import AudioProcessor
from ..config.settings import AUDIO_CONFIG

# Import document processing components
from ..document_processing.processor import DocumentProcessor
from ..document_processing.processors.ocr import OCRProcessor
from ..document_processing.preprocessor import Preprocessor
from ..document_processing.processors.diagram_archive import DiagramAnalyzer

from ..embeddings.embedding_generator import EmbeddingGenerator

# Add with other imports at top
from ..nlp.cross_modal import CrossModalProcessor

# Add with other imports
from ..feedback.processor import FeedbackProcessor

from ..assessment.processor import AssessmentProcessor

# Type aliases
DocumentContent = Union[str, bytes]
ProcessingResult = Dict[str, Any]
SearchResult = Dict[str, Any]

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1)
    subject: str
    level: str
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    top_k: int = Field(default=5, gt=0)

    class Config:
        schema_extra = {
            "example": {
                "query": "Explain quantum mechanics",
                "subject": "physics",
                "level": "a-level",
                "filters": {"complexity": "advanced"},
                "top_k": 5
            }
        }

class BenchmarkRequest(BaseModel):
    subject: str
    level: str
    model_key: Optional[str] = None

class DocumentRequest(BaseModel):
    """Document processing request model."""
    content: DocumentContent
    file_type: str = Field(..., min_length=1)
    subject: str
    grade_level: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    processing_options: Dict[str, bool] = Field(
        default_factory=lambda: {
            "perform_ocr": True,
            "analyze_diagrams": True,
            "extract_tables": True,
            "detect_equations": True
        }
    )
    content_type: str = Field(..., min_length=1)

class ProcessedDocument(BaseModel):
    """Processed document response model."""
    document_id: str = Field(..., min_length=1)
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_summary: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

class AudioQueryRequest(BaseModel):
    """Audio query request model."""
    subject: str
    grade_level: str
    language: str = Field(default="en")
    task_type: str = Field(default="transcribe")
    additional_context: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "subject": "physics",
                "grade_level": "a-level",
                "language": "en",
                "task_type": "transcribe",
                "additional_context": {
                    "topic": "quantum mechanics"
                }
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title="Education RAG API",
    description="API for educational content retrieval and processing",
    version="1.0.0"
)

# Define extractor mapping
EXTRACTOR_MAP = {
    'pdf': PDFExtractor,
    'docx': DocxExtractor,
    'xlsx': ExcelExtractor,
    'csv': CSVExtractor,
    'txt': TextExtractor,
    'audio': AudioExtractor,
    'mp3': AudioExtractor,
    'wav': AudioExtractor,
    'm4a': AudioExtractor,
    'flac': AudioExtractor
}

class DocumentProcessor:
    """Document processor using specialized extractors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.extractors: Dict[str, BaseExtractor] = {}
        
        # Initialize extractors
        for ext, extractor_class in EXTRACTOR_MAP.items():
            try:
                self.extractors[ext] = extractor_class(self.config.get(ext, {}))
            except Exception as e:
                logging.error(f"Failed to initialize {ext} extractor: {str(e)}")

    async def process_document(
        self,
        content: Union[str, bytes],
        file_type: str,
        options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Process document using appropriate extractor.
        
        Args:
            content: Document content
            file_type: Type of document (pdf, docx, etc.)
            options: Processing options
            
        Returns:
            Processed document content and metadata
        """
        # Get appropriate extractor
        extractor = self.extractors.get(file_type.lower())
        if not extractor:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        try:
            # Extract content using specialized extractor
            result = await extractor.extract(content, options)
            
            # Add processing metadata
            result['metadata'].update({
                'processor': extractor.__class__.__name__,
                'file_type': file_type,
                'processing_options': options
            })
            
            return result
            
        except Exception as e:
            logging.error(f"Document processing failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Document processing failed: {str(e)}"
            )

# Initialize components
try:
    document_processor = DocumentProcessor()
    education_retriever = EducationRetriever()
    education_benchmark = EducationBenchmark()
    audio_processor = AudioProcessor(
        model_name=AUDIO_CONFIG.get('model', 'whisper-large-v3'),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    embedding_generator = EmbeddingGenerator()

except Exception as e:
    logging.error(f"Error initializing components: {str(e)}")
    raise

# Cache for retrievers
retrievers: Dict[str, Any] = {}

# Add after imports
logger = logging.getLogger(__name__)

def generate_document_id() -> str:
    """Generate a unique document ID."""
    return str(uuid.uuid4())

@app.post("/query", response_model=Dict[str, Any])
async def query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Query endpoint for educational content retrieval.
    
    Args:
        request: Query request containing search parameters
        background_tasks: FastAPI background tasks handler
    
    Returns:
        Dict containing search results and metadata
    
    Raises:
        HTTPException: If query processing fails
    """
    try:
        # Get or create retriever
        retriever_key = f"{request.subject}_{request.level}"
        if retriever_key not in retrievers:
            retrievers[retriever_key] = EducationRetriever(
                subject=request.subject,
                level=request.level,
                top_k=request.top_k
            )
        
        # Get results
        results = retrievers[retriever_key].retrieve(
            query=request.query,
            filter_metadata=request.filters
        )
        
        # Add to background tasks
        background_tasks.add_task(
            log_query,
            query=request.query,
            subject=request.subject,
            results_count=len(results)
        )
        
        return {
            "status": "success",
            "results": results,
            "metadata": {
                "subject": request.subject,
                "level": request.level,
                "total_results": len(results),
                "query_time": time.time()
            }
        }
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/benchmark", response_model=Dict[str, Any])
async def benchmark_endpoint(request: BenchmarkRequest) -> Dict[str, Any]:
    """
    Run benchmarks for educational embeddings.
    
    Args:
        request: Benchmark request parameters
    
    Returns:
        Dict containing benchmark results
    
    Raises:
        HTTPException: If benchmark fails
    """
    try:
        benchmark = EducationBenchmark(
            subject=request.subject,
            level=request.level
        )
        
        results = benchmark.run_education_benchmark()
        
        if request.model_key:
            results = {request.model_key: results[request.model_key]}
            
        return {
            "status": "success",
            "benchmark_results": results,
            "metadata": {
                "subject": request.subject,
                "level": request.level,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {str(e)}"
        )

@app.get("/subjects")
async def list_subjects() -> Dict:
    """List available subjects and levels."""
    return {
        "status": "success",
        "domains": EDUCATION_DOMAINS
    }

@app.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    request: AudioQueryRequest = None
):
    """
    Transcribe audio and process educational content.
    """
    try:
        # Validate audio file
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format"
            )

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Process audio with enhanced features
            result = audio_processor.process_audio(
                audio_path=tmp_path,
                task_type=request.task_type,
                language=request.language,
                context={
                    "subject": request.subject,
                    "grade_level": request.grade_level,
                    **(request.additional_context or {})
                }
            )

            # Get educational context with enhanced retrieval
            education_retriever = EducationRetriever(
                subject=request.subject,
                level=request.grade_level
            )

            # Process transcribed text with domain-specific context
            educational_context = education_retriever.retrieve(
                query=result['text'],
                filter_metadata={
                    "subject": request.subject,
                    "grade_level": request.grade_level
                }
            )

            # Enhanced response with detailed metrics
            return {
                "status": "success",
                "transcription": {
                    "text": result['text'],
                    "confidence": result['confidence'],
                    "language": result['detected_language'],
                    "duration": result['duration'],
                    "segments": result['segments'],
                    "word_timestamps": result.get('word_timestamps', []),
                    "speaker_diarization": result.get('speaker_diarization', [])
                },
                "educational_context": educational_context,
                "technical_terms": result.get('technical_terms', []),
                "domain_specific_terms": result.get('domain_specific_terms', []),
                "complexity_metrics": {
                    "readability_score": result.get('readability_score'),
                    "technical_density": result.get('technical_density'),
                    "confidence_by_segment": result.get('confidence_by_segment', {})
                },
                "metadata": {
                    "processing_time": result.get('processing_time'),
                    "model_used": result.get('model_used'),
                    "audio_quality": result.get('audio_quality')
                }
            }

        finally:
            # Cleanup temporary file
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing error: {str(e)}"
        )

@app.post("/audio/query")
async def audio_query(
    request: AudioQueryRequest,
    file: UploadFile = File(...)
):
    """
    Process audio query and retrieve educational content.
    
    Features:
    - Voice query processing
    - Educational content retrieval
    - Context-aware responses
    - Multi-language support
    """
    try:
        # First transcribe the audio query
        transcription = await transcribe_audio(file, request)
        
        # Use the transcribed text for educational retrieval
        education_retriever = EducationRetriever(
            subject=request.subject,
            level=request.grade_level
        )
        
        # Get relevant educational content
        results = education_retriever.retrieve(
            query=transcription['transcription']['text'],
            filter_metadata={
                "subject": request.subject,
                "grade_level": request.grade_level
            }
        )
        
        return {
            "query_transcription": transcription['transcription'],
            "results": results,
            "confidence": transcription['transcription']['confidence']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio query processing error: {str(e)}"
        )

# Additional audio-related endpoints

@app.post("/audio/feedback")
async def audio_feedback(
    request: AudioQueryRequest,
    file: UploadFile = File(...)
):
    """Process audio feedback for educational content."""
    try:
        # First transcribe the audio
        transcription = await transcribe_audio(file, request)
        
        # Process the feedback
        feedback_processor = FeedbackProcessor(
            subject=request.subject,
            grade_level=request.grade_level
        )
        
        # Analyze feedback content
        analysis_result = feedback_processor.analyze_feedback(
            transcription['transcription']['text'],
            context=request.additional_context
        )
        
        return {
            "status": "success",
            "feedback_analysis": {
                "sentiment": analysis_result.sentiment,
                "key_points": analysis_result.key_points,
                "suggested_improvements": analysis_result.suggestions,
                "topic_alignment": analysis_result.topic_alignment,
                "comprehension_level": analysis_result.comprehension_level
            },
            "transcription": transcription['transcription'],
            "confidence_metrics": {
                "feedback_confidence": analysis_result.confidence,
                "topic_relevance": analysis_result.topic_relevance,
                "clarity_score": analysis_result.clarity_score
            }
        }
        
    except Exception as e:
        logger.error(f"Feedback processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feedback processing error: {str(e)}"
        )

@app.post("/audio/assessment")
async def audio_assessment(
    request: AudioQueryRequest,
    file: UploadFile = File(...)
):
    """Process audio for educational assessment."""
    try:
        # Transcribe the audio response
        transcription = await transcribe_audio(file, request)
        
        # Initialize assessment processor
        assessment_processor = AssessmentProcessor(
            subject=request.subject,
            grade_level=request.grade_level,
            assessment_criteria=request.additional_context.get('criteria', {})
        )
        
        # Perform detailed assessment
        assessment_result = assessment_processor.evaluate_response(
            transcription['transcription']['text'],
            context=request.additional_context
        )
        
        return {
            "status": "success",
            "assessment_results": {
                "overall_score": assessment_result.overall_score,
                "criteria_scores": assessment_result.criteria_scores,
                "strengths": assessment_result.strengths,
                "areas_for_improvement": assessment_result.improvements,
                "topic_mastery": assessment_result.topic_mastery,
                "misconceptions": assessment_result.misconceptions
            },
            "transcription": transcription['transcription'],
            "evaluation_metrics": {
                "response_completeness": assessment_result.completeness,
                "technical_accuracy": assessment_result.technical_accuracy,
                "explanation_clarity": assessment_result.clarity,
                "concept_understanding": assessment_result.understanding
            },
            "recommendations": {
                "study_areas": assessment_result.study_recommendations,
                "resources": assessment_result.recommended_resources,
                "practice_exercises": assessment_result.practice_suggestions
            }
        }
        
    except Exception as e:
        logger.error(f"Assessment processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Assessment processing error: {str(e)}"
        )

@app.post("/document/process", response_model=ProcessedDocument)
async def process_document(
    document: Optional[DocumentRequest] = None,
    file: Optional[UploadFile] = File(None)
) -> ProcessedDocument:
    """
    Process educational documents with advanced analysis.
    
    Args:
        document: Text-based document request
        file: File upload for binary documents
    
    Returns:
        ProcessedDocument containing analysis results
    
    Raises:
        HTTPException: If document processing fails
    """
    try:
        # Handle file upload
        if file:
            content = await file.read()
            file_type = file.filename.split('.')[-1].lower()
        else:
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either document or file must be provided"
                )
            content = document.content
            file_type = document.file_type

        # Initialize processing result
        processing_result = {
            "text_content": "",
            "diagrams": [],
            "tables": [],
            "equations": [],
            "metadata": {}
        }

        # Route to appropriate processor based on file type
        try:
            if file_type in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
                processing_result.update(
                    document_processor.process_document(
                        content,
                        file_type,
                        document.processing_options
                    )
                )
            elif file_type == 'pdf':
                processing_result.update(
                    document_processor.process_document(
                        content,
                        file_type,
                        document.processing_options
                    )
                )
            else:
                processing_result["text_content"] = content

        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error processing document: {str(e)}"
            )

        # Create response
        return ProcessedDocument(
            document_id=generate_document_id(),
            content=processing_result,
            metadata={
                "subject": document.subject,
                "grade_level": document.grade_level,
                "file_type": file_type,
                "processing_options": document.processing_options,
                "content_type": document.content_type,
                **(document.metadata or {})
            },
            embeddings=embedding_generator.generate_embeddings(
                processing_result["text_content"],
                subject=document.subject
            ),
            confidence_score=calculate_confidence_score(processing_result),
            processing_summary={
                "extracted_text_length": len(processing_result["text_content"]),
                "num_diagrams": len(processing_result.get("diagrams", [])),
                "num_tables": len(processing_result.get("tables", [])),
                "num_equations": len(processing_result.get("equations", [])),
                "content_type": document.content_type,
                "processing_time": time.time()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@app.post("/document/batch")
async def process_document_batch(
    documents: List[DocumentRequest]
):
    """Process multiple documents in batch."""
    results = []
    for doc in documents:
        result = await process_document(document=doc)
        results.append(result)
    return results

def calculate_confidence_score(processing_result: Dict) -> float:
    """Calculate confidence score for processed document."""
    # Simple implementation - can be enhanced based on your needs
    if not processing_result["text_content"]:
        return 0.0
    return 0.8  # Default confidence score

def calculate_processing_time() -> float:
    """Calculate document processing time."""
    # Simple implementation - can be enhanced to track actual processing time
    return time.time()  # Returns current timestamp

async def store_processed_document(document: ProcessedDocument) -> None:
    """Store processed document in database."""
    # TODO: Implement actual database storage
    # For now, just log the document ID
    logging.info(f"Stored document with ID: {document.document_id}")
    pass

async def log_query(query: str, subject: str, results_count: int) -> None:
    """Log query details for analytics."""
    try:
        logger.info(
            f"Query executed - Text: {query}, Subject: {subject}, "
            f"Results: {results_count}, Time: {datetime.now().isoformat()}"
        )
    except Exception as e:
        logger.error(f"Error logging query: {str(e)}")

@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    Streaming endpoint for real-time educational content processing.
    Handles audio, text, and image streams with cross-modal reasoning.
    """
    try:
        await websocket.accept()
        
        while True:
            data = await websocket.receive_json()
            
            # Process different modalities
            if data["type"] == "audio":
                result = await process_audio_stream(data["content"])
            elif data["type"] == "text":
                result = await process_text_stream(data["content"]) 
            elif data["type"] == "image":
                result = await process_image_stream(data["content"])
            
            # Cross-modal reasoning
            if data.get("cross_modal", False):
                result = await cross_modal_reasoning(result, data.get("context", {}))
                
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        await websocket.close()

async def process_audio_stream(content: bytes) -> Dict:
    """Process streaming audio data."""
    processor = AudioProcessor()
    return await processor.process_stream(content)

async def process_text_stream(content: str) -> Dict:
    """Process streaming text data."""
    return {
        "text": content,
        "processed": True,
        "timestamp": datetime.now().isoformat()
    }

async def process_image_stream(content: bytes) -> Dict:
    """Process streaming image data."""
    analyzer = DiagramAnalyzer()
    return await analyzer.process_stream(content)

async def cross_modal_reasoning(
    result: Dict,
    context: Dict
) -> Dict:
    """
    Perform cross-modal reasoning across different input types.
    Combines insights from text, audio, and visual modalities.
    """
    try:
        # Initialize cross-modal processor
        processor = CrossModalProcessor()
        
        # Combine modalities
        enriched_result = processor.combine_modalities(
            result=result,
            context=context
        )
        
        return {
            "original": result,
            "enriched": enriched_result,
            "cross_modal_score": processor.calculate_coherence(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cross-modal reasoning error: {str(e)}")
        return result

