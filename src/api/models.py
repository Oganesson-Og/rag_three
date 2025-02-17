"""
API Models Module
-------------

Pydantic models for handling API requests and responses in the educational RAG system.
These models provide data validation and serialization for various API endpoints.

Key Features:
- Request/Response model validation
- Educational query handling
- Document processing support
- Batch operation capabilities
- Error handling structures
- Audio processing support

Technical Details:
- Built with Pydantic BaseModel
- Type validation and coercion
- Default value handling
- Configurable field constraints
- Timestamp automation
- Arbitrary type support

Dependencies:
- pydantic>=2.0.0
- python-dateutil>=2.8.2

Example Usage:
    # Query Request
    query = QueryRequest(
        query="What is photosynthesis?",
        subject="biology",
        level="high_school",
        filters={"topic": "cellular_processes"},
        max_results=5
    )
    
    # Document Processing
    doc_request = DocumentRequest(
        document_id="doc123",
        content="Document content here...",
        metadata={"subject": "physics", "grade": "11"}
    )
    
    # Batch Processing
    batch = BatchQueryRequest(
        queries=[query1, query2],
        batch_id="batch_001"
    )

Model Categories:
- Query Models (QueryRequest, QueryResponse)
- Document Models (DocumentRequest, DocumentResponse)
- Error Handling (ErrorResponse)
- Batch Processing (BatchQueryRequest)
- Audio Processing (AudioRequest)

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ContentType(Enum):
    """Educational content type classification."""
    SYLLABUS = "syllabus"
    STUDY_MATERIAL = "study_material"
    EXERCISE = "exercise"
    REVISION_QUESTION = "revision_question"
    REVISION_ANSWER = "revision_answer"
    ASSESSMENT = "assessment"
    SUPPLEMENTARY = "supplementary"

class QueryRequest(BaseModel):
    """
    Query request model for educational content retrieval.
    
    Attributes:
        query (str): The search query text
        subject (str): Academic subject (e.g., "physics", "mathematics")
        level (str): Educational level (e.g., "high_school", "university")
        filters (Dict[str, Any]): Optional query filters
        max_results (int): Maximum number of results to return (default: 5)
    
    Example:
        query = QueryRequest(
            query="Newton's Laws",
            subject="physics",
            level="high_school",
            filters={"topic": "mechanics"},
            max_results=3
        )
    """
    query: str = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    level: str = Field(..., min_length=1)
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = Field(default=5, gt=0)

    class Config:
        arbitrary_types_allowed = True

class QueryResponse(BaseModel):
    """Query response model."""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

class DocumentRequest(BaseModel):
    """
    Document processing request model for content ingestion.
    
    Attributes:
        document_id (str): Unique identifier for the document
        content (str): Document content, limited to 100,000 characters
        metadata (Dict[str, Any]): Additional document metadata
        content_type (ContentType): Type of educational content
    
    Example:
        doc = DocumentRequest(
            document_id="phys_101_ch1",
            content="Chapter 1 content...",
            metadata={
                "subject": "physics",
                "chapter": 1,
                "topic": "kinematics"
            },
            content_type=ContentType.STUDY_MATERIAL
        )
    """
    document_id: str = Field(..., min_length=1)
    content: str = Field(..., max_length=100000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_type: ContentType = Field(
        ...,
        description="Type of educational content"
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

class DocumentResponse(BaseModel):
    """Document response model."""
    document_id: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_type: ContentType
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    error_code: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class BatchQueryRequest(BaseModel):
    """
    Batch query request model for processing multiple queries.
    
    Attributes:
        queries (List[QueryRequest]): List of query requests to process
        batch_id (Optional[str]): Optional identifier for the batch
    
    Example:
        batch = BatchQueryRequest(
            queries=[
                QueryRequest(query="Topic 1", subject="math"),
                QueryRequest(query="Topic 2", subject="physics")
            ],
            batch_id="batch_001"
        )
    """
    queries: List[QueryRequest]
    batch_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "queries": [
                    {
                        "query": "What is quantum mechanics?",
                        "filters": {"subject": "physics"},
                        "top_k": 5
                    }
                ],
                "batch_id": "batch_001"
            }
        }

class AudioRequest(BaseModel):
    """Audio processing request model."""
    language: str = Field(default="en")
    task_type: str = Field(default="transcribe")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True 