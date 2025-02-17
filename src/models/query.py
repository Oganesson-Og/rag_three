"""
Query Models Module
-----------------

Provides data models for handling search queries, results, and metadata in the RAG system.

Key Features:
- Query representation
- Search result modeling
- Metadata management
- JSON serialization
- Dictionary conversion
- Type safety

Technical Details:
- Uses dataclasses for type safety
- Implements serialization methods
- Handles datetime conversions
- Supports metadata enrichment
- Provides factory methods

Dependencies:
- dataclasses
- datetime
- typing
- json

Example Usage:
    # Create a query
    query = Query(
        text="quantum physics",
        filters={"subject": "physics"},
        max_results=5
    )
    
    # Create search result
    result = QueryResult(
        text="Example text",
        score=0.95,
        metadata=SearchMetadata(
            source="textbook_1",
            confidence=0.95
        )
    )
    
    # Convert to dictionary
    query_dict = query.to_dict()
    
    # Create from dictionary
    new_query = Query.from_dict(query_dict)

Data Models:
- SearchMetadata: Result metadata
- Query: Search query parameters
- QueryResult: Search result with metadata

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from dataclasses import dataclass, field
import json

class QueryMetadata(BaseModel):
    """Query metadata model."""
    timestamp: datetime = Field(default_factory=datetime.now)
    subject: str
    grade_level: str
    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class Query(BaseModel):
    """Query model."""
    text: str = Field(..., min_length=1)
    metadata: Optional[QueryMetadata] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, gt=0)

    class Config:
        arbitrary_types_allowed = True

class QueryResult(BaseModel):
    """Query result model."""
    text: str = Field(..., min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

@dataclass
class SearchMetadata:
    """Metadata for search results"""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    confidence: float = 0.0
    relevance_score: float = 0.0
    context_window: Optional[int] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None
    model_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "context_window": self.context_window,
            "additional_info": self.additional_info,
            "processing_time": self.processing_time,
            "model_info": self.model_info
        }

@dataclass
class Query:
    """Represents a search query"""
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 10
    min_confidence: float = 0.0
    context_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary"""
        return {
            "text": self.text,
            "filters": self.filters,
            "metadata": self.metadata,
            "max_results": self.max_results,
            "min_confidence": self.min_confidence,
            "context_size": self.context_size
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create Query from dictionary"""
        return cls(**data)

@dataclass
class QueryResult:
    """Represents a search result"""
    text: str
    score: float
    metadata: SearchMetadata
    source_document: Optional[str] = None
    context: Optional[str] = None
    chunk_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata.to_dict(),
            "source_document": self.source_document,
            "context": self.context,
            "chunk_id": self.chunk_id
        }

    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """Create QueryResult from dictionary"""
        metadata_dict = data.pop("metadata")
        metadata = SearchMetadata(**metadata_dict)
        return cls(metadata=metadata, **data) 