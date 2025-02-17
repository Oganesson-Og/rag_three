"""
Document Models Module
--------------------

Core document models for managing educational content with metadata tracking
and embedding support.

Key Features:
- Document metadata
- Content management
- Embedding storage
- Language support
- Category tracking
- Tag management
- Text processing

Technical Details:
- Pydantic models
- Type validation
- Embedding handling
- Text truncation
- Metadata tracking
- Age calculation
- Format conversion

Dependencies:
- pydantic>=2.0.0
- numpy>=1.24.0
- typing (standard library)
- datetime (standard library)

Example Usage:
    # Create document
    doc = Document(
        id="doc123",
        text="Sample educational content...",
        metadata=DocumentMetadata(
            source="textbook",
            language="en",
            category="physics",
            tags=["quantum", "mechanics"]
        )
    )
    
    # Access metadata
    print(f"Age: {doc.metadata.age_hours} hours")
    print(f"Created: {doc.metadata.format_created_at()}")

Document Types:
- Educational Content
- Student Submissions
- Reference Materials
- Assessment Items
- Learning Resources

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
from numpy.typing import NDArray

class DocumentMetadata(BaseModel):
    """
    Document metadata model.
    
    Attributes:
        source (str): Source of the document
        created_at (datetime): Creation timestamp
        language (str): Document language code
        category (Optional[str]): Document category
        tags (List[str]): List of tags
    """
    source: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.now)
    language: str = "en"
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    def format_created_at(self) -> str:
        """Format creation timestamp as string."""
        return self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        
    @property
    def age_hours(self) -> float:
        """Calculate document age in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600

class Document(BaseModel):
    """
    Document model.
    
    Attributes:
        id (str): Unique document identifier
        text (str): Document content
        metadata (DocumentMetadata): Associated metadata
        embedding (Optional[Union[List[float], NDArray[np.float32]]]): Vector embedding
    """
    id: str = Field(..., min_length=1)
    text: str = Field(..., max_length=100000)
    metadata: DocumentMetadata
    embedding: Optional[Union[List[float], NDArray[np.float32]]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def text_length(self) -> int:
        """Get text length."""
        return len(self.text)
        
    def get_truncated_text(self, max_length: int) -> str:
        """
        Get truncated version of text.
        
        Args:
            max_length: Maximum length of returned text
            
        Returns:
            str: Truncated text with ellipsis if needed
        """
        return self.text[:max_length] + "..." if len(self.text) > max_length else self.text 