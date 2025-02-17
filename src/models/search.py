"""
Search Models Module
------------------

Core search models for managing educational content search operations with 
comprehensive filtering and sorting capabilities.

Key Features:
- Search filtering
- Result sorting
- Pagination support
- Score tracking
- Metadata handling
- Type validation
- Error handling

Technical Details:
- Pydantic models
- Type validation
- Filter operations
- Sort handling
- Score normalization
- Result pagination
- Error validation

Dependencies:
- pydantic>=2.0.0
- typing (standard library)
- datetime (standard library)

Example Usage:
    # Create search query
    query = SearchQuery(
        text="quantum mechanics",
        filters=[
            SearchFilter(
                field="subject",
                value="physics",
                operator="eq"
            )
        ],
        sort_by="relevance",
        limit=10
    )
    
    # Process search results
    result = SearchResult(
        id="doc123",
        score=0.95,
        content={"text": "Quantum mechanics is..."},
        metadata={"source": "textbook"}
    )

Filter Operators:
- eq (equals)
- gt (greater than)
- lt (less than)
- in (in list)
- contains (string contains)
- range (numeric range)

Sort Options:
- relevance
- timestamp
- score
- custom fields

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

class SearchFilter(BaseModel):
    """
    Search filter model.
    
    Attributes:
        field (str): Field to filter on
        value (Union[str, int, float, bool, List[Any]]): Filter value
        operator (str): Filter operator (default: "eq")
    """
    field: str = Field(..., min_length=1)
    value: Union[str, int, float, bool, List[Any]]
    operator: str = Field(default="eq")

    class Config:
        arbitrary_types_allowed = True

class SearchQuery(BaseModel):
    """
    Search query model.
    
    Attributes:
        text (str): Search query text
        filters (List[SearchFilter]): List of search filters
        sort_by (Optional[str]): Field to sort by
        sort_order (str): Sort order ("asc" or "desc")
        limit (int): Maximum number of results
        offset (int): Number of results to skip
    """
    text: str = Field(..., min_length=1)
    filters: List[SearchFilter] = Field(default_factory=list)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc")
    limit: int = Field(default=10, gt=0)
    offset: int = Field(default=0, ge=0)

    class Config:
        arbitrary_types_allowed = True

class SearchResult(BaseModel):
    """
    Search result model.
    
    Attributes:
        id (str): Unique identifier for the result
        score (float): Relevance score (0.0 to 1.0)
        content (Dict[str, Any]): Result content
        metadata (Optional[Dict[str, Any]]): Additional metadata
        timestamp (datetime): Result timestamp
    """
    id: str = Field(..., min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True 