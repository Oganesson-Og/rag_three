"""
Query Model Module
--------------

Core query models for managing educational RAG queries with metadata tracking
and result handling.

Key Features:
- Query metadata
- Result management
- Vector handling
- Filter support
- Score tracking
- Session management
- Type validation

Technical Details:
- Pydantic models
- Type validation
- Vector operations
- Metadata tracking
- Score normalization
- Session handling
- Error validation

Dependencies:
- pydantic>=2.0.0
- numpy>=1.24.0
- typing (standard library)
- datetime (standard library)

Example Usage:
    # Create query
    query = Query(
        text="What is quantum entanglement?",
        metadata={
            'user_id': 'user123',
            'session_id': 'session456',
            'filters': {'subject': 'physics'}
        },
        top_k=5
    )
    
    # Process results
    results = [
        QueryResult(
            text="Quantum entanglement is...",
            score=0.95,
            metadata={'source': 'textbook'},
            vector=[0.1, 0.2, 0.3]
        )
    ]

Query Types:
- Educational Queries
- Research Questions
- Assessment Items
- Clarification Requests
- Topic Exploration

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Union, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np

class QueryMetadata(TypedDict):
    """
    Query metadata type.
    
    Attributes:
        timestamp (datetime): Query timestamp
        user_id (Optional[str]): User identifier
        session_id (Optional[str]): Session identifier
        filters (Dict): Query filters
    """
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    filters: Dict[str, Union[str, int, float, List[str]]]

class QueryResult(TypedDict):
    """
    Query result type.
    
    Attributes:
        text (str): Result text content
        score (float): Relevance score
        metadata (Dict): Result metadata
        vector (Optional[Union[List[float], np.ndarray]]): Result vector
    """
    text: str
    score: float
    metadata: Dict[str, Union[str, int, float]]
    vector: Optional[Union[List[float], np.ndarray]]

class Query(BaseModel):
    """
    Query model.
    
    Attributes:
        text (str): Query text
        metadata (QueryMetadata): Query metadata
        top_k (int): Number of results to return
    """
    text: str
    metadata: QueryMetadata = Field(default_factory=lambda: {
        'timestamp': datetime.now(),
        'filters': {}
    })
    top_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True 