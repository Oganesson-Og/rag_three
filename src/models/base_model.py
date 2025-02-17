"""
Base Model Module
-------------
"""

from typing import Dict, List, Union, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np

class BaseMetadata(TypedDict):
    created_at: datetime
    updated_at: datetime
    version: str
    source: str
    tags: List[str]

class BaseDocument(BaseModel):
    id: str
    content: str
    metadata: BaseMetadata = Field(default_factory=lambda: {
        'created_at': datetime.now(),
        'updated_at': datetime.now(),
        'version': '1.0',
        'source': 'unknown',
        'tags': []
    })
    embedding: Optional[Union[List[float], np.ndarray]] = None
    
    class Config:
        arbitrary_types_allowed = True 