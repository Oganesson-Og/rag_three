"""
Embedding Models
-------------
"""

from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime

class EmbeddingMetadata(BaseModel):
    model_name: str
    dimension: int
    created_at: datetime = Field(default_factory=datetime.now)

class EmbeddingVector(BaseModel):
    vector: Union[List[float], np.ndarray]
    metadata: Optional[EmbeddingMetadata] = None
    
    class Config:
        arbitrary_types_allowed = True 