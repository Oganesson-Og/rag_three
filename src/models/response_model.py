"""
Response Model Module
-----------------
"""

from typing import Dict, List, Union, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field

class ResponseMetadata(TypedDict):
    timestamp: datetime
    processing_time: float
    model_version: str
    query_id: str

class SearchResponse(BaseModel):
    results: List[Dict[str, Union[str, float, Dict]]]
    metadata: ResponseMetadata
    total_hits: int
    page: int = 1
    page_size: int = 10
    
    class Config:
        arbitrary_types_allowed = True
        
    @property
    def has_next_page(self) -> bool:
        return self.total_hits > self.page * self.page_size 