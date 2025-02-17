"""
Audio Models
---------
"""

from typing import Dict, List, Union, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class AudioConfig(BaseModel):
    model_name: str = "base"
    sample_rate: int = 16000
    language: Optional[str] = None
    device: str = "cpu"

class TranscriptionResult(BaseModel):
    text: str
    segments: List[Dict[str, Union[str, float]]]
    language: str
    metadata: Dict = Field(default_factory=dict) 