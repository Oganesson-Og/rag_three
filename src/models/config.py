from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    model_name: str = Field(..., min_length=1)
    dimension: int = Field(gt=0)
    batch_size: int = Field(default=32, gt=0)
    device: str = "cpu"
    additional_params: Optional[Dict[str, Any]] = None

class ProcessingConfig(BaseModel):
    chunk_size: int = Field(default=1000, gt=0)
    overlap: int = Field(default=200, ge=0)
    clean_text: bool = True
    max_length: Optional[int] = Field(default=None, gt=0)
    preserve_formatting: bool = False 