from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    cache_dir: Optional[str] = None 