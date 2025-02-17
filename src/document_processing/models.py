from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_size: int = 100
    respect_sentences: bool = True
    separators: List[str] = None

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    id: Optional[str] = None 

@dataclass
class PreprocessingConfig:
    remove_html: bool = True
    remove_urls: bool = False
    normalize_unicode: bool = False
    case: str = None  # 'lower', 'upper', or None 