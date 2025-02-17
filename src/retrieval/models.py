from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SearchConfig:
    field_weights: Dict[str, float] = None
    min_score: float = 0.0
    max_results: int = 10
    use_cache: bool = True

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None 