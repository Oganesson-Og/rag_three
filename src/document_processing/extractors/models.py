from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class Document:
    """Document model for extraction stage."""
    content: Union[str, bytes]
    modality: Optional[str] = None
    doc_info: dict = None
    
    def __post_init__(self):
        if self.doc_info is None:
            self.doc_info = {} 