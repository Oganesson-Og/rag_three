"""
Base Text Chunking Module
-----------------------

Core text chunking functionality providing base implementations for
document splitting and chunk management.

Key Features:
- Text splitting
- Chunk management
- Metadata handling
- Size control
- Overlap handling
- ID generation
- Statistics tracking

Technical Details:
- Regex-based splitting
- MD5 hashing for IDs
- Metadata preservation
- Chunk validation
- Size optimization
- Error handling

Dependencies:
- re (standard library)
- typing (standard library)
- dataclasses (standard library)
- hashlib (standard library)
- logging (standard library)

Example Usage:
    # Basic chunking
    chunker = BaseChunker()
    chunks = chunker.chunk_text(document_text)
    
    # Chunking with metadata
    chunks = chunker.chunk_text(
        text=document_text,
        metadata={"source": "textbook"}
    )

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
import logging
from hashlib import md5
from .utils.text import clean_text, split_into_sentences

@dataclass
class Chunk:
    """Represents a document chunk."""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    index: Optional[int] = None
    
    def __post_init__(self):
        """Validate chunk after initialization."""
        if not self.text or len(self.text.strip()) == 0:
            raise ValueError("Chunk text cannot be empty")
        if self.metadata is None:
            self.metadata = {}

    def validate(self) -> bool:
        """Basic chunk validation"""
        return bool(self.text and len(self.text.strip()) > 0)

    @property
    def length(self) -> int:
        return len(self.text)

class BaseChunker:
    """Base chunking functionality."""
    
    def __init__(
        self, 
        chunk_size: int = 512,
        overlap: int = 50,
        respect_sentences: bool = True,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)

    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into chunks."""
        if not text:
            return []

        metadata = metadata or {}
        text = clean_text(text)
        
        try:
            if self.respect_sentences:
                return self._chunk_by_sentences(text, metadata)
            return self._chunk_by_size(text, metadata)
        except Exception as e:
            self.logger.error(f"Error chunking text: {str(e)}")
            raise

    def _chunk_by_sentences(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split text respecting sentence boundaries."""
        sentences = split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunk = self._create_chunk(
                    ' '.join(current_chunk),
                    start_idx,
                    metadata
                )
                if chunk.validate():
                    chunks.append(chunk)
                current_chunk = []
                current_length = 0
                start_idx += len(chunk.text)
            
            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunk = self._create_chunk(
                ' '.join(current_chunk),
                start_idx,
                metadata
            )
            if chunk.validate():
                chunks.append(chunk)

        return chunks

    def _chunk_by_size(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split text by fixed size."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            if end > text_length:
                end = text_length
            
            chunk = self._create_chunk(
                text[start:end],
                start,
                metadata
            )
            if chunk.validate():
                chunks.append(chunk)
            
            start = end - self.overlap

        return self._merge_small_chunks(chunks)

    def _create_chunk(
        self, 
        text: str, 
        start_idx: int,
        metadata: Dict[str, Any]
    ) -> Chunk:
        """Create a new chunk."""
        return Chunk(
            text=text,
            start_idx=start_idx,
            end_idx=start_idx + len(text),
            metadata=metadata.copy(),
            chunk_id=self._generate_chunk_id(text)
        )

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge small chunks."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if current.length < self.min_chunk_size:
                current = self._merge_chunks(current, next_chunk)
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged

    def _merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks."""
        return Chunk(
            text=chunk1.text + " " + chunk2.text,
            start_idx=chunk1.start_idx,
            end_idx=chunk2.end_idx,
            metadata=chunk1.metadata,
            chunk_id=self._generate_chunk_id(chunk1.text + chunk2.text)
        )

    def _generate_chunk_id(self, text: str) -> str:
        """Generate unique chunk ID."""
        return md5(text.encode()).hexdigest()
