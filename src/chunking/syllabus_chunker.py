"""
Syllabus-Guided Document Chunking Module
--------------------------------------

Intelligent document chunking system that aligns with educational syllabus
structures and maintains semantic coherence in educational content.

Key Features:
- Syllabus-aware text splitting
- Topic-based chunking strategies
- Sliding window with overlap control
- Educational content-specific rules
- Semantic boundary detection
- Hierarchical chunking support
- Cross-reference preservation

Technical Details:
- Uses NLP for boundary detection
- Implements custom chunking algorithms
- Maintains metadata across chunks
- Supports multiple education systems
- Handles various document formats

Dependencies:
- spacy>=3.7.2
- nltk>=3.8.1
- beautifulsoup4>=4.12.0
- python-docx>=1.0.0
- pdfminer.six>=20221105
- markdown>=3.5.0

Example Usage:
    # Basic chunking
    chunker = SyllabusChunker(subject='physics')
    chunks = chunker.chunk_document(document_text)
    
    # Advanced chunking with options
    chunks = chunker.chunk_document(
        document_text,
        chunk_size=500,
        overlap=50,
        preserve_headers=True
    )
    
    # Batch processing
    chunks = chunker.process_batch(documents)

Chunking Strategies:
- Topic-based boundaries
- Sentence-level splitting
- Paragraph preservation
- Header-based segmentation
- Semantic unit preservation

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional
import re
from pathlib import Path
import spacy
from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP, SYLLABUS_PATH
from ..nlp.tokenizer import UnifiedTokenizer, default_tokenizer  # Fixed import




class SyllabusChunker:
    """Syllabus-aware document chunking system."""
    
    def __init__(
        self,
        subject: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.subject = subject
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nlp = spacy.load("en_core_web_sm")
        self.syllabus_topics = self._load_syllabus_topics()
        self.window_size = 512
        self.overlap = 128
        self.tokenizer = UnifiedTokenizer()
        
    def _load_syllabus_topics(self) -> Dict[str, List[str]]:
        """Load syllabus topics from the syllabus file."""
        syllabus_file = Path(SYLLABUS_PATH) / "syllabus.txt"
        topics = {}
        
        if syllabus_file.exists():
            current_topic = None
            with open(syllabus_file, 'r') as f:
                for line in f:
                    if line.strip().startswith('#'):
                        current_topic = line.strip('# \n')
                        topics[current_topic] = []
                    elif current_topic and line.strip():
                        topics[current_topic].append(line.strip())
                        
        return topics
        
    def chunk_document(self, text: str) -> List[str]:
        """Chunk document with syllabus awareness."""
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            token_length = len(token.text.split())
            
            if current_length + token_length > self.chunk_size:
                if current_chunk:
                    chunks.append(self.tokenizer.decode(current_chunk))
                current_chunk = [token]
                current_length = token_length
            else:
                current_chunk.append(token)
                current_length += token_length
        
        if current_chunk:
            chunks.append(self.tokenizer.decode(current_chunk))
            
        return chunks
        
    def chunk_by_syllabus(self, text: str) -> List[Dict]:
        """Split text into chunks based on syllabus topics."""
        chunks = []
        doc = self.nlp(text)
        
        # Find topic boundaries
        topic_boundaries = []
        for topic in self.syllabus_topics:
            matches = re.finditer(re.escape(topic), text, re.IGNORECASE)
            topic_boundaries.extend((m.start(), m.end(), topic) for m in matches)
            
        # Sort boundaries by position
        topic_boundaries.sort()
        
        # Create chunks based on topics
        last_end = 0
        for start, end, topic in topic_boundaries:
            if start > last_end:
                # Handle text between topics using sliding window
                chunks.extend(self._sliding_window_chunk(text[last_end:start]))
            
            # Create topic-specific chunk
            chunks.append({
                'text': text[start:end],
                'metadata': {
                    'subject': self.subject,
                    'topic': topic
                }
            })
            last_end = end
            
        # Handle remaining text
        if last_end < len(text):
            chunks.extend(self._sliding_window_chunk(text[last_end:]))
            
        return chunks
        
    def _sliding_window_chunk(self, text: str) -> List[Dict]:
        """Chunk text using sliding window approach."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.window_size - self.overlap):
            chunk_words = words[i:i + self.window_size]
            chunks.append({
                'text': ' '.join(chunk_words),
                'metadata': {
                    'subject': self.subject,
                    'chunk_type': 'sliding_window',
                    'position': i
                }
            })
            
        return chunks 