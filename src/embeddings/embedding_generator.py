"""
Embedding Generator Module
------------------------

Core module for generating text embeddings using transformer models.
Provides efficient and cached embedding generation with batch processing.

Features:
- Multiple model support
- Device optimization (CPU/GPU/MPS)
- Caching system
- Batch processing
- Progress tracking
- Memory optimization

Key Components:
1. Model Management: Loading and configuration
2. Embedding Generation: Single and batch processing
3. Cache System: LRU caching for efficiency
4. Device Handling: Multi-device support
5. Metadata Processing: Document enrichment

Technical Details:
- Transformer architecture
- Normalized embeddings
- Automatic device selection
- Memory-efficient batching
- Progress monitoring

Dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- numpy>=1.24.0
- tqdm>=4.65.0

Example Usage:
    generator = EmbeddingGenerator(model_name='minilm')
    embedding = generator.generate_embeddings("Sample text")
    batch_embeddings = generator.generate_embeddings(["Text 1", "Text 2"])

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
""" 

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import torch
from transformers import AutoModel, AutoTokenizer
from functools import lru_cache
import logging
from ..config.settings import EMBEDDING_CONFIG
from datetime import datetime

Vector = Union[List[float], NDArray[np.float32]]
BatchInput = Union[str, List[str]]
EmbeddingOutput = Union[Vector, List[Vector]]

class EmbeddingResult:
    vector: Vector
    metadata: Dict[str, Union[str, int, float]]

class EmbeddingGenerator:
    """Generates embeddings for text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or EMBEDDING_CONFIG
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(
            self.config["model_name"],
            cache_folder=self.config["cache_dir"]
        )
    
    async def generate(self, text: str) -> np.ndarray:
        """Generate embeddings for text."""
        return self.model.encode([text])[0]

    @lru_cache(maxsize=10000)
    def _get_embedding_cached(self, text: str) -> np.ndarray:
        """Get embedding for a single text with caching.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            max_length=384,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        # Convert to numpy and normalize
        embedding = embeddings[0].cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> Union[Vector, List[Vector]]:
        """Generate embeddings for text(s)."""
        try:
            if isinstance(texts, str):
                return self._get_embedding(texts)
            
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                embeddings = [self._get_embedding(text) for text in batch]
                all_embeddings.extend(embeddings)
                
                if show_progress:
                    self.logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
                    
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation error: {str(e)}")
            raise

    def _get_embedding(self, text: str) -> Vector:
        """Generate embedding for single text."""
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings[0].cpu().numpy()
    
    def generate_embeddings_with_metadata(
        self,
        documents: List[Dict],
        text_key: str = 'content'
    ) -> List[Dict]:
        """Generate embeddings for documents with metadata.
        
        Args:
            documents: List of document dictionaries
            text_key: Key for text content in documents
            
        Returns:
            Documents with added embeddings
        """
        # Extract texts
        texts = [doc[text_key] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, show_progress=True)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding.tolist()
            
        return documents

    def generate_embedding(
        self,
        text: str
    ) -> Union[List[float], np.ndarray]:
        """Generate embedding for text."""
        pass
        
    def batch_generate(
        self,
        texts: List[str]
    ) -> List[Union[List[float], np.ndarray]]:
        """Generate embeddings for multiple texts."""
        pass

    def normalize_vector(
        self,
        vector: Union[List[float], np.ndarray]
    ) -> Union[List[float], np.ndarray]:
        """Normalize vector to unit length."""
        pass

# Create default instance
embedding_generator = EmbeddingGenerator(**EMBEDDING_CONFIG) 

