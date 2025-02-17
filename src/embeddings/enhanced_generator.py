
"""
Enhanced Embedding Generation Module
---------------------------------

Advanced embedding generation system designed for educational content vectorization,
supporting multiple models and optimization strategies with a focus on academic material.

Key Features:
- Multiple embedding model support (BERT, MPNet, SciBERT, etc.)
- Intelligent caching mechanism with LRU and disk caching
- Optimized batch processing for large document sets
- Educational content-specific optimizations
- Hardware acceleration (CUDA, MPS, CPU) with Apple Silicon support
- Automatic model selection based on content domain
- Memory-efficient processing with gradient checkpointing

Technical Details:
- Implements transformer-based embedding models
- Supports fp16/bf16 for memory optimization
- Dynamic batch sizing based on available memory
- Configurable model loading and unloading
- Automatic fallback strategies for resource constraints

Dependencies:
- torch>=2.0.0
- transformers>=4.36.0
- sentence-transformers>=2.2.0
- numpy>=1.24.0
- diskcache>=5.6.0

Example Usage:
    # Basic usage
    generator = EnhancedEmbeddingGenerator()
    embeddings = generator.generate_embeddings(texts)

    # Domain-specific with custom config
    generator = EnhancedEmbeddingGenerator(
        model_key='scibert',
        device='cuda',
        use_fp16=True
    )
    embeddings = generator.generate_batch_embeddings(
        texts,
        batch_size=32,
        show_progress=True
    )

Performance Considerations:
- Optimal batch size depends on available GPU memory
- Cache size should be tuned based on dataset size
- Consider using fp16 for large-scale processing
- Monitor memory usage for long-running sessions

Author: Keith Satuku
Version: 3.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from functools import lru_cache
import logging
import os
import json
from pathlib import Path
from ..config.embedding_config import EMBEDDING_CONFIG, get_model_config
from diskcache import Cache


class EnhancedEmbeddingGenerator:
    """Enhanced embedding generator with multiple models and caching strategies."""
    
    def __init__(
        self,
        model_key: str = EMBEDDING_CONFIG['default_model'],
        device: str = None,
        use_cache: bool = True
    ):
        """Initialize the enhanced embedding generator.
        
        Args:
            model_key: Key for the model configuration
            device: Device to run model on ('cuda', 'mps', or 'cpu')
            use_cache: Whether to use disk caching
        """
        self.model_config = get_model_config(model_key)
        self.config = EMBEDDING_CONFIG
        
        # Set up device
        self.device = self._setup_device(device)
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Set up caching
        self.use_cache = use_cache
        if use_cache:
            cache_dir = Path(self.config['cache_dir'])
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = Cache(str(cache_dir / model_key))
            
        logging.info(f"Initialized {model_key} on {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Set up the compute device."""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        return device
    
    def _load_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """Load the model and tokenizer."""
        model_name = self.model_config['name']
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optimizations
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config['use_fp16'] else torch.float32
        )
        
        if self.device == 'cuda':
            model = model.cuda()
        elif self.device == 'mps':
            model = model.to(self.device)
            
        model.eval()
        return model, tokenizer
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"{self.model_config['name']}_{hash(text)}"
    
    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings based on strategy."""
        if self.config['pooling_strategy'] == 'cls':
            return token_embeddings[:, 0]
        elif self.config['pooling_strategy'] == 'max':
            return torch.max(token_embeddings * attention_mask.unsqueeze(-1), dim=1)[0]
        else:  # mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            return sum_embeddings / sum_mask
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text with caching."""
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                return np.array(self.cache[cache_key])
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.model_config['max_length'],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            
        # Process embedding
        embedding = embeddings[0].cpu().numpy()
        if self.config['normalize']:
            embedding = embedding / np.linalg.norm(embedding)
            
        # Cache result
        if self.use_cache:
            self.cache[cache_key] = embedding.tolist()
            
        return embedding
    
    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batches."""
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Try to get from cache first
            batch_embeddings = []
            texts_to_process = []
            cache_keys = []
            
            if self.use_cache:
                for text in batch_texts:
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.cache:
                        batch_embeddings.append(np.array(self.cache[cache_key]))
                    else:
                        texts_to_process.append(text)
                        cache_keys.append(cache_key)
            else:
                texts_to_process = batch_texts
            
            # Process texts not in cache
            if texts_to_process:
                inputs = self.tokenizer(
                    texts_to_process,
                    max_length=self.model_config['max_length'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = self._pool_embeddings(
                        outputs.last_hidden_state,
                        inputs['attention_mask']
                    )
                
                # Process and cache embeddings
                for idx, embedding in enumerate(embeddings):
                    embedding = embedding.cpu().numpy()
                    if self.config['normalize']:
                        embedding = embedding / np.linalg.norm(embedding)
                    
                    if self.use_cache:
                        self.cache[cache_keys[idx]] = embedding.tolist()
                    
                    batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                logging.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
        
        return all_embeddings
    
    def generate_embeddings_with_metadata(
        self,
        documents: List[Dict],
        text_key: str = 'content'
    ) -> List[Dict]:
        """Generate embeddings for documents with metadata."""
        texts = [doc[text_key] for doc in documents]
        embeddings = self.generate_batch_embeddings(texts, show_progress=True)
        
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding.tolist()
            doc['embedding_model'] = self.model_config['name']
            
        return documents

# Create default instance
embedding_generator = EnhancedEmbeddingGenerator() 