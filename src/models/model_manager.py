"""
Model Manager Module
-----------------

Comprehensive model management system for handling model selection, loading,
and fallback mechanisms in educational RAG applications.

Key Features:
- Model selection
- Automatic loading
- Fallback handling
- Cache management
- Domain adaptation
- Resource optimization
- Error recovery

Technical Details:
- Model caching
- Memory management
- Lazy loading
- GPU optimization
- Error handling
- Version control
- Resource cleanup

Dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- sentence-transformers>=2.2.0
- whisper>=1.0.0
- typing (standard library)
- pathlib (standard library)
- logging (standard library)

Example Usage:
    # Initialize manager
    manager = ModelManager(cache_dir=Path(".cache/models"))
    
    # Get model for specific task and domain
    model = manager.get_model(
        task="embedding",
        domain="science",
        fallback="instructor-base"
    )
    
    # Use model
    embeddings = model.encode("Sample text")
    
    # Cleanup
    manager.cleanup()

Supported Tasks:
- Embedding Generation
- Text Classification
- Speech Transcription
- Language Understanding
- Content Generation

Domain Support:
- Science
- Mathematics
- History
- General Education
- Cross-Domain

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Optional, Any
import torch
from pathlib import Path
import logging
import whisper
from transformers import AutoModel, WhisperModel
from sentence_transformers import SentenceTransformer
from ..config.embedding_config import EMBEDDING_MODELS, get_model_config

class ModelManager:
    """Manages model selection and integration."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(".cache/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.loaded_models = {}
        
    def get_model(
        self,
        task: str,
        domain: str,
        fallback: Optional[str] = None
    ) -> Any:
        """Get appropriate model for task and domain."""
        try:
            model_key = self._select_model(task, domain)
            
            if model_key in self.loaded_models:
                return self.loaded_models[model_key]
                
            model_config = get_model_config(model_key)
            model = self._load_model(model_key, model_config)
            
            self.loaded_models[model_key] = model
            return model
            
        except Exception as e:
            self.logger.error(f"Model loading error: {str(e)}")
            if fallback:
                self.logger.info(f"Falling back to {fallback}")
                return self._load_fallback_model(fallback)
            raise
            
    def _select_model(self, task: str, domain: str) -> str:
        """
        Select appropriate model based on task and domain.
        
        Args:
            task: The task type (e.g., 'embedding', 'classification', 'transcription')
            domain: The domain/subject area (e.g., 'science', 'math', 'history')
            
        Returns:
            str: The selected model key
        """
        model_mapping = {
            'embedding': {
                'science': 'scibert',
                'math': 'instructor-xl',
                'history': 'mpnet',
                'default': 'instructor-base'
            },
            'classification': {
                'science': 'roberta-large',
                'math': 'deberta-v3',
                'history': 'bert-base',
                'default': 'roberta-base'
            },
            'transcription': {
                'default': 'whisper-large-v3'
            }
        }
        
        # Get task-specific mapping
        task_models = model_mapping.get(task, {})
        
        # Select domain-specific model or default
        model_key = task_models.get(domain, task_models.get('default'))
        
        if not model_key:
            raise ValueError(f"No suitable model found for task '{task}' and domain '{domain}'")
            
        return model_key
        
    def _load_model(self, model_key: str, config: Dict) -> Any:
        """
        Load model with configuration.
        
        Args:
            model_key: The model identifier
            config: Model configuration dictionary
            
        Returns:
            The loaded model instance
        """
        try:
            # Check if model files exist in cache
            model_path = self.cache_dir / model_key
            
            if model_path.exists():
                self.logger.info(f"Loading model {model_key} from cache")
                return self._load_from_cache(model_key, model_path, config)
            
            # Download and load model
            self.logger.info(f"Downloading model {model_key}")
            model = self._download_and_load_model(model_key, config)
            
            # Cache the model if specified in config
            if config.get('cache_model', True):
                self._save_to_cache(model, model_key, model_path)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_key}: {str(e)}")
            raise
        
    def _load_fallback_model(self, model_key: str) -> Any:
        """
        Load fallback model.
        
        Args:
            model_key: The fallback model identifier
            
        Returns:
            The loaded fallback model instance
        """
        fallback_configs = {
            'instructor-base': {
                'model_type': 'embedding',
                'dimension': 768,
                'quantization': 'int8'
            },
            'roberta-base': {
                'model_type': 'classification',
                'quantization': 'int8'
            },
            'whisper-base': {
                'model_type': 'transcription',
                'language': ['en'],
                'quantization': 'int8'
            }
        }
        
        config = fallback_configs.get(model_key, {})
        if not config:
            raise ValueError(f"No fallback configuration found for model {model_key}")
            
        # Load lightweight version with minimal configuration
        config['optimize_memory'] = True
        config['cache_model'] = False
        
        return self._load_model(model_key, config)

    def _load_from_cache(self, model_key: str, cache_path: Path, config: Dict) -> Any:
        """Load model from local cache."""
        try:
            if model_key.startswith('instructor'):
                return SentenceTransformer(str(cache_path), **config)
            elif model_key.startswith('whisper'):
                return whisper.load_model(cache_path, **config)
            else:
                return AutoModel.from_pretrained(cache_path, **config)
        except Exception as e:
            self.logger.error(f"Cache loading error for {model_key}: {str(e)}")
            raise

    def _download_and_load_model(self, model_key: str, config: Dict) -> Any:
        """Download and load model from source."""
        try:
            if model_key.startswith('instructor'):
                return SentenceTransformer(f"hkunlp/{model_key}", **config)
            elif model_key.startswith('whisper'):
                return whisper.load_model(model_key, **config)
            else:
                return AutoModel.from_pretrained(f"huggingface/{model_key}", **config)
        except Exception as e:
            self.logger.error(f"Model download error for {model_key}: {str(e)}")
            raise

    def _save_to_cache(self, model: Any, model_key: str, cache_path: Path) -> None:
        """Save model to local cache."""
        try:
            model.save_pretrained(cache_path)
            self.logger.info(f"Model {model_key} saved to cache")
        except Exception as e:
            self.logger.warning(f"Failed to cache model {model_key}: {str(e)}")
        
    def cleanup(self):
        """Clean up loaded models."""
        self.loaded_models.clear()
        torch.cuda.empty_cache() 