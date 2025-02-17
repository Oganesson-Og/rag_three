"""
Embedding Configuration Module
---------------------------

Comprehensive configuration management system for embedding models and strategies,
supporting multiple educational domains and content types.

Key Features:
- Model configurations for different domains
- Pooling strategy settings
- Domain-specific optimizations
- Performance tuning parameters
- Hardware-specific configurations
- Caching policies
- Resource management

Technical Details:
- YAML-based configuration
- Environment variable support
- Dynamic configuration updates
- Model version management
- Resource allocation settings
- Caching strategies

Dependencies:
- pyyaml>=6.0.1
- pydantic>=2.5.0
- python-dotenv>=1.0.0
- typing-extensions>=4.8.0
- psutil>=5.9.0

Example Usage:
    # Load default configuration
    config = EMBEDDING_CONFIG
    
    # Get model configuration
    model_config = get_model_config('minilm')
    
    # Update configuration
    EMBEDDING_CONFIG.update({
        'batch_size': 64,
        'use_fp16': True
    })

Configuration Options:
- Model selection parameters
- Batch processing settings
- Memory management options
- Cache configuration
- Hardware optimization
- Logging preferences

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""


from typing import Dict

EMBEDDING_MODELS = {
    'minilm': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimensions': 384,
        'max_length': 384
    },
    'mpnet': {
        'name': 'sentence-transformers/all-mpnet-base-v2',
        'dimensions': 768,
        'max_length': 512
    },
    'e5': {
        'name': 'intfloat/e5-large-v2',
        'dimensions': 1024,
        'max_length': 512
    },
    'bge': {
        'name': 'BAAI/bge-large-en-v1.5',
        'dimensions': 1024,
        'max_length': 512
    },
    'instructor': {
        'name': 'hkunlp/instructor-large',
        'dimensions': 768,
        'max_length': 512
    },
    'multilingual': {
        'name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'dimensions': 768,
        'max_length': 512
    },
    'scibert': {
        'name': 'allenai/scibert_scivocab_uncased',
        'dimensions': 768,
        'max_length': 512
    },
    'specter': {
        'name': 'allenai/specter',
        'dimensions': 768,
        'max_length': 512
    }
}

EMBEDDING_CONFIG = {
    'default_model': 'minilm',
    'batch_size': 32,
    'cache_size': 10000,
    'use_fp16': True,  # Use half precision for memory efficiency
    'normalize': True,
    'pooling_strategy': 'mean',  # Options: mean, cls, max
    'cache_dir': '.cache/embeddings'
}

# Add domain-specific configurations
DOMAIN_CONFIGS = {
    'science': {
        'model': 'scibert',
        'pooling': 'weighted_mean',
        'preprocessing': ['clean_latex', 'expand_abbreviations']
    },
    'general': {
        'model': 'mpnet',
        'pooling': 'mean',
        'preprocessing': ['basic_clean']
    },
    'multilingual': {
        'model': 'multilingual',
        'pooling': 'attention',
        'preprocessing': ['normalize_unicode', 'clean_text']
    }
}

def get_model_config(model_key: str) -> Dict:
    """Get configuration for specific model."""
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available models: {list(EMBEDDING_MODELS.keys())}")
    return EMBEDDING_MODELS[model_key] 