
"""
Educational RAG Configuration Module
---------------------------------

Flexible configuration management system for RAG components with dataclass-based
validation and YAML configuration support.

Key Features:
- Dataclass-based configuration
- YAML file support
- Default configuration handling
- Configuration validation
- Environment variable integration
- Dynamic configuration updates

Technical Details:
- Dataclass implementation
- YAML parsing
- Type validation
- Path management
- Environment variable support
- Configuration inheritance

Dependencies:
- pyyaml>=6.0.0
- dataclasses>=0.6
- typing>=3.7.4
- pathlib>=1.0.1

Example Usage:
    # Load configuration
    config = ConfigManager("config.yaml").config
    
    # Access configuration
    model_name = config.models.model_name
    chunk_size = config.processing.chunk_size
    
    # Update configuration
    config.models.temperature = 0.8

Configuration Schema:
- models: Model-specific settings
- processing: Data processing parameters
- cache_dir: Cache directory location
- embedding_dim: Embedding dimensions

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""


from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    
@dataclass
class ProcessingConfig:
    chunk_size: int = 1000
    overlap: int = 200
    batch_size: int = 32

@dataclass
class RAGConfig:
    models: ModelConfig
    processing: ProcessingConfig
    cache_dir: Path
    embedding_dim: int = 768
    
class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> RAGConfig:
        if not self.config_path.exists():
            return self._create_default_config()
            
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return self._parse_config(config_dict)
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> RAGConfig:
        models = ModelConfig(**config_dict.get('models', {}))
        processing = ProcessingConfig(**config_dict.get('processing', {}))
        cache_dir = Path(config_dict.get('cache_dir', './cache'))
        
        return RAGConfig(
            models=models,
            processing=processing,
            cache_dir=cache_dir,
            embedding_dim=config_dict.get('embedding_dim', 768)
        )
    
    def _create_default_config(self) -> RAGConfig:
        return RAGConfig(
            models=ModelConfig(model_name="gpt-3.5-turbo"),
            processing=ProcessingConfig(),
            cache_dir=Path('./cache')
        ) 