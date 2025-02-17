"""
RAG Configuration Module
----------------------

Centralized configuration management for the RAG pipeline, ensuring consistency
across different modalities and components.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging
from pydantic import BaseModel, Field, validator
import os

logger = logging.getLogger(__name__)

class AudioConfig(BaseModel):
    """Audio processing configuration."""
    whisper_model_size: str = Field(default="base", description="Whisper model size")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    chunk_duration_ms: int = Field(default=30000, description="Chunk duration in ms")
    cache_audio: bool = Field(default=True, description="Whether to cache audio files")
    supported_formats: list = Field(
        default=["mp3", "wav", "m4a", "flac", "ogg"],
        description="Supported audio formats"
    )

class EmbeddingConfig(BaseModel):
    """Embedding generation configuration."""
    model_name: str = Field(default="sentence-transformers/all-mpnet-base-v2")
    dimension: int = Field(default=768)
    batch_size: int = Field(default=32)
    device: str = Field(default="cpu")
    normalize: bool = Field(default=True)

class CacheConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = Field(default=True)
    cache_dir: Path = Field(default=Path("./cache"))
    vector_cache_size: int = Field(default=10000)
    document_cache_size: int = Field(default=1000)
    ttl_seconds: Optional[int] = Field(default=86400)  # 24 hours

class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    chunk_type: str = Field(default="token")
    respect_sentences: bool = Field(default=True)

class RAGConfig(BaseModel):
    """
    Main RAG configuration.
    
    Attributes:
        audio (AudioConfig): Audio processing settings
        embedding (EmbeddingConfig): Embedding generation settings
        cache (CacheConfig): Caching settings
        chunking (ChunkingConfig): Document chunking settings
        models (Dict): Model configurations
        prompts (Dict): Prompt templates and settings
    """
    audio: AudioConfig = Field(default_factory=AudioConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    models: Dict[str, Any] = Field(default_factory=dict)
    prompts: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('cache')
    def validate_cache_dir(cls, v):
        """Ensure cache directory exists."""
        if v.enabled:
            v.cache_dir.mkdir(parents=True, exist_ok=True)
        return v

class ConfigManager:
    """
    Manages RAG configuration loading and validation.
    
    Attributes:
        config (RAGConfig): Validated configuration
        env (str): Environment name
    """
    
    DEFAULT_CONFIG_PATH = Path("config/rag_config.yaml")
    
    def __init__(self, config_path: Optional[Path] = None, env: str = "development"):
        self.env = env
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        
    def _load_config(self) -> RAGConfig:
        """Load and validate configuration."""
        try:
            # Load base config
            config_dict = self._load_yaml(self.config_path)
            
            # Load environment-specific config
            env_config_path = self.config_path.parent / f"rag_config.{self.env}.yaml"
            if env_config_path.exists():
                env_config = self._load_yaml(env_config_path)
                config_dict = self._deep_merge(config_dict, env_config)
            
            # Load environment variables
            config_dict = self._apply_env_vars(config_dict)
            
            # Validate configuration
            return RAGConfig(**config_dict)
            
        except Exception as e:
            logger.error(f"Configuration loading failed: {str(e)}", exc_info=True)
            raise
            
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {str(e)}")
            return {}
            
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result:
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
        
    @staticmethod
    def _apply_env_vars(config: Dict) -> Dict:
        """Apply environment variable overrides."""
        prefix = "RAG_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                try:
                    # Handle nested keys (e.g., RAG_AUDIO_SAMPLE_RATE)
                    parts = config_key.split('_')
                    current = config
                    for part in parts[:-1]:
                        current = current.setdefault(part, {})
                    current[parts[-1]] = value
                except Exception as e:
                    logger.warning(f"Failed to apply env var {key}: {str(e)}")
        return config
        
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        return getattr(self.config, component, {})
        
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        config_dict = self.config.dict()
        config_dict = self._deep_merge(config_dict, updates)
        self.config = RAGConfig(**config_dict)

llm_config = {
    'api_key': 'your-openai-api-key',
    'organization': 'your-org-id',  # Optional
    'model_name': 'gpt-4',
    'temperature': 0.7,
    'max_tokens': 1000,
    'top_p': 1.0,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
    'system_prompt': 'You are a helpful assistant...'
} 