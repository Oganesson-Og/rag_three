"""
Configuration Management Module
----------------------------

Central configuration management system providing dynamic configuration handling,
validation, and environment-specific settings for the RAG pipeline.

Key Features:
- Configuration loading and saving
- Environment-specific settings management
- Dynamic configuration updates
- Validation rules and schema checking
- Default configuration handling
- Configuration inheritance
- Secret management

Technical Details:
- YAML-based configuration files
- Environment variable integration
- Schema validation using Pydantic
- Hierarchical configuration structure
- Configuration versioning support

Dependencies:
- pyyaml>=6.0.1
- pydantic>=2.5.0
- python-dotenv>=1.0.0
- typing-extensions>=4.8.0
- jsonschema>=4.20.0

Example Usage:
    # Basic initialization
    config = ConfigManager()
    
    # Custom config path
    config = ConfigManager(config_path='custom_config.yaml')
    
    # Get subject configuration
    subject_config = config.get_subject_config(
        subject='physics',
        level='a-level'
    )
    
    # Update configuration
    config.update_config({
        'embedding': {'batch_size': 64}
    })

Configuration Structure:
- embedding: Model and processing settings
- retrieval: Search and ranking parameters
- database: Connection and storage settings
- api: Service configuration
- logging: Log management settings

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""
from typing import Dict, Union, Optional, Any, List
from pathlib import Path
import json
import logging
from datetime import datetime
from .domain_config import EDUCATION_DOMAINS
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ConfigSection:
    """Configuration section with type hints."""
    embedding: Dict[str, Union[str, int, float]]
    processing: Dict[str, Union[str, int, float]]
    storage: Dict[str, Union[str, int, float]]
    search: Dict[str, Union[str, int, float]]
    cache: Optional[Dict[str, Union[str, int, float]]] = None

ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]
ConfigDict = Dict[str, ConfigValue]

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize config manager."""
        try:
            # Load config from file
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Validate config
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise
            
    def _validate_config(self):
        """Validate loaded configuration."""
        required_sections = ['extraction', 'processing', 'storage']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get configuration for specific component."""
        component_config = self.config.get('components', {}).get(component, {})
        # Merge with any global settings that might apply
        return {
            **self.config.get('global', {}),
            **component_config
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self.config.get('cache', {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self.config.get('vector_store', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path) as f:
                self.config.update(json.load(f))
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def _save_config(self) -> None:
        """Save configuration to file."""
        if not self.config_path:
            return
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            raise

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        return self.config.get(section, {})
        
    def update_section(
        self,
        section: str,
        values: Dict[str, Any]
    ) -> None:
        """Update configuration section."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)
        self._save_config()
        
    def get_subject_config(self, subject: str, level: str) -> Dict:
        """Get subject-specific configuration."""
        for domain, config in EDUCATION_DOMAINS.items():
            if subject.lower() in config['subjects']:
                return {
                    "domain": domain,
                    "model": config['models'].get(subject.lower(), config['models']['default']),
                    "preprocessing": config['preprocessing'],
                    "keywords": config['keywords'].get(subject.lower(), [])
                }
        raise ValueError(f"Invalid subject: {subject}")
    
    def update_config(self, updates: Dict) -> None:
        """Update configuration with new values."""
        def deep_update(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, (dict, list)):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = {**self.config, **updates}
        self._save_config() 