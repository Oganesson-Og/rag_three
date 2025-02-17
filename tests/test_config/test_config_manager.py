"""
Test Configuration Manager Module
-----------------------------

Tests for configuration management, including loading, validation, and updates.
"""

import pytest
import yaml
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.config.models import (
    RagConfig,
    EmbeddingConfig,
    SearchConfig,
    ProcessingConfig
)

class TestConfigManager:
    
    @pytest.fixture
    def config_dir(self, tmp_path):
        """Fixture providing a temporary config directory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return config_dir
        
    @pytest.fixture
    def sample_config(self, config_dir):
        """Fixture providing a sample configuration file."""
        config = {
            'embedding': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384,
                'batch_size': 32
            },
            'search': {
                'top_k': 5,
                'score_threshold': 0.7,
                'use_hybrid': True
            },
            'processing': {
                'chunk_size': 500,
                'chunk_overlap': 50,
                'clean_text': True
            },
            'cache': {
                'enabled': True,
                'max_size_mb': 1000,
                'ttl_seconds': 3600
            }
        }
        
        config_path = config_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return config_path
        
    @pytest.fixture
    def config_manager(self, sample_config):
        """Fixture providing a config manager instance."""
        return ConfigManager(config_path=sample_config)
        
    def test_config_loading(self, config_manager, sample_config):
        """Test configuration loading from file."""
        config = config_manager.load_config()
        
        assert isinstance(config, RagConfig)
        assert config.embedding.model_name == 'sentence-transformers/all-MiniLM-L6-v2'
        assert config.search.top_k == 5
        assert config.processing.chunk_size == 500
        
    def test_config_validation(self, config_dir):
        """Test configuration validation."""
        # Test invalid embedding dimension
        invalid_config = {
            'embedding': {
                'model_name': 'test-model',
                'dimension': -1  # Invalid dimension
            }
        }
        
        config_path = config_dir / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
            
        with pytest.raises(ValueError):
            ConfigManager(config_path=config_path)
            
    def test_config_update(self, config_manager):
        """Test configuration updates."""
        updates = {
            'embedding': {
                'batch_size': 64
            },
            'search': {
                'top_k': 10
            }
        }
        
        config_manager.update_config(updates)
        config = config_manager.get_config()
        
        assert config.embedding.batch_size == 64
        assert config.search.top_k == 10
        
    def test_config_persistence(self, config_manager, sample_config):
        """Test configuration persistence."""
        # Update config
        updates = {'search': {'top_k': 15}}
        config_manager.update_config(updates)
        
        # Create new manager instance
        new_manager = ConfigManager(config_path=sample_config)
        config = new_manager.get_config()
        
        assert config.search.top_k == 15
        
    def test_environment_override(self, config_manager, monkeypatch):
        """Test environment variable configuration override."""
        # Set environment variable
        monkeypatch.setenv('RAG_SEARCH_TOP_K', '20')
        
        config = config_manager.load_config()
        assert config.search.top_k == 20
        
    def test_config_defaults(self, config_dir):
        """Test default configuration values."""
        minimal_config = {
            'embedding': {
                'model_name': 'test-model'
            }
        }
        
        config_path = config_dir / "minimal_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f)
            
        manager = ConfigManager(config_path=config_path)
        config = manager.get_config()
        
        # Check defaults
        assert config.search.top_k == 5  # Default value
        assert config.processing.chunk_size == 500  # Default value
        
    def test_config_schema_validation(self, config_dir):
        """Test configuration schema validation."""
        invalid_config = {
            'embedding': {
                'model_name': 123  # Should be string
            }
        }
        
        config_path = config_dir / "schema_invalid.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
            
        with pytest.raises(TypeError):
            ConfigManager(config_path=config_path)
            
    def test_config_sections(self, config_manager):
        """Test access to specific configuration sections."""
        embedding_config = config_manager.get_embedding_config()
        search_config = config_manager.get_search_config()
        processing_config = config_manager.get_processing_config()
        
        assert isinstance(embedding_config, EmbeddingConfig)
        assert isinstance(search_config, SearchConfig)
        assert isinstance(processing_config, ProcessingConfig)
        
    def test_config_export(self, config_manager, tmp_path):
        """Test configuration export functionality."""
        export_path = tmp_path / "exported_config.yaml"
        config_manager.export_config(export_path)
        
        # Verify exported file
        assert export_path.exists()
        with open(export_path) as f:
            exported_config = yaml.safe_load(f)
            
        assert exported_config['embedding']['model_name'] == 'sentence-transformers/all-MiniLM-L6-v2'
        
    def test_config_reload(self, config_manager, sample_config):
        """Test configuration reload functionality."""
        # Modify config file
        config = yaml.safe_load(open(sample_config))
        config['search']['top_k'] = 25
        with open(sample_config, 'w') as f:
            yaml.dump(config, f)
            
        # Reload config
        config_manager.reload_config()
        assert config_manager.get_config().search.top_k == 25
        
    def test_invalid_paths(self):
        """Test handling of invalid configuration paths."""
        with pytest.raises(FileNotFoundError):
            ConfigManager(config_path="nonexistent.yaml")
            
    def test_config_inheritance(self, config_dir):
        """Test configuration inheritance functionality."""
        base_config = {
            'embedding': {
                'model_name': 'base-model',
                'dimension': 384
            }
        }
        
        child_config = {
            'embedding': {
                'model_name': 'child-model'
            },
            '_inherit': 'base_config.yaml'
        }
        
        # Write configs
        base_path = config_dir / "base_config.yaml"
        child_path = config_dir / "child_config.yaml"
        
        with open(base_path, 'w') as f:
            yaml.dump(base_config, f)
        with open(child_path, 'w') as f:
            yaml.dump(child_config, f)
            
        manager = ConfigManager(config_path=child_path)
        config = manager.get_config()
        
        assert config.embedding.model_name == 'child-model'
        assert config.embedding.dimension == 384
        
    def test_config_validation_rules(self, config_manager):
        """Test custom configuration validation rules."""
        # Test invalid chunk size and overlap combination
        with pytest.raises(ValueError):
            config_manager.update_config({
                'processing': {
                    'chunk_size': 100,
                    'chunk_overlap': 150  # Overlap larger than chunk size
                }
            }) 