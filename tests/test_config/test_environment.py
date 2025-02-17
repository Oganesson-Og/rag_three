"""
Test Environment Configuration
---------------------------

Tests for environment setup and management.
"""

import pytest
import os
from pathlib import Path
from src.config.environment import EnvironmentManager

class TestEnvironment:
    
    @pytest.fixture
    def env_manager(self):
        """Fixture providing an environment manager."""
        return EnvironmentManager()
    
    def test_environment_setup(self, env_manager, tmp_path):
        """Test environment setup."""
        env_manager.setup_environment(
            root_dir=tmp_path,
            create_dirs=True
        )
        
        # Check required directories
        assert (tmp_path / "vectors").exists()
        assert (tmp_path / "documents").exists()
        assert (tmp_path / "cache").exists()
        assert (tmp_path / "logs").exists()
        
    def test_environment_variables(self, env_manager, monkeypatch):
        """Test environment variable handling."""
        # Set test environment variables
        monkeypatch.setenv('RAG_HOME', '/test/path')
        monkeypatch.setenv('RAG_CACHE_DIR', '/test/cache')
        
        env_vars = env_manager.get_environment_variables()
        
        assert env_vars['RAG_HOME'] == '/test/path'
        assert env_vars['RAG_CACHE_DIR'] == '/test/cache'
        
    def test_environment_validation(self, env_manager):
        """Test environment validation."""
        # Test with missing required directory
        with pytest.raises(ValueError):
            env_manager.validate_environment(
                required_dirs=['nonexistent_dir']
            )
            
        # Test with missing required environment variable
        with pytest.raises(ValueError):
            env_manager.validate_environment(
                required_vars=['NONEXISTENT_VAR']
            )
            
    def test_environment_cleanup(self, env_manager, tmp_path):
        """Test environment cleanup."""
        # Setup test environment
        test_dirs = {
            'cache': tmp_path / "cache",
            'temp': tmp_path / "temp"
        }
        
        for dir_path in test_dirs.values():
            dir_path.mkdir()
            (dir_path / "test_file.txt").touch()
            
        # Cleanup
        env_manager.cleanup_environment(
            dirs_to_clean=list(test_dirs.values())
        )
        
        # Check directories are empty
        assert not any((test_dirs['cache']).iterdir())
        assert not any((test_dirs['temp']).iterdir())
        
    def test_environment_persistence(self, env_manager, tmp_path):
        """Test environment state persistence."""
        state_file = tmp_path / "env_state.json"
        
        # Save state
        env_manager.save_environment_state(
            state_file,
            {'last_cleanup': '2024-01-01'}
        )
        
        # Load state
        loaded_state = env_manager.load_environment_state(state_file)
        assert loaded_state['last_cleanup'] == '2024-01-01'
        
    def test_environment_locks(self, env_manager, tmp_path):
        """Test environment resource locking."""
        lock_file = tmp_path / "resource.lock"
        
        # Acquire lock
        with env_manager.environment_lock(lock_file):
            assert lock_file.exists()
            
            # Try to acquire same lock (should fail)
            with pytest.raises(RuntimeError):
                with env_manager.environment_lock(lock_file):
                    pass
                    
        # Lock should be released
        assert not lock_file.exists()
        
    def test_environment_resources(self, env_manager, tmp_path):
        """Test environment resource management."""
        # Setup resource limits
        resource_limits = {
            'max_cache_size': 1024 * 1024,  # 1MB
            'max_open_files': 100
        }
        
        env_manager.set_resource_limits(resource_limits)
        
        # Test resource usage
        usage = env_manager.get_resource_usage()
        assert 'cache_size' in usage
        assert 'open_files' in usage
        
        # Test resource cleanup when exceeding limits
        env_manager.cleanup_resources() 