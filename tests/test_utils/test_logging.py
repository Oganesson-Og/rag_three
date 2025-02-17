"""
Test Logging Configuration
------------------------

Tests for logging setup and functionality.
"""

import pytest
import logging
from pathlib import Path
from src.utils.logging import setup_logging, LogConfig

class TestLogging:
    
    def test_logging_setup(self, tmp_path):
        """Test basic logging setup."""
        log_file = tmp_path / "test.log"
        config = LogConfig(
            log_file=log_file,
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        logger = setup_logging(config)
        
        # Test log file creation
        assert log_file.exists()
        
        # Test logging levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        # Check log content
        log_content = log_file.read_text()
        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content
        
    def test_logging_rotation(self, tmp_path):
        """Test log file rotation."""
        config = LogConfig(
            log_file=tmp_path / "rotating.log",
            max_bytes=1000,
            backup_count=3
        )
        
        logger = setup_logging(config)
        
        # Generate enough logs to trigger rotation
        long_message = "X" * 100
        for _ in range(20):
            logger.info(long_message)
            
        # Check rotation files exist
        assert (tmp_path / "rotating.log").exists()
        assert (tmp_path / "rotating.log.1").exists()
        
    def test_logging_formats(self, tmp_path):
        """Test different logging formats."""
        log_file = tmp_path / "format_test.log"
        custom_format = "%(levelname)s - %(message)s"
        
        config = LogConfig(
            log_file=log_file,
            format=custom_format
        )
        
        logger = setup_logging(config)
        logger.info("Test message")
        
        log_content = log_file.read_text()
        assert "INFO - Test message" in log_content
        
    def test_error_handling(self, tmp_path):
        """Test logging error handling."""
        # Test with invalid log file path
        with pytest.raises(Exception):
            config = LogConfig(log_file="/invalid/path/test.log")
            setup_logging(config)
            
        # Test with invalid log level
        with pytest.raises(ValueError):
            config = LogConfig(log_file=tmp_path / "test.log", level="INVALID")
            setup_logging(config) 