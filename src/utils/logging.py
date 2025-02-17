"""
Logging Configuration Module
--------------------------

Configurable logging setup for the RAG system with rotation support.

Key Features:
- Rotating file handlers
- Configurable log levels
- Custom formatting
- Size-based rotation
- Backup management
- Thread safety
- Structured logging

Technical Details:
- RotatingFileHandler implementation
- Log level management
- Format customization
- File size monitoring
- Backup file handling
- Thread-safe operations

Dependencies:
- logging
- dataclasses
- pathlib
- typing-extensions>=4.7.0

Example Usage:
    # Create log config
    config = LogConfig(
        log_file=Path("logs/app.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        max_bytes=1024*1024,  # 1MB
        backup_count=3
    )
    
    # Setup logging
    logger = setup_logging(config)
    
    # Use logger
    logger.info("Application started")
    logger.error("Error occurred", exc_info=True)

Performance Considerations:
- Efficient file rotation
- Optimized I/O operations
- Memory-efficient logging
- Thread synchronization
- Backup file management

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

@dataclass
class LogConfig:
    log_file: Path
    level: int = logging.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_bytes: int = 1024 * 1024  # 1MB
    backup_count: int = 3

def setup_logging(config: LogConfig) -> logging.Logger:
    """Setup logging with the given configuration."""
    logger = logging.getLogger('rag')
    logger.setLevel(config.level)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count
    )
    
    # Create formatters
    formatter = logging.Formatter(config.format)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    
    return logger 