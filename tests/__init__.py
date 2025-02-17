"""
Root test package for the RAG system.
Provides access to common test utilities and configurations.
"""

from .conftest import (
    test_data_dir,
    sample_documents,
    mock_audio_processor,
    mock_document_processor,
    mock_embedding_generator,
    config_manager,
    preprocessor,
    ocr_processor
)

__version__ = '0.1.0'
__author__ = 'Keith Oganesson'

__all__ = [
    'test_data_dir',
    'sample_documents',
    'mock_audio_processor',
    'mock_document_processor',
    'mock_embedding_generator',
    'config_manager',
    'preprocessor',
    'ocr_processor'
]