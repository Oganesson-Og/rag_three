"""
Test Metadata Extraction Module
-----------------------------

Tests for document metadata extraction and processing functionality.
"""

import pytest
from datetime import datetime
from pathlib import Path
from src.document_processing.processor import DocumentProcessor
from src.document_processing.preprocessor import DocumentPreprocessor

class TestMetadataExtraction:
    
    @pytest.fixture
    def sample_pdf_metadata(self):
        """Fixture providing sample PDF metadata."""
        return {
            'Title': 'Introduction to Machine Learning',
            'Author': 'John Doe',
            'Subject': 'Computer Science',
            'Keywords': 'AI, ML, Neural Networks',
            'CreationDate': 'D:20230615123456',
            'Producer': 'Adobe PDF Library 15.0',
            'Creator': 'Microsoft® Word 2019'
        }
    
    def test_basic_metadata_extraction(self, test_data_dir, preprocessor):
        """Test basic metadata extraction from documents."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        pdf_path = test_data_dir / "documents" / "sample.pdf"
        
        metadata = doc_processor.extract_metadata(pdf_path)
        
        assert isinstance(metadata, dict)
        assert 'file_type' in metadata
        assert 'file_size' in metadata
        assert 'last_modified' in metadata
        
    def test_pdf_metadata_extraction(self, test_data_dir, preprocessor):
        """Test PDF-specific metadata extraction."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        pdf_path = test_data_dir / "documents" / "sample.pdf"
        
        metadata = doc_processor.extract_metadata(pdf_path)
        
        # Check PDF-specific fields
        assert 'title' in metadata
        assert 'author' in metadata
        assert 'creation_date' in metadata
        assert isinstance(metadata['creation_date'], datetime)
        
    def test_educational_metadata_extraction(self, test_data_dir, preprocessor):
        """Test extraction of educational metadata."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        doc_path = test_data_dir / "documents" / "sample_with_metadata.pdf"
        
        metadata = doc_processor.extract_metadata(
            doc_path,
            extract_educational_metadata=True
        )
        
        # Check educational metadata fields
        assert 'subject_area' in metadata
        assert 'grade_level' in metadata
        assert 'learning_objectives' in metadata
        assert isinstance(metadata['learning_objectives'], list)
        
    def test_metadata_normalization(self, test_data_dir, preprocessor):
        """Test metadata normalization and standardization."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        doc_path = test_data_dir / "documents" / "sample.pdf"
        
        metadata = doc_processor.extract_metadata(
            doc_path,
            normalize=True
        )
        
        # Check normalized fields
        assert all(isinstance(key, str) and key.islower() for key in metadata.keys())
        assert all(value is not None for value in metadata.values())
        
    def test_custom_metadata_extraction(self, test_data_dir, preprocessor):
        """Test custom metadata extraction rules."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        custom_rules = {
            'difficulty_level': r'Difficulty:\s*(Beginner|Intermediate|Advanced)',
            'prerequisites': r'Prerequisites:\s*(.+?)(?=\n|$)',
            'estimated_time': r'Estimated Time:\s*(\d+)\s*hours?'
        }
        
        metadata = doc_processor.extract_metadata(
            test_data_dir / "documents" / "sample.pdf",
            custom_rules=custom_rules
        )
        
        # Check custom metadata fields
        assert any(key in metadata for key in custom_rules.keys())
        
    def test_metadata_extraction_error_handling(self, preprocessor):
        """Test error handling in metadata extraction."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            doc_processor.extract_metadata(Path("nonexistent.pdf"))
            
        # Test with unsupported file type
        with pytest.raises(ValueError):
            doc_processor.extract_metadata(Path("invalid.xyz"))
            
    def test_batch_metadata_extraction(self, test_data_dir, preprocessor):
        """Test batch metadata extraction from multiple files."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        files = [
            test_data_dir / "documents" / "sample.pdf",
            test_data_dir / "documents" / "sample_with_metadata.pdf"
        ]
        
        batch_metadata = doc_processor.batch_extract_metadata(files)
        
        assert isinstance(batch_metadata, dict)
        assert len(batch_metadata) == len(files)
        assert all(isinstance(meta, dict) for meta in batch_metadata.values())
        
    def test_metadata_update_and_merge(self, test_data_dir, preprocessor):
        """Test updating and merging metadata."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        doc_path = test_data_dir / "documents" / "sample.pdf"
        
        # Extract initial metadata
        initial_metadata = doc_processor.extract_metadata(doc_path)
        
        # Update with new metadata
        new_metadata = {
            'tags': ['education', 'mathematics'],
            'difficulty': 'intermediate'
        }
        
        updated_metadata = doc_processor.update_metadata(
            doc_path,
            new_metadata,
            merge=True
        )
        
        assert all(key in updated_metadata for key in new_metadata)
        assert all(key in updated_metadata for key in initial_metadata)
