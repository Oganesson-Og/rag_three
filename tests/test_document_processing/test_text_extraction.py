"""
Test Text Extraction Module
-------------------------

Tests for text extraction functionality from various document formats.
"""

import pytest
from pathlib import Path
from src.document_processing.processor import DocumentProcessor
from src.document_processing.ocr_processor import OCRProcessor

class TestTextExtraction:
    
    def test_text_extraction_from_txt(self, test_data_dir, preprocessor):
        """Test extracting text from plain text files."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        txt_path = test_data_dir / "documents" / "sample.txt"
        
        extracted_text = doc_processor.extract_text(txt_path)
        assert isinstance(extracted_text, str)
        assert len(extracted_text) > 0
        assert "Pythagorean theorem" in extracted_text
        
    def test_text_extraction_from_pdf(self, test_data_dir, preprocessor, ocr_processor):
        """Test extracting text from PDF files."""
        doc_processor = DocumentProcessor(
            preprocessor=preprocessor,
            ocr_processor=ocr_processor
        )
        pdf_path = test_data_dir / "documents" / "sample.pdf"
        
        extracted_text = doc_processor.extract_text(pdf_path)
        assert isinstance(extracted_text, str)
        assert len(extracted_text) > 0
        
    def test_text_extraction_from_html(self, test_data_dir, preprocessor):
        """Test extracting text from HTML files."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        html_path = test_data_dir / "documents" / "sample.html"
        
        extracted_text = doc_processor.extract_text(html_path)
        assert isinstance(extracted_text, str)
        assert len(extracted_text) > 0
        
    def test_text_extraction_with_equations(self, test_data_dir, preprocessor):
        """Test extracting text with mathematical equations."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        math_doc_path = test_data_dir / "documents" / "sample_equations.md"
        
        extracted_text = doc_processor.extract_text(math_doc_path)
        assert isinstance(extracted_text, str)
        assert len(extracted_text) > 0
        assert "=" in extracted_text  # Basic check for equation presence
        
    def test_text_extraction_error_handling(self, test_data_dir, preprocessor):
        """Test error handling during text extraction."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            doc_processor.extract_text(Path("nonexistent.txt"))
            
        # Test with unsupported file type
        with pytest.raises(ValueError):
            doc_processor.extract_text(Path("invalid.xyz"))
            
    def test_text_extraction_with_preprocessing(self, test_data_dir, preprocessor):
        """Test text extraction with preprocessing options."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        txt_path = test_data_dir / "documents" / "sample.txt"
        
        # Test with different preprocessing configurations
        extracted_text = doc_processor.extract_text(
            txt_path,
            normalize=True,
            remove_noise=True
        )
        assert isinstance(extracted_text, str)
        assert len(extracted_text) > 0
        
    def test_batch_text_extraction(self, test_data_dir, preprocessor):
        """Test batch text extraction from multiple files."""
        doc_processor = DocumentProcessor(preprocessor=preprocessor)
        
        # Process multiple files
        files = [
            test_data_dir / "documents" / "sample.txt",
            test_data_dir / "documents" / "sample_equations.md"
        ]
        
        results = doc_processor.batch_extract_text(files)
        assert len(results) == len(files)
        assert all(isinstance(text, str) for text in results.values())
        assert all(len(text) > 0 for text in results.values())
