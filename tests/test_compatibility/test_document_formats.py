"""
Test Document Format Compatibility
-------------------------------

Tests for handling different document formats and ensuring proper extraction.
"""

import pytest
from pathlib import Path
import tempfile
from docx import Document
from PyPDF2 import PdfWriter
import pandas as pd
from src.document_processing.processor import DocumentProcessor
from src.document_processing.extractors import (
    PDFExtractor,
    DocxExtractor,
    ExcelExtractor,
    TextExtractor,
    CSVExtractor
)

class TestDocumentFormats:
    
    @pytest.fixture
    def document_processor(self, config_manager):
        """Fixture providing a document processor instance."""
        return DocumentProcessor(config_manager)
    
    @pytest.fixture
    def sample_documents(self, tmp_path):
        """Fixture providing sample documents in different formats."""
        docs = {}
        
        # Create PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_writer = PdfWriter()
        pdf_writer.add_blank_page(width=612, height=792)
        with open(pdf_path, 'wb') as f:
            pdf_writer.write(f)
        docs['pdf'] = pdf_path
        
        # Create DOCX
        docx_path = tmp_path / "test.docx"
        doc = Document()
        doc.add_paragraph("Test document content")
        doc.save(docx_path)
        docs['docx'] = docx_path
        
        # Create Excel
        excel_path = tmp_path / "test.xlsx"
        df = pd.DataFrame({'col1': ['data1', 'data2'], 'col2': ['data3', 'data4']})
        df.to_excel(excel_path, index=False)
        docs['excel'] = excel_path
        
        # Create CSV
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        docs['csv'] = csv_path
        
        # Create Text
        text_path = tmp_path / "test.txt"
        text_path.write_text("Test plain text content")
        docs['text'] = text_path
        
        return docs
    
    def test_pdf_extraction(self, document_processor, sample_documents):
        """Test PDF document extraction."""
        pdf_path = sample_documents['pdf']
        result = document_processor.process_document(pdf_path)
        
        assert result is not None
        assert 'text' in result
        assert 'metadata' in result
        assert result['metadata']['file_type'] == 'pdf'
        
    def test_docx_extraction(self, document_processor, sample_documents):
        """Test DOCX document extraction."""
        docx_path = sample_documents['docx']
        result = document_processor.process_document(docx_path)
        
        assert result is not None
        assert 'Test document content' in result['text']
        assert result['metadata']['file_type'] == 'docx'
        
    def test_excel_extraction(self, document_processor, sample_documents):
        """Test Excel document extraction."""
        excel_path = sample_documents['excel']
        result = document_processor.process_document(excel_path)
        
        assert result is not None
        assert 'data1' in result['text']
        assert 'data2' in result['text']
        assert result['metadata']['file_type'] == 'xlsx'
        
    def test_csv_extraction(self, document_processor, sample_documents):
        """Test CSV document extraction."""
        csv_path = sample_documents['csv']
        result = document_processor.process_document(csv_path)
        
        assert result is not None
        assert 'col1' in result['text']
        assert 'data1' in result['text']
        assert result['metadata']['file_type'] == 'csv'
        
    def test_text_extraction(self, document_processor, sample_documents):
        """Test plain text document extraction."""
        text_path = sample_documents['text']
        result = document_processor.process_document(text_path)
        
        assert result is not None
        assert 'Test plain text content' in result['text']
        assert result['metadata']['file_type'] == 'txt'
        
    def test_unsupported_format(self, document_processor, tmp_path):
        """Test handling of unsupported document formats."""
        unsupported_path = tmp_path / "test.xyz"
        unsupported_path.write_text("Test content")
        
        with pytest.raises(ValueError):
            document_processor.process_document(unsupported_path)
            
    def test_corrupted_files(self, document_processor, tmp_path):
        """Test handling of corrupted files."""
        # Create corrupted PDF
        corrupted_pdf = tmp_path / "corrupted.pdf"
        corrupted_pdf.write_text("This is not a valid PDF file")
        
        with pytest.raises(Exception):
            document_processor.process_document(corrupted_pdf)
            
    def test_large_file_handling(self, document_processor, tmp_path):
        """Test handling of large files."""
        large_text = "Test content\n" * 100000  # Create large content
        large_file = tmp_path / "large.txt"
        large_file.write_text(large_text)
        
        result = document_processor.process_document(large_file)
        assert result is not None
        assert len(result['text']) > 0
        
    def test_metadata_extraction(self, document_processor, sample_documents):
        """Test metadata extraction from different formats."""
        for doc_type, doc_path in sample_documents.items():
            result = document_processor.process_document(doc_path)
            
            assert 'metadata' in result
            assert 'file_type' in result['metadata']
            assert 'file_size' in result['metadata']
            assert 'created_at' in result['metadata']
            
    def test_batch_processing(self, document_processor, sample_documents):
        """Test batch processing of different document formats."""
        results = document_processor.batch_process(list(sample_documents.values()))
        
        assert len(results) == len(sample_documents)
        assert all('text' in result for result in results)
        assert all('metadata' in result for result in results)
        
    def test_encoding_handling(self, document_processor, tmp_path):
        """Test handling of different text encodings."""
        # UTF-8 with special characters
        utf8_text = "Hello, 世界! ñ, ü, é"
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text(utf8_text, encoding='utf-8')
        
        result = document_processor.process_document(utf8_file)
        assert utf8_text in result['text']
        
        # Test other encodings if supported
        encodings = ['latin-1', 'utf-16']
        for encoding in encodings:
            try:
                file_path = tmp_path / f"{encoding}.txt"
                file_path.write_text(utf8_text, encoding=encoding)
                result = document_processor.process_document(file_path)
                assert utf8_text in result['text']
            except UnicodeError:
                continue 