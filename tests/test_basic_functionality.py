import pytest
from pathlib import Path
from src.embeddings.embedding_generator import EmbeddingGenerator
from tests.test_utils.helpers import wait_for_processing

def test_document_processing(test_config, sample_documents):
    """Test basic document processing functionality"""
    pipeline = EmbeddingGenerator()
    
    # Process text document
    text_file = sample_documents / "sample.txt"
    assert text_file.exists(), "Sample text file not found"
    
    # Add wait_for_processing to ensure document is fully processed
    result = pipeline.process_document(text_file)
    assert result, "Document processing failed"
    
    # Wait for processing to complete
    def check_processing():
        return pipeline.is_document_processed(text_file)
    
    assert wait_for_processing(check_processing), "Document processing timeout"
    
    # Test basic query with specific assertions
    query = "What technical terms are mentioned?"
    response = pipeline.query(query)
    
    # More specific assertions about the response
    assert response, "Query returned empty response"
    assert isinstance(response, str), "Response should be a string"
    assert "AI" in response, "Expected term 'AI' not found in response"
    assert "Machine Learning" in response, "Expected term 'Machine Learning' not found in response"

def test_equation_processing(test_config, sample_equations):
    """Test equation processing functionality"""
    pipeline = EmbeddingGenerator()
    
    # Process equations file
    equations_file = sample_equations / "math_equations.txt"
    assert equations_file.exists(), "Equations file not found"
    
    # Extract and validate equations
    with equations_file.open() as f:
        content = f.read()
    
    # Test equation extraction
    equations = pipeline.diagram_analyzer._extract_equations(content)
    assert len(equations) > 0, "No equations extracted"
    
    # Test each equation with more specific assertions
    expected_equation_parts = ['x', '=', '±', '√']  # Basic parts we expect to find
    
    for eq in equations:
        # Validate equation format
        assert pipeline.diagram_analyzer._validate_equation(eq), f"Invalid equation: {eq}"
        
        # Check LaTeX conversion
        latex = pipeline.diagram_analyzer._convert_to_latex(eq)
        assert latex, f"LaTeX conversion failed for equation: {eq}"
        
        # Verify equation contains expected mathematical elements
        assert any(part in eq for part in expected_equation_parts), \
            f"Equation missing expected mathematical elements: {eq}"

def test_error_handling(test_config):
    """Test error handling scenarios"""
    pipeline = EmbeddingGenerator()
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        pipeline.process_document(Path("nonexistent.txt"))
    
    # Test with empty query
    response = pipeline.query("")
    assert "error" in response.lower(), "Expected error message for empty query"
    
    # Test with invalid file type
    with pytest.raises(ValueError):
        pipeline.process_document(Path("invalid.xyz"))
    
    # Test with corrupted document
    corrupted_file = test_config.TEST_DATA_DIR / "corrupted.txt"
    with open(corrupted_file, 'wb') as f:
        f.write(b'\x00\x00\x00')  # Write some invalid bytes
    
    with pytest.raises(Exception):
        pipeline.process_document(corrupted_file)

def test_batch_processing(test_config, sample_documents):
    """Test processing multiple documents"""
    pipeline = EmbeddingGenerator()
    
    # Create multiple test documents
    docs = [
        sample_documents / "doc1.txt",
        sample_documents / "doc2.txt",
        sample_documents / "doc3.txt"
    ]
    
    # Write some content to test files
    for i, doc in enumerate(docs):
        doc.write_text(f"This is test document {i+1} with some content.")
    
    # Process all documents
    results = []
    for doc in docs:
        result = pipeline.process_document(doc)
        results.append(result)
    
    # Verify all documents were processed
    assert all(results), "Batch processing failed"
    
    # Test query across all documents
    query = "test document"
    response = pipeline.query(query)
    assert response, "Query across multiple documents failed"
    
    # Cleanup test files
    for doc in docs:
        doc.unlink() 