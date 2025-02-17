"""
RAG Pipeline Tests
----------------

Comprehensive test suite for the RAG (Retrieval-Augmented Generation) pipeline.
Tests core functionality, integration, and performance.

Author: Keith Satuku
Version: 1.0.0
Created: 2025
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from src.document_processing.processor import DocumentProcessor
from ..src.document_processing.ocr_processor import OCRProcessor
from ..src.embeddings.embedding_generator import EmbeddingGenerator
from ..src.nlp.tokenizer import rag_tokenizer
from ..src.retrieval.pipeline import RetrievalPipeline

@pytest.fixture
def sample_documents():
    return [
        {
            "content": "This is a test document about physics. It discusses Newton's laws of motion.",
            "metadata": {"subject": "physics", "grade": "high-school"}
        },
        {
            "content": "Chemistry is the study of matter. Atoms are the building blocks of matter.",
            "metadata": {"subject": "chemistry", "grade": "high-school"}
        }
    ]

@pytest.fixture
def rag_pipeline(sample_documents):
    processor = DocumentProcessor()
    embedding_gen = EmbeddingGenerator()
    retriever = RetrievalPipeline()
    
    # Index sample documents
    for doc in sample_documents:
        retriever.index_document(doc)
    
    return {
        "processor": processor,
        "embedding_generator": embedding_gen,
        "retriever": retriever
    }

class TestRAGPipeline:
    def test_document_processing(self, rag_pipeline):
        """Test document processing capabilities"""
        processor = rag_pipeline["processor"]
        test_text = "Test document with some content."
        
        result = processor.process(test_text)
        assert "content" in result
        assert "metadata" in result
        
    def test_embedding_generation(self, rag_pipeline):
        """Test embedding generation"""
        generator = rag_pipeline["embedding_generator"]
        test_text = "Test embedding generation"
        
        embedding = generator.generate_embeddings(test_text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[-1] == 768  # Standard embedding size
        
    def test_retrieval(self, rag_pipeline):
        """Test document retrieval"""
        retriever = rag_pipeline["retriever"]
        query = "What are Newton's laws?"
        
        results = retriever.retrieve(query, top_k=2)
        assert len(results) > 0
        assert hasattr(results[0], 'content')
        assert hasattr(results[0], 'score')
        
    @pytest.mark.asyncio
    async def test_async_processing(self, rag_pipeline):
        """Test async processing capabilities"""
        processor = rag_pipeline["processor"]
        texts = ["Text 1", "Text 2", "Text 3"]
        
        results = await processor.process_batch_async(texts)
        assert len(results) == len(texts)

class TestOCRIntegration:
    @pytest.fixture
    def ocr_processor(self):
        return OCRProcessor()
    
    def test_image_processing(self, ocr_processor, tmp_path):
        """Test OCR processing with sample image"""
        # Create a sample image for testing
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (100, 30), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Test Text", fill='black')
        
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        result = ocr_processor.process_image(img_path)
        assert "Test" in result
        
class TestPerformance:
    def test_retrieval_speed(self, rag_pipeline):
        """Test retrieval performance"""
        retriever = rag_pipeline["retriever"]
        query = "test query"
        
        import time
        start_time = time.time()
        results = retriever.retrieve(query, top_k=5)
        end_time = time.time()
        
        assert end_time - start_time < 1.0  # Should complete within 1 second
        
    def test_batch_processing(self, rag_pipeline):
        """Test batch processing performance"""
        generator = rag_pipeline["embedding_generator"]
        texts = ["Text " + str(i) for i in range(10)]
        
        embeddings = generator.generate_embeddings(texts, batch_size=5)
        assert len(embeddings) == len(texts)

class TestErrorHandling:
    def test_invalid_input(self, rag_pipeline):
        """Test error handling for invalid inputs"""
        retriever = rag_pipeline["retriever"]
        
        with pytest.raises(ValueError):
            retriever.retrieve("")  # Empty query
            
    def test_missing_document(self, rag_pipeline):
        """Test handling of missing documents"""
        retriever = rag_pipeline["retriever"]
        
        result = retriever.get_document("non_existent_id")
        assert result is None 