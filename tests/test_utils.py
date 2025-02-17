"""
Test utilities and helper functions
"""
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path
import json
import pytest
from datetime import datetime

class TestUtils:
    @staticmethod
    def generate_test_embedding(dim: int = 768) -> np.ndarray:
        """Generate test embedding vector"""
        return np.random.randn(dim)
    
    @staticmethod
    def create_test_document(
        content: str,
        doc_type: str,
        output_dir: Path
    ) -> Path:
        """Create test document file"""
        file_ext = {
            "pdf": ".pdf",
            "text": ".txt",
            "math": ".tex",
            "image": ".png"
        }[doc_type]
        
        output_path = output_dir / f"test_doc_{datetime.now().timestamp()}{file_ext}"
        
        with open(output_path, "w") as f:
            f.write(content)
            
        return output_path
    
    @staticmethod
    def compare_embeddings(
        emb1: np.ndarray,
        emb2: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """Compare two embeddings with tolerance"""
        return np.allclose(emb1, emb2, rtol=tolerance)

class MockResponses:
    """Mock response generator for testing"""
    def __init__(self, mock_dir: Path):
        self.mock_dir = mock_dir
        
    def get_mock_response(
        self,
        response_type: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Get mock response data"""
        response_file = self.mock_dir / f"{response_type}.json"
        
        if not response_file.exists():
            raise ValueError(f"Mock response not found: {response_type}")
            
        with open(response_file, 'r') as f:
            response = json.load(f)
            
        if params:
            # Modify response based on params
            pass
            
        return response 