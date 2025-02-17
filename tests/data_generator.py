"""
Test data generation utilities
"""
from pathlib import Path
import numpy as np
from faker import Faker
from typing import Dict, Any, List

class TestDataGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.faker = Faker()
        
    def generate_test_suite(self) -> Dict[str, Path]:
        """Generate complete test data suite"""
        return {
            "documents": self.generate_documents(),
            "embeddings": self.generate_embeddings(),
            "mock_responses": self.generate_mock_responses()
        } 