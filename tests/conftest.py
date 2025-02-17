"""
Test Configuration Module
-----------------------

Provides fixtures and configuration for testing the RAG system components.

Key Features:
- Common test fixtures
- Mock data generation
- Test environment setup
- Resource management
"""

import pytest
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from unittest.mock import MagicMock
from dotenv import load_dotenv
import whisper
from src.utils.rag_tokenizer import RagTokenizer
import shutil
import logging
from typing import Generator, Dict, Any
import sys

from src.audio.processor import AudioProcessor
from src.document_processing.processor import DocumentProcessor
from src.document_processing.ocr_processor import OCRProcessor
from src.document_processing.preprocessor import DocumentPreprocessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.config.config_manager import ConfigManager
from src.core.rag_orchestrator import RAGOrchestrator
from src.error_handling.error_manager import ErrorManager
from src.distributed.distributed_processor import DistributedProcessor

# Load environment variables
load_dotenv()

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configuration
class TestConfig:
    HOST_ADDRESS = os.getenv('HOST_ADDRESS', 'http://127.0.0.1:9380')
    API_KEY = os.getenv('API_KEY', 'default-test-key')
    TEST_DATA_DIR = Path(__file__).parent / 'test_data'
    TIMEOUT = 30  # seconds

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration"""
    return {
        "test_data_dir": Path("tests/test_data"),
        "temp_dir": Path("tests/temp"),
        "vector_dimension": 768,
        "batch_size": 32,
        "max_length": 512,
        "timeout": 30,
        "mock_responses_dir": Path("tests/mock_responses"),
        "test_documents": {
            "pdf": "sample.pdf",
            "text": "sample.txt",
            "math": "math_content.tex",
            "image": "diagram.png"
        }
    }

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(test_config: Dict[str, Any]) -> Generator:
    """Setup test environment and cleanup after tests"""
    # Create necessary directories
    test_config["temp_dir"].mkdir(parents=True, exist_ok=True)
    test_config["test_data_dir"].mkdir(parents=True, exist_ok=True)
    test_config["mock_responses_dir"].mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=test_config["temp_dir"] / "test.log"
    )
    
    # Generate test data if needed
    _generate_test_data(test_config)
    
    yield
    
    # Cleanup
    shutil.rmtree(test_config["temp_dir"])

@pytest.fixture(scope="function")
def rag_orchestrator(test_config: Dict[str, Any]) -> RAGOrchestrator:
    """Create RAG orchestrator instance for testing"""
    return RAGOrchestrator(
        config_path=test_config["temp_dir"] / "test_config.json",
        base_path=test_config["temp_dir"]
    )

def _generate_test_data(config: Dict[str, Any]) -> None:
    """Generate test data files"""
    # Implementation for generating test data
    pass

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Fixture providing a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture(scope="session")
def sample_documents(test_data_dir):
    return {
        'txt': os.path.join(test_data_dir, 'documents/sample.txt'),
        'pdf': os.path.join(test_data_dir, 'documents/sample.pdf'),
        'html': os.path.join(test_data_dir, 'documents/sample.html'),
        'md': os.path.join(test_data_dir, 'documents/sample_equations.md')
    }

@pytest.fixture(scope="session")
def sample_equations(test_data_dir):
    """Create and provide sample equations"""
    equations_dir = test_data_dir / "equations"
    equations_dir.mkdir(exist_ok=True)
    
    equations_file = equations_dir / "math_equations.txt"
    equations_file.write_text("""
    1. x = (-b ± √(b² - 4ac)) / (2a)
    2. E = mc²
    3. F = G(m₁m₂)/r²
    4. ∫(x²)dx = x³/3 + C
    """)
    
    return equations_dir

@pytest.fixture(scope="session")
def auth_token():
    """Fixture to handle authentication"""
    # Implement your authentication logic here
    return "test-auth-token"

def load_test_document(filepath):
    """Helper to load test documents"""
    with open(filepath, 'r') as f:
        return f.read()

def compare_embeddings(emb1, emb2, tolerance=1e-6):
    """Helper to compare embedding vectors"""
    return np.allclose(emb1, emb2, atol=tolerance)

@pytest.fixture
def sample_text():
    """Fixture providing sample educational text."""
    return """
    The Pythagorean theorem states that in a right triangle, 
    the square of the length of the hypotenuse is equal to 
    the sum of squares of the other two sides. This can be 
    written as a² + b² = c², where c is the hypotenuse.
    """

@pytest.fixture
def sample_embeddings():
    """Fixture providing sample document embeddings."""
    return np.random.rand(5, 384)  # Common embedding dimension

@pytest.fixture
def mock_audio_processor():
    """Fixture providing a mocked audio processor."""
    processor = MagicMock(spec=AudioProcessor)
    processor.process_audio.return_value = {
        "text": "Sample transcribed text",
        "confidence": 0.95,
        "speakers": ["Speaker 1"]
    }
    return processor

@pytest.fixture
def mock_document_processor():
    """Fixture providing a mocked document processor."""
    processor = MagicMock(spec=DocumentProcessor)
    return processor

@pytest.fixture
def mock_embedding_generator():
    """Fixture providing a mocked embedding generator."""
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.generate_embedding.return_value = np.random.rand(384)
    return generator

@pytest.fixture
def temp_config_file():
    """Fixture providing a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
embedding:
    cache_dir: .cache/embeddings
    batch_size: 32
    model_name: sentence-transformers/all-MiniLM-L6-v2
audio:
    model: whisper-large-v3
    device: cpu
        """)
        path = f.name
    yield path
    os.unlink(path)

@pytest.fixture
def config_manager(temp_config_file):
    """Fixture providing a configuration manager instance."""
    return ConfigManager(config_path=temp_config_file)

@pytest.fixture
def device():
    """Fixture providing compute device configuration."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture
def preprocessor():
    """Fixture providing a document preprocessor instance."""
    return DocumentPreprocessor(
        normalize_unicode=True,
        fix_encoding=True,
        detect_language=True,
        remove_noise=True
    )

@pytest.fixture
def ocr_processor():
    """Fixture providing an OCR processor instance."""
    return OCRProcessor()

@pytest.fixture
def sample_education_metadata():
    """Fixture providing sample educational metadata."""
    return {
        "subject": "mathematics",
        "grade_level": "high_school",
        "topics": ["geometry", "trigonometry"],
        "difficulty": "intermediate",
        "prerequisites": ["basic_algebra", "basic_geometry"],
        "learning_objectives": ["understand_pythagorean_theorem", "apply_triangle_properties"]
    }

@pytest.fixture(scope="session")
def config_manager():
    """Fixture providing a config manager instance for all tests."""
    config_path = Path("config/test_config.yaml")
    return ConfigManager(config_path)

@pytest.fixture(scope="session")
def whisper_model():
    """Fixture providing a Whisper model instance."""
    return whisper.load_model("base")

@pytest.fixture(scope="session")
def audio_processor(whisper_model):
    """Fixture providing an audio processor instance."""
    return AudioProcessor(model=whisper_model)

@pytest.fixture(scope="session")
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        {
            "text": "This is a test document about machine learning.",
            "metadata": {
                "source": "test",
                "category": "ML"
            }
        },
        {
            "text": "Python is a popular programming language.",
            "metadata": {
                "source": "test",
                "category": "programming"
            }
        }
    ]

@pytest.fixture(scope="session")
def sample_audio_file(test_data_dir):
    """Fixture providing a sample audio file for testing."""
    audio_path = test_data_dir / "test_audio.wav"
    # Create a simple test audio file if needed
    if not audio_path.exists():
        import numpy as np
        import soundfile as sf
        
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        sf.write(audio_path, audio_data, sample_rate)
    
    return audio_path

@pytest.fixture(autouse=True)
def setup_logging():
    """Fixture to set up logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def tokenizer():
    """Create tokenizer with test dictionary."""
    test_dict = os.path.join(
        os.path.dirname(__file__),
        'fixtures',
        'huqie.txt'
    )
    return RagTokenizer(dict_path=test_dict)

@pytest.fixture
def mock_diagram_analyzer():
    """Provide a mock diagram analyzer for testing."""
    analyzer = MagicMock()
    analyzer.process.return_value = {
        "objects": [],
        "relationships": [],
        "confidence": 0.9
    }
    return analyzer 