"""
Test Helper Functions
-------------------

Common utility functions used across tests.
"""

import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

def wait_for_processing(
    check_status_fn: callable,
    timeout: int = 60,
    interval: int = 1
) -> bool:
    """
    Wait for an async operation to complete.
    
    Args:
        check_status_fn: Function that returns True when processing is complete
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
    
    Returns:
        bool: True if processing completed, False if timed out
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_status_fn():
            return True
        time.sleep(interval)
    return False

def load_test_document(filepath: Path) -> str:
    """
    Load a test document and return its contents.
    
    Args:
        filepath: Path to the test document
    
    Returns:
        str: Document contents
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def compare_embeddings(
    emb1: np.ndarray,
    emb2: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    Compare two embeddings for approximate equality.
    
    Args:
        emb1: First embedding array
        emb2: Second embedding array
        tolerance: Maximum allowed difference
    
    Returns:
        bool: True if embeddings are approximately equal
    """
    return np.allclose(emb1, emb2, atol=tolerance)

def create_test_metadata(
    num_docs: int = 1,
    base_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create sample metadata for test documents.
    
    Args:
        num_docs: Number of metadata entries to create
        base_metadata: Base metadata to extend
    
    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries
    """
    base_metadata = base_metadata or {}
    return [
        {
            'id': f'doc_{i}',
            'title': f'Test Document {i}',
            'created_at': '2024-01-01',
            'type': 'test',
            **base_metadata
        }
        for i in range(num_docs)
    ]

def setup_test_environment() -> Dict[str, Path]:
    """
    Set up temporary test environment with necessary directories.
    
    Returns:
        Dict[str, Path]: Dictionary of created test directories
    """
    import tempfile
    import shutil
    
    test_dir = Path(tempfile.mkdtemp())
    dirs = {
        'root': test_dir,
        'documents': test_dir / 'documents',
        'vectors': test_dir / 'vectors',
        'cache': test_dir / 'cache'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
        
    return dirs

def create_test_file(path: Union[str, Path], content: str) -> Path:
    """Create a test file with given content"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path

def cleanup_test_files(paths: List[Union[str, Path]]):
    """Clean up test files"""
    for path in paths:
        path = Path(path)
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path) 