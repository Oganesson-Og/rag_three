"""
Custom pytest markers for test categorization
"""
import pytest

# Performance markers
slow_test = pytest.mark.slow
benchmark = pytest.mark.benchmark

# Test types
integration = pytest.mark.integration
unit = pytest.mark.unit
e2e = pytest.mark.e2e

# Resource requirements
gpu = pytest.mark.gpu
memory_intensive = pytest.mark.memory_intensive
network = pytest.mark.network

# Test categories
document_processing = pytest.mark.document_processing
embedding = pytest.mark.embedding
retrieval = pytest.mark.retrieval
math_content = pytest.mark.math_content
error_handling = pytest.mark.error_handling 