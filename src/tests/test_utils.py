"""
Test Utilities Module
-------------------

Common utilities and helper functions for testing.
Provides shared functionality across test modules.

Features:
- Mock data generation
- Test fixtures
- Helper functions
- Assertion utilities
- Performance metrics

Key Components:
1. MockData: Test data generation
2. TestHelpers: Common test operations
3. Assertions: Custom test assertions
4. Metrics: Performance measurement

Technical Details:
- Configurable mock data
- Reusable fixtures
- Time measurement
- Resource cleanup

Dependencies:
- pytest>=7.0.0
- mock>=4.0.0
- numpy>=1.19.0

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from ..utils.text_cleaner import TextCleaner
from ..utils.token_counter import num_tokens_from_string
from ..utils.image_utils import convert_to_grayscale, resize_image
from ..utils.rag_tokenizer import RagTokenizer

class TestTextCleaner:
    @pytest.fixture
    def cleaner(self):
        return TextCleaner()
    
    def test_basic_cleaning(self, cleaner):
        text = "This is a test\nwith multiple   spaces\tand tabs."
        cleaned = cleaner.clean(text)
        assert isinstance(cleaned, str)
        assert "  " not in cleaned  # No double spaces
        assert "\t" not in cleaned  # No tabs
        
    def test_latex_cleaning(self, cleaner):
        latex_text = r"This is a \textbf{bold} and \emph{emphasized} text with $\alpha = \beta$"
        cleaned = cleaner.clean_latex(latex_text)
        assert "\\textbf" not in cleaned
        assert "\\emph" not in cleaned
        
    def test_html_cleaning(self, cleaner):
        html_text = "<p>This is a <b>bold</b> text with <a href='#'>link</a></p>"
        cleaned = cleaner.clean_html(html_text)
        assert "<" not in cleaned
        assert ">" not in cleaned

class TestTokenCounter:
    def test_token_counting(self):
        text = "This is a test sentence for counting tokens."
        count = num_tokens_from_string(text)
        assert isinstance(count, int)
        assert count > 0
        
    def test_empty_string(self):
        assert num_tokens_from_string("") == 0
        
    def test_special_characters(self):
        text = "Special chars: !@#$%^&*()"
        count = num_tokens_from_string(text)
        assert count > 0

class TestImageUtils:
    @pytest.fixture
    def sample_image(self):
        # Create a simple test image
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_grayscale_conversion(self, sample_image):
        gray = convert_to_grayscale(sample_image)
        assert len(gray.shape) == 2
        assert gray.shape[:2] == sample_image.shape[:2]
        
    def test_image_resizing(self, sample_image):
        scale_factor = 0.5
        resized = resize_image(sample_image, scale_factor)
        expected_shape = (
            int(sample_image.shape[0] * scale_factor),
            int(sample_image.shape[1] * scale_factor),
            sample_image.shape[2]
        )
        assert resized.shape == expected_shape

class TestRagTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return RagTokenizer()
    
    def test_tokenization(self, tokenizer):
        text = "This is a test sentence."
        tokens = tokenizer.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        
    def test_batch_tokenization(self, tokenizer):
        texts = ["First sentence.", "Second sentence."]
        batch_tokens = tokenizer.batch_tokenize(texts)
        assert len(batch_tokens) == len(texts)
        assert all(len(tokens) > 0 for tokens in batch_tokens)
        
    def test_special_tokens(self, tokenizer):
        text = "[CLS] This is a test [SEP]"
        tokens = tokenizer.tokenize(text)
        assert "[CLS]" in tokens
        assert "[SEP]" in tokens

class TestPerformanceMetrics:
    @pytest.fixture
    def timer(self):
        return Timer()
    
    def test_execution_time(self, timer):
        with timer as t:
            # Simulate some work
            _ = [i * i for i in range(1000)]
        assert t.elapsed >= 0
        
    @patch('time.time')
    def test_timer_precision(self, mock_time, timer):
        mock_time.side_effect = [0, 0.5]  # 500ms difference
        with timer as t:
            pass
        assert t.elapsed == 0.5

class Timer:
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()
        self.elapsed = (self.end - self.start).total_seconds() 