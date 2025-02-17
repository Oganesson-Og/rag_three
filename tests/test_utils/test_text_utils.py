"""
Test Text Utilities
----------------
"""

import pytest
from src.utils.text_utils import TextUtils

class TestTextUtils:
    @pytest.fixture
    def text_utils(self):
        return TextUtils()
        
    @pytest.fixture
    def sample_text(self):
        return """This is a sample text.
        It has multiple lines   with extra spaces.
        It also has some <html> tags and special chars: @#$%^&*"""
        
    def test_normalize_text(self, text_utils, sample_text):
        """Test text normalization."""
        normalized = text_utils.normalize_text(sample_text)
        assert "  " not in normalized
        assert "\n" not in normalized
        assert normalized.startswith("This")
        assert len(normalized) < len(sample_text)
        
    def test_sanitize_text(self, text_utils):
        """Test text sanitization."""
        dirty_text = "<p>Hello</p> with @#$ special chars!"
        clean_text = text_utils.sanitize_text(dirty_text)
        assert "<p>" not in clean_text
        assert "</p>" not in clean_text
        assert "@#$" not in clean_text
        assert "Hello with special chars!" == clean_text
        
    def test_get_text_metrics(self, text_utils, sample_text):
        """Test text metrics calculation."""
        metrics = text_utils.get_text_metrics(sample_text)
        assert 'char_count' in metrics
        assert 'word_count' in metrics
        assert 'sentence_count' in metrics
        assert 'avg_word_length' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
        
    def test_truncate(self, text_utils):
        """Test text truncation."""
        text = "This is a long text that needs truncation"
        truncated = text_utils.truncate(text, max_length=10)
        assert len(truncated) == 10
        assert truncated.endswith("...")
        
        # Test short text
        short_text = "Short"
        assert text_utils.truncate(short_text, max_length=10) == short_text 