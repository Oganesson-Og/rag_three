"""
Test Text Preprocessing Module
---------------------------

Tests for text cleaning, normalization, and preprocessing functionality.
"""

import pytest
from src.document_processing.preprocessing import TextPreprocessor
from src.document_processing.models import PreprocessingConfig

class TestTextPreprocessing:
    
    @pytest.fixture
    def text_preprocessor(self, config_manager):
        """Fixture providing a text preprocessor instance."""
        return TextPreprocessor(config_manager)
    
    @pytest.fixture
    def sample_texts(self):
        """Fixture providing various sample texts for preprocessing."""
        return {
            'basic': "This is a basic test text.",
            'noisy': "This   has\tmultiple   spaces\n\nand\tTABS!!",
            'html': "<p>This has <b>HTML</b> tags.</p>",
            'urls': "Check this link: https://example.com and email@domain.com",
            'special_chars': "Special chars: @#$%^&*()",
            'unicode': "Unicode chars: é, ñ, ü, ß, 漢字",
            'numbers': "Contains numbers 123 and dates 2024-01-01",
            'mixed_case': "MiXeD CaSe TeXt",
            'bullets': "• First item\n• Second item\n• Third item",
            'code_snippets': "```python\ndef test():\n    pass\n```"
        }
        
    def test_basic_cleaning(self, text_preprocessor):
        """Test basic text cleaning functionality."""
        text = "This   has\tmultiple   spaces\n\nand\tTABS!!"
        cleaned = text_preprocessor.clean_text(text)
        
        assert "  " not in cleaned  # No multiple spaces
        assert "\t" not in cleaned  # No tabs
        assert "\n\n" not in cleaned  # No multiple newlines
        assert cleaned.strip() == "This has multiple spaces and TABS!!"
        
    def test_html_removal(self, text_preprocessor):
        """Test HTML tag removal."""
        text = "<p>This has <b>HTML</b> tags.</p>"
        cleaned = text_preprocessor.clean_text(text, remove_html=True)
        
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert cleaned.strip() == "This has HTML tags."
        
    def test_url_handling(self, text_preprocessor):
        """Test URL handling options."""
        text = "Check this link: https://example.com"
        
        # Test URL removal
        cleaned_removed = text_preprocessor.clean_text(
            text,
            remove_urls=True
        )
        assert "https://" not in cleaned_removed
        
        # Test URL replacement
        cleaned_replaced = text_preprocessor.clean_text(
            text,
            replace_urls=True
        )
        assert "[URL]" in cleaned_replaced
        
    def test_number_handling(self, text_preprocessor):
        """Test number handling options."""
        text = "Contains numbers 123 and dates 2024-01-01"
        
        # Test number removal
        cleaned_removed = text_preprocessor.clean_text(
            text,
            remove_numbers=True
        )
        assert not any(c.isdigit() for c in cleaned_removed)
        
        # Test number normalization
        cleaned_normalized = text_preprocessor.clean_text(
            text,
            normalize_numbers=True
        )
        assert "[NUMBER]" in cleaned_normalized
        
    def test_case_normalization(self, text_preprocessor):
        """Test case normalization options."""
        text = "MiXeD CaSe TeXt"
        
        # Test lowercase
        cleaned_lower = text_preprocessor.clean_text(
            text,
            case='lower'
        )
        assert cleaned_lower == "mixed case text"
        
        # Test uppercase
        cleaned_upper = text_preprocessor.clean_text(
            text,
            case='upper'
        )
        assert cleaned_upper == "MIXED CASE TEXT"
        
    def test_special_character_handling(self, text_preprocessor):
        """Test special character handling."""
        text = "Special chars: @#$%^&*()"
        
        # Test removal
        cleaned_removed = text_preprocessor.clean_text(
            text,
            remove_special_chars=True
        )
        assert not any(c in cleaned_removed for c in "@#$%^&*()")
        
        # Test replacement
        cleaned_replaced = text_preprocessor.clean_text(
            text,
            replace_special_chars=True
        )
        assert "[SPECIAL]" in cleaned_replaced
        
    def test_unicode_handling(self, text_preprocessor):
        """Test Unicode character handling."""
        text = "Unicode chars: é, ñ, ü, ß, 漢字"
        
        # Test normalization
        cleaned = text_preprocessor.clean_text(
            text,
            normalize_unicode=True
        )
        assert "é" not in cleaned
        assert "e" in cleaned
        
        # Test preservation
        cleaned_preserved = text_preprocessor.clean_text(
            text,
            normalize_unicode=False
        )
        assert "é" in cleaned_preserved
        
    def test_whitespace_normalization(self, text_preprocessor):
        """Test whitespace normalization."""
        text = "Text  with \t multiple\n\nspaces"
        cleaned = text_preprocessor.clean_text(text)
        
        assert cleaned == "Text with multiple spaces"
        
    def test_custom_replacements(self, text_preprocessor):
        """Test custom text replacements."""
        text = "Replace custom patterns"
        replacements = [
            ('custom', 'specialized'),
            ('patterns', 'rules')
        ]
        
        cleaned = text_preprocessor.clean_text(
            text,
            custom_replacements=replacements
        )
        assert cleaned == "Replace specialized rules"
        
    def test_preprocessing_config(self, text_preprocessor):
        """Test preprocessing with configuration object."""
        config = PreprocessingConfig(
            remove_html=True,
            remove_urls=True,
            normalize_unicode=True,
            case='lower'
        )
        
        text = "<p>Test URL: https://example.com with é</p>"
        cleaned = text_preprocessor.clean_text(text, config=config)
        
        assert "<p>" not in cleaned
        assert "https://" not in cleaned
        assert "é" not in cleaned
        assert cleaned == cleaned.lower()
        
    def test_markdown_handling(self, text_preprocessor):
        """Test Markdown formatting handling."""
        text = "# Heading\n\n**Bold** and *italic*"
        
        cleaned = text_preprocessor.clean_text(
            text,
            remove_markdown=True
        )
        
        assert "#" not in cleaned
        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "Heading" in cleaned
        assert "Bold" in cleaned
        assert "italic" in cleaned
        
    def test_error_handling(self, text_preprocessor):
        """Test error handling in preprocessing."""
        # Test None input
        with pytest.raises(ValueError):
            text_preprocessor.clean_text(None)
            
        # Test empty string
        with pytest.raises(ValueError):
            text_preprocessor.clean_text("")
            
        # Test invalid config
        with pytest.raises(ValueError):
            text_preprocessor.clean_text("text", case='invalid')
            
    def test_preprocessing_consistency(self, text_preprocessor, sample_texts):
        """Test consistency of preprocessing results."""
        for name, text in sample_texts.items():
            # Multiple runs should produce same results
            result1 = text_preprocessor.clean_text(text)
            result2 = text_preprocessor.clean_text(text)
            assert result1 == result2
            
    def test_batch_preprocessing(self, text_preprocessor, sample_texts):
        """Test batch preprocessing functionality."""
        texts = list(sample_texts.values())
        
        # Test batch processing
        cleaned_batch = text_preprocessor.clean_texts(texts)
        
        assert len(cleaned_batch) == len(texts)
        assert all(isinstance(text, str) for text in cleaned_batch)
        
    def test_performance(self, text_preprocessor):
        """Test preprocessing performance."""
        # Generate large text
        large_text = "Sample text with various patterns. " * 1000
        
        import time
        start_time = time.time()
        
        text_preprocessor.clean_text(large_text)
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0  # Should process quickly 