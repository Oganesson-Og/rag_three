"""
Test Conversion Utilities
----------------------
"""

import pytest
from datetime import datetime
from src.utils.conversion_utils import ConversionUtils

class TestConversionUtils:
    @pytest.fixture
    def conversion_utils(self):
        return ConversionUtils()
        
    def test_str_to_date(self, conversion_utils):
        """Test string to date conversion."""
        date_str = "2024-01-01"
        date = conversion_utils.str_to_date(date_str)
        assert isinstance(date, datetime)
        assert date.year == 2024
        assert date.month == 1
        assert date.day == 1
        
    def test_date_to_str(self, conversion_utils):
        """Test date to string conversion."""
        date = datetime(2024, 1, 1)
        date_str = conversion_utils.date_to_str(date)
        assert date_str == "2024-01-01"
        
    def test_datetime_timestamp_conversion(self, conversion_utils):
        """Test datetime and timestamp conversions."""
        now = datetime.now()
        timestamp = conversion_utils.datetime_to_timestamp(now)
        converted = conversion_utils.timestamp_to_datetime(timestamp)
        
        assert isinstance(timestamp, float)
        assert isinstance(converted, datetime)
        assert abs(converted.timestamp() - now.timestamp()) < 1
        
    def test_to_json(self, conversion_utils):
        """Test JSON conversion."""
        data = {
            'name': 'Test',
            'value': 123,
            'list': [1, 2, 3]
        }
        
        # Test normal conversion
        json_str = conversion_utils.to_json(data)
        assert isinstance(json_str, str)
        assert '"name":"Test"' in json_str.replace(" ", "")
        
        # Test pretty print
        pretty_json = conversion_utils.to_json(data, pretty=True)
        assert "\n" in pretty_json
        assert "  " in pretty_json 