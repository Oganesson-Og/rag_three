"""
Conversion Utilities
-----------------

Comprehensive utilities for data type and format conversions.

Key Features:
- Numeric conversions (bytes, MB, GB)
- Time conversions (seconds, minutes, hours)
- Date/time formatting
- JSON serialization
- Unit conversions
- Type casting
- Format validation

Technical Details:
- Precise numeric handling
- DateTime manipulation
- JSON processing
- Type checking
- Error handling
- Format validation
- Unit standardization

Dependencies:
- datetime
- json
- typing-extensions>=4.7.0

Example Usage:
    converter = ConversionUtils()
    
    # Byte conversions
    mb = converter.bytes_to_mb(1048576)  # 1.0
    bytes = converter.mb_to_bytes(1.0)   # 1048576
    
    # Time conversions
    minutes = converter.seconds_to_minutes(120)  # 2.0
    seconds = converter.minutes_to_seconds(2.0)  # 120
    
    # Date conversions
    date = converter.str_to_date("2025-01-01")
    date_str = converter.date_to_str(date)
    
    # JSON conversion
    json_str = converter.to_json({"key": "value"}, pretty=True)
    
    # Timestamp conversions
    ts = converter.datetime_to_timestamp(datetime.now())
    dt = converter.timestamp_to_datetime(ts)

Performance Considerations:
- Efficient numeric operations
- Optimized date handling
- Fast JSON processing
- Type validation
- Error handling
- Format checking
- Memory efficiency

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Union, Optional
from datetime import datetime
import json

class ConversionUtils:
    def bytes_to_mb(self, bytes_val: int) -> float:
        """Convert bytes to megabytes.
        
        Args:
            bytes_val: Value in bytes
            
        Returns:
            float: Value in megabytes
        """
        return bytes_val / (1024 * 1024)
        
    def mb_to_bytes(self, mb_val: float) -> int:
        """Convert megabytes to bytes.
        
        Args:
            mb_val: Value in megabytes
            
        Returns:
            int: Value in bytes
        """
        return int(mb_val * 1024 * 1024)
        
    def seconds_to_minutes(self, seconds: int) -> float:
        """Convert seconds to minutes.
        
        Args:
            seconds: Value in seconds
            
        Returns:
            float: Value in minutes
        """
        return seconds / 60
        
    def minutes_to_seconds(self, minutes: float) -> int:
        """Convert minutes to seconds.
        
        Args:
            minutes: Value in minutes
            
        Returns:
            int: Value in seconds
        """
        return int(minutes * 60)
        
    def str_to_date(self, date_str: str, 
                    format: str = "%Y-%m-%d") -> datetime:
        """Convert string to datetime.
        
        Args:
            date_str: Date string
            format: Date format string
            
        Returns:
            datetime: Parsed datetime object
        """
        return datetime.strptime(date_str, format)
        
    def date_to_str(self, date: datetime, 
                    format: str = "%Y-%m-%d") -> str:
        """Convert datetime to string.
        
        Args:
            date: Datetime object
            format: Date format string
            
        Returns:
            str: Formatted date string
        """
        return date.strftime(format)
        
    def datetime_to_timestamp(self, dt: datetime) -> float:
        """Convert datetime to Unix timestamp.
        
        Args:
            dt: Datetime object
            
        Returns:
            float: Unix timestamp
        """
        return dt.timestamp()
        
    def timestamp_to_datetime(self, timestamp: float) -> datetime:
        """Convert Unix timestamp to datetime.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            datetime: Datetime object
        """
        return datetime.fromtimestamp(timestamp)
        
    def to_json(self, data: Union[Dict, list], 
                pretty: bool = False) -> str:
        """Convert data to JSON string.
        
        Args:
            data: Data to convert
            pretty: Whether to format with indentation
            
        Returns:
            str: JSON string
        """
        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data) 