"""
Audio Models Module
-----------------

Pydantic models for handling audio processing configurations, segments,
and results in the educational content processing system.

Key Features:
- Audio configuration management
- Segment-based processing
- Multi-speaker support
- Feature extraction models
- Metadata handling
- Timestamp tracking
- Quality metrics

Technical Details:
- Built with Pydantic models
- NumPy array support
- Configurable sampling rates
- Multi-channel support
- Flexible format handling
- Time-based segmentation
- Feature extraction support

Dependencies:
- pydantic>=2.0.0
- numpy>=1.24.0
- python-dateutil>=2.8.2
- typing-extensions>=4.7.0

Example Usage:
    # Audio configuration
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        format="wav",
        model_name="whisper-large-v3"
    )
    
    # Process audio segments
    segment = AudioSegment(
        start=0.0,
        end=5.0,
        text="Audio content here",
        confidence=0.95
    )
    
    # Handle processing results
    result = AudioResult(
        text="Transcribed text",
        segments=[segment],
        language="en"
    )

Model Categories:
- Configuration Models (AudioConfig)
- Processing Models (AudioSegment)
- Result Models (AudioResult)
- Feature Models (AudioFeatures)

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
from numpy.typing import NDArray

class AudioConfig(BaseModel):
    """
    Audio configuration model for processing settings.
    
    Attributes:
        sample_rate (int): Audio sampling rate in Hz
        channels (int): Number of audio channels
        duration (Optional[float]): Audio duration in seconds
        format (str): Audio file format (e.g., "wav", "mp3")
        device (Optional[str]): Processing device (e.g., "cpu", "cuda")
        model_name (str): Name of the processing model
    """
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    duration: Optional[float] = None
    format: str = Field(default="wav")
    device: Optional[str] = None
    model_name: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

class AudioSegment(BaseModel):
    """
    Audio segment model for time-based content chunks.
    
    Attributes:
        start (float): Start time in seconds
        end (float): End time in seconds
        text (str): Transcribed text content
        confidence (float): Confidence score (0.0 to 1.0)
        speaker (Optional[str]): Speaker identifier
    """
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    speaker: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class AudioResult(BaseModel):
    """
    Audio processing result model with metadata.
    
    Attributes:
        text (str): Complete transcribed text
        segments (List[AudioSegment]): List of processed segments
        language (Optional[str]): Detected language
        metadata (Dict[str, Any]): Additional processing metadata
        timestamp (datetime): Processing timestamp
    """
    text: str = Field(..., min_length=1)
    segments: List[AudioSegment] = Field(default_factory=list)
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

class AudioFeatures(BaseModel):
    """
    Audio features model for extracted characteristics.
    
    Attributes:
        mfcc (Optional[NDArray]): Mel-frequency cepstral coefficients
        spectral (Optional[NDArray]): Spectral features
        temporal (Optional[NDArray]): Temporal features
        metadata (Dict[str, Any]): Feature extraction metadata
    """
    mfcc: Optional[NDArray[np.float32]] = None
    spectral: Optional[NDArray[np.float32]] = None
    temporal: Optional[NDArray[np.float32]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True 