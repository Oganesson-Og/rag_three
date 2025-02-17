"""
Audio Extractor Module
--------------------

Handles audio file processing and transcription with advanced features
and comprehensive error handling.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, BinaryIO, Optional
import tempfile
from datetime import datetime

import whisper
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from transformers import pipeline

from ...rag.models import (
    Document,
    ProcessingEvent,
    ProcessingMetrics,
    ContentModality,
    ProcessingStage
)
from .base import BaseExtractor
from ...utils.cache import AdvancedCache

logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass

class AudioExtractor(BaseExtractor):
    """
    Enhanced audio content extractor with advanced processing capabilities.
    
    Features:
    - Multiple audio format support
    - Automatic format detection
    - Audio preprocessing
    - Chunked processing for large files
    - Multiple transcription engine support
    - Caching at multiple levels
    """
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
    CHUNK_SIZE = 30000  # 30 seconds in milliseconds
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize audio extractor."""
        super().__init__(config)  # Properly call parent class init
        self.cache = AdvancedCache()
        
        # Initialize transcription models
        self.whisper_model = self._init_whisper_model()
        self.backup_transcriber = self._init_backup_transcriber()
        
        # Audio processing settings
        self.target_sample_rate = config.get('sample_rate', 16000)
        self.target_channels = config.get('channels', 1)
        self.normalize_audio = config.get('normalize_audio', True)
        self.sample_rate = config.get('sample_rate', 16000)
        self.model_name = config.get('model_name', 'whisper-1')
        
    def _init_whisper_model(self):
        """Initialize primary Whisper model."""
        model_size = self.config.get('whisper_model_size', 'base')
        try:
            return whisper.load_model(model_size)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            return None
            
    def _init_backup_transcriber(self):
        """Initialize backup transcription model."""
        try:
            return pipeline(
                "automatic-speech-recognition",
                model=self.config.get('backup_model', 'facebook/wav2vec2-base-960h')
            )
        except Exception as e:
            logger.error(f"Failed to load backup transcriber: {str(e)}")
            return None
    
    async def extract(self, document: Document) -> Document:
        """
        Extract and transcribe audio content.
        
        Args:
            document: Input document with audio content
            
        Returns:
            Document with transcribed content
        """
        try:
            start_time = datetime.now()
            
            # Check cache first
            cache_key = self._generate_cache_key(document)
            if cached_doc := await self.cache.get(cache_key):
                logger.info(f"Cache hit for {cache_key}")
                return cached_doc
            
            # Load and preprocess audio
            audio_data = await self._load_and_preprocess_audio(document.content)
            
            # Process audio in chunks if necessary
            if len(audio_data) > self.CHUNK_SIZE:
                transcription = await self._process_large_audio(audio_data)
            else:
                transcription = await self._transcribe_audio(audio_data)
            
            # Update document with transcription
            document.content = transcription
            document.metadata.update({
                "duration_seconds": len(audio_data) / 1000,
                "sample_rate": self.target_sample_rate,
                "channels": self.target_channels,
                "transcription_engine": "whisper" if self.whisper_model else "backup",
                "processing_time": (datetime.now() - start_time).total_seconds()
            })
            
            # Record processing event
            document.add_processing_event(ProcessingEvent(
                stage=ProcessingStage.EXTRACTED,
                processor="AudioExtractor",
                metrics=ProcessingMetrics(
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    token_count=len(transcription.split())
                )
            ))
            
            # Cache processed document
            await self.cache.set(cache_key, document)
            
            return document
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}", exc_info=True)
            raise AudioProcessingError(f"Audio extraction failed: {str(e)}")
    
    async def _load_and_preprocess_audio(
        self,
        source: Union[str, Path, BinaryIO]
    ) -> AudioSegment:
        """Load and preprocess audio data."""
        try:
            # Load audio
            if isinstance(source, (str, Path)):
                audio = AudioSegment.from_file(source)
            else:
                with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                    tmp.write(source.read())
                    tmp.flush()
                    audio = AudioSegment.from_wav(tmp.name)
            
            # Preprocess audio
            audio = (audio
                    .set_frame_rate(self.target_sample_rate)
                    .set_channels(self.target_channels))
            
            if self.normalize_audio:
                audio = audio.normalize()
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}", exc_info=True)
            raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")
    
    async def _transcribe_audio(
        self,
        audio: AudioSegment,
        start_time: float = 0
    ) -> str:
        """Transcribe audio segment."""
        try:
            # Convert to numpy array
            audio_array = np.array(audio.get_array_of_samples())
            
            # Try primary transcription
            if self.whisper_model:
                result = self.whisper_model.transcribe(
                    audio_array,
                    start_time=start_time
                )
                return result["text"]
            
            # Fall back to backup transcriber
            if self.backup_transcriber:
                result = self.backup_transcriber(audio_array)
                return result["text"]
            
            raise AudioProcessingError("No transcription engine available")
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise AudioProcessingError(f"Transcription failed: {str(e)}")
    
    async def _process_large_audio(self, audio: AudioSegment) -> str:
        """Process large audio files in chunks."""
        transcriptions = []
        
        for i in range(0, len(audio), self.CHUNK_SIZE):
            chunk = audio[i:i + self.CHUNK_SIZE]
            transcription = await self._transcribe_audio(
                chunk,
                start_time=i / 1000.0  # Convert to seconds
            )
            transcriptions.append(transcription)
        
        return " ".join(transcriptions)
    
    def _generate_cache_key(self, document: Document) -> str:
        """Generate cache key for document."""
        content_hash = self._hash_content(document.content)
        return f"audio:{content_hash}:{self.config.get('whisper_model_size')}"
    
    @staticmethod
    def _hash_content(content: Union[str, bytes, BinaryIO]) -> str:
        """Generate hash for content."""
        import hashlib
        
        if isinstance(content, str):
            return hashlib.md5(content.encode()).hexdigest()
        elif isinstance(content, bytes):
            return hashlib.md5(content).hexdigest()
        else:
            content.seek(0)
            file_hash = hashlib.md5(content.read()).hexdigest()
            content.seek(0)
            return file_hash 