"""
Test Audio Processing Module
-------------------------

Tests for audio processing, transcription, and feature extraction.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
from src.audio.processor import AudioProcessor
from src.audio.models import AudioConfig, TranscriptionResult

class TestAudioProcessor:
    
    @pytest.fixture
    def audio_processor(self, config_manager):
        """Fixture providing an audio processor instance."""
        return AudioProcessor(config_manager)
    
    @pytest.fixture
    def sample_audio(self):
        """Fixture providing a sample audio file."""
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 2.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
            sf.write(tf.name, audio_data, sample_rate)
            return Path(tf.name)
            
    def test_audio_loading(self, audio_processor, sample_audio):
        """Test audio file loading."""
        audio_data, sample_rate = audio_processor.load_audio(sample_audio)
        
        assert isinstance(audio_data, np.ndarray)
        assert sample_rate == 16000
        assert len(audio_data.shape) == 1  # Mono audio
        assert not np.any(np.isnan(audio_data))
        
    def test_audio_preprocessing(self, audio_processor, sample_audio):
        """Test audio preprocessing."""
        # Load audio
        audio_data, sample_rate = audio_processor.load_audio(sample_audio)
        
        # Test normalization
        normalized = audio_processor.normalize_audio(audio_data)
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test resampling
        resampled = audio_processor.resample_audio(audio_data, sample_rate, target_rate=8000)
        assert len(resampled) == len(audio_data) * 8000 // sample_rate
        
    def test_audio_transcription(self, audio_processor, sample_audio):
        """Test audio transcription."""
        result = audio_processor.transcribe(sample_audio)
        
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert isinstance(result.segments, list)
        assert all('start' in segment for segment in result.segments)
        
    def test_batch_transcription(self, audio_processor, tmp_path):
        """Test batch audio transcription."""
        # Create multiple test audio files
        audio_files = []
        for i in range(3):
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * (440 + i*100) * t)
            
            audio_path = tmp_path / f"test_audio_{i}.wav"
            sf.write(audio_path, audio_data, sample_rate)
            audio_files.append(audio_path)
            
        results = audio_processor.batch_transcribe(audio_files)
        
        assert len(results) == len(audio_files)
        assert all(isinstance(result, TranscriptionResult) for result in results)
        
    def test_feature_extraction(self, audio_processor, sample_audio):
        """Test audio feature extraction."""
        # Load audio
        audio_data, sample_rate = audio_processor.load_audio(sample_audio)
        
        # Extract features
        features = audio_processor.extract_features(audio_data, sample_rate)
        
        assert isinstance(features, dict)
        assert 'mfcc' in features
        assert 'spectral_centroid' in features
        assert isinstance(features['mfcc'], np.ndarray)
        
    def test_audio_segmentation(self, audio_processor, sample_audio):
        """Test audio segmentation."""
        segments = audio_processor.segment_audio(sample_audio)
        
        assert isinstance(segments, list)
        assert all('start' in segment for segment in segments)
        assert all('end' in segment for segment in segments)
        assert all('audio_data' in segment for segment in segments)
        
    def test_noise_reduction(self, audio_processor, sample_audio):
        """Test noise reduction."""
        # Load audio
        audio_data, sample_rate = audio_processor.load_audio(sample_audio)
        
        # Add noise
        noise = np.random.normal(0, 0.01, len(audio_data))
        noisy_audio = audio_data + noise
        
        # Apply noise reduction
        cleaned_audio = audio_processor.reduce_noise(noisy_audio)
        
        assert len(cleaned_audio) == len(audio_data)
        assert np.mean(np.abs(cleaned_audio)) < np.mean(np.abs(noisy_audio))
        
    def test_error_handling(self, audio_processor):
        """Test error handling in audio processing."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            audio_processor.load_audio(Path("nonexistent.wav"))
            
        # Test with invalid audio file
        with tempfile.NamedTemporaryFile(suffix='.wav') as tf:
            tf.write(b"invalid audio data")
            tf.flush()
            
            with pytest.raises(Exception):
                audio_processor.load_audio(Path(tf.name))
                
    def test_audio_config(self, config_manager):
        """Test audio configuration."""
        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            model_type='whisper-small',
            language='en'
        )
        
        processor = AudioProcessor(config_manager, config=config)
        
        assert processor.config.sample_rate == 16000
        assert processor.config.channels == 1
        assert processor.config.model_type == 'whisper-small'
        
    def test_performance_metrics(self, audio_processor, sample_audio):
        """Test performance metrics collection."""
        with audio_processor.collect_metrics() as metrics:
            audio_processor.transcribe(sample_audio)
            
        assert 'processing_time' in metrics
        assert 'memory_usage' in metrics
        assert metrics['processing_time'] > 0 