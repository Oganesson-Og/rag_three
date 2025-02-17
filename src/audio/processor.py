"""
Audio Processing Module
--------------------

Comprehensive audio processing system for educational content using Whisper
and advanced audio analysis techniques.

Key Features:
- Audio transcription with Whisper
- Speaker diarization
- Technical term extraction
- Audio quality analysis
- Domain-specific processing
- Multi-language support
- Educational context awareness
- Batch processing capabilities

Technical Details:
- Whisper model integration
- Speaker recognition
- Signal processing
- Feature extraction
- Quality metrics calculation
- Educational term detection
- Cross-domain analysis
- Performance optimization

Dependencies:
- whisper>=1.0.0
- torch>=2.0.0
- numpy>=1.24.0
- librosa>=0.10.0
- speechbrain>=0.5.15
- keybert>=0.7.0
- nltk>=3.8.1
- soundfile>=0.12.1

Example Usage:
    # Basic audio processing
    processor = AudioProcessor(model_name="whisper-large-v3")
    result = processor.process_audio("lecture.wav")
    
    # Advanced processing with context
    result = processor.process_audio(
        "lecture.wav",
        context={
            "subject": "physics",
            "topic": "quantum_mechanics",
            "keywords": ["wave function", "uncertainty"]
        }
    )
    
    # Multi-language processing
    result = processor.translate_to_english("foreign_lecture.mp3")

Processing Features:
- Audio quality analysis
- Speaker identification
- Technical term extraction
- Domain-specific processing
- Educational context enhancement
- Cross-language support
- Batch processing

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import whisper
import torch
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Optional, List, Tuple, Union, Any, Set
import librosa
from dataclasses import dataclass
from datetime import datetime
from .models import AudioConfig
import logging
from numpy.typing import NDArray
import soundfile as sf
import time
from nltk.corpus import stopwords
import tempfile
from openai import OpenAI

# Type aliases
AudioData = NDArray[np.float32]
AudioResult = Dict[str, Any]
AudioArray = NDArray[np.float32]
ProcessingResult = Dict[str, Any]

@dataclass
class TranscriptionSegment:
    """
    Represents a segment of transcribed audio.
    
    Attributes:
        text (str): Transcribed text content
        start (float): Start time in seconds
        end (float): End time in seconds
        confidence (float): Confidence score (0.0 to 1.0)
    """
    text: str
    start: float
    end: float
    confidence: float

class AudioProcessor:
    """
    Enhanced audio processing and feature extraction for educational content.
    
    Attributes:
        model_name (str): Name of the Whisper model to use
        device (str): Processing device (CPU/CUDA)
        config (Dict): Configuration options
        logger (logging.Logger): Logger instance
        whisper_model: Loaded Whisper model
        diarization_model: Speaker diarization model
        technical_term_extractor: Term extraction model
    
    Methods:
        process_audio: Main audio processing pipeline
        detect_language: Detect audio language
        translate_to_english: Translate non-English audio
        extract_features: Extract audio features
        calculate_quality: Calculate audio quality metrics
    """
    
    def __init__(
        self,
        model_name: str = "whisper-large-v3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize audio processor with specified model and configuration.
        
        Args:
            model_name: Whisper model variant to use
            device: Processing device (CPU/CUDA)
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.device = device
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.whisper_model = whisper.load_model(model_name).to(device)
        self.diarization_model = self._initialize_diarization_model()
        self.technical_term_extractor = self._initialize_term_extractor()
        
    def process_audio(
        self,
        audio_path: Union[str, Path],
        task_type: str = "transcribe",
        language: str = "en",
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process audio with enhanced features."""
        try:
            start_time = time.time()
            
            # Load and preprocess audio
            audio = self._load_and_preprocess_audio(audio_path)
            
            # Transcribe audio
            transcription = self.whisper_model.transcribe(
                audio,
                task=task_type,
                language=language
            )
            
            # Perform speaker diarization
            speakers = self.diarization_model.diarize(audio)
            
            # Extract technical terms
            technical_terms = self.technical_term_extractor.extract_terms(
                transcription["text"],
                context=context
            )
            
            # Calculate quality metrics
            audio_quality = self._calculate_audio_quality(audio)
            
            # Enhance with educational context
            if context and context.get("subject"):
                domain_terms = self._extract_domain_terms(
                    transcription["text"],
                    subject=context["subject"]
                )
            else:
                domain_terms = []
            
            return {
                "text": transcription["text"],
                "confidence": transcription["confidence"],
                "detected_language": transcription["language"],
                "duration": len(audio) / self.whisper_model.sample_rate,
                "segments": transcription["segments"],
                "word_timestamps": transcription.get("word_timestamps", []),
                "speaker_diarization": speakers,
                "technical_terms": technical_terms,
                "domain_specific_terms": domain_terms,
                "audio_quality": audio_quality,
                "processing_time": time.time() - start_time,
                "model_used": self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Audio processing error: {str(e)}", exc_info=True)
            raise
            
    def _initialize_diarization_model(self):
        """Initialize speaker diarization model."""
        try:
            # Import the specific module
            from speechbrain.pretrained.interfaces import SpeakerDiarization
            
            diarization = SpeakerDiarization.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            return diarization
            
            # Option 2: Using resemblyzer
            # from resemblyzer import VoiceEncoder
            # return VoiceEncoder()
            
            # Option 3: Using NeMo
            # import nemo.collections.asr as nemo_asr
            # model = nemo_asr.models.ClusteringDiarizer.from_pretrained('titanet_large')
            # return model
            
        except ImportError:
            self.logger.warning("Diarization libraries not installed. Diarization disabled.")
            return None

    def _initialize_term_extractor(self):
        """Initialize technical term extraction model."""
        try:
            from keybert import KeyBERT
            
            model = KeyBERT(model=self.config.get("term_extract_model", "all-MiniLM-L6-v2"))
            return model
            
        except ImportError:
            self.logger.warning("KeyBERT not installed. Term extraction disabled.")
            return None

    def _calculate_audio_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate audio quality metrics."""
        return {
            "signal_to_noise": self._calculate_snr(audio),
            "clarity": self._calculate_clarity(audio),
            "volume_level": self._calculate_volume(audio)
        }
        
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        if len(audio) == 0:
            return 0.0
            
        # Estimate noise from silent regions
        frame_length = 2048
        hop_length = 512
        
        # Calculate short-time energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
        
        # Estimate noise floor as the 10th percentile of energy
        noise_energy = np.percentile(energy, 10)
        
        # Calculate signal energy as the mean of the top 10% of energy values
        signal_energy = np.mean(np.percentile(energy, 90))
        
        # Avoid division by zero
        if noise_energy == 0:
            return 100.0
            
        snr = 10 * np.log10(signal_energy / noise_energy)
        return float(snr)

    def _calculate_clarity(self, audio: np.ndarray) -> float:
        """Calculate audio clarity score."""
        # Calculate spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio)
        
        # Calculate spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio)
        
        # Normalize and combine metrics
        norm_centroid = np.mean(centroid) / 1000  # Normalize to 0-1 range
        norm_bandwidth = np.mean(bandwidth) / 5000  # Normalize to 0-1 range
        
        clarity_score = (norm_centroid + (1 - norm_bandwidth)) / 2
        return float(clarity_score)

    def _calculate_volume(self, audio: np.ndarray) -> float:
        """Calculate average volume level."""
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio)
        
        # Convert to decibels and normalize
        db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Normalize to 0-1 range
        normalized_db = (db - db.min()) / (db.max() - db.min())
        
        return float(np.mean(normalized_db))

    def _extract_domain_terms(
        self,
        text: str,
        subject: str
    ) -> List[str]:
        """Extract domain-specific terms."""
        if not self.technical_term_extractor:
            return []
            
        # Load domain-specific stopwords
        domain_stopwords = self._load_domain_stopwords(subject)
        
        # Extract keywords with domain context
        keywords = self.technical_term_extractor.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words=domain_stopwords,
            use_maxsum=True,
            nr_candidates=20,
            top_n=10
        )
        
        return [term for term, score in keywords if score > 0.3]

    def _load_domain_stopwords(self, subject: str) -> Set[str]:
        """Load domain-specific stopwords."""
        base_stopwords = set(stopwords.words('english'))
        
        # Add domain-specific stopwords
        domain_stops = {
            'science': {'experiment', 'observe', 'hypothesis', 'theory'},
            'math': {'calculate', 'solve', 'equation', 'problem'},
            'history': {'year', 'date', 'period', 'era'}
        }
        
        return base_stopwords.union(domain_stops.get(subject, set()))

    def _load_audio(
        self,
        audio: Union[str, bytes]
    ) -> AudioArray:
        """Load audio from file or bytes."""
        try:
            if isinstance(audio, str):
                return librosa.load(audio, sr=self.whisper_model.sample_rate)[0]
            else:
                return sf.read(audio)[0]
        except Exception as e:
            self.logger.error(f"Audio loading error: {str(e)}")
            raise

    def _extract_features(
        self,
        audio: AudioArray
    ) -> Dict[str, NDArray[np.float32]]:
        """Extract audio features."""
        try:
            return {
                'mfcc': librosa.feature.mfcc(y=audio, sr=self.whisper_model.sample_rate),
                'spectral': librosa.feature.spectral_centroid(y=audio, sr=self.whisper_model.sample_rate)
            }
        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            raise

    def _transcribe_audio(
        self,
        audio: AudioArray,
        system_prompt: Optional[str] = None,
        timestamp_granularity: Optional[List[str]] = None,
        post_process: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI's Whisper API with enhanced features.
        
        Args:
            audio: Audio data array
            system_prompt: Optional prompt for post-processing corrections
            timestamp_granularity: List of timestamp detail levels ("segment" and/or "word")
            post_process: Whether to apply GPT-4 post-processing for accuracy
        
        Returns:
            Dictionary containing transcription results and metadata
        """
        try:
            # Save audio array to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio:
                sf.write(temp_audio.name, audio, self.whisper_model.sample_rate)
                
                # Initialize OpenAI client
                client = OpenAI()
                
                # Prepare transcription parameters
                transcription_params = {
                    "model": "whisper-1",
                    "file": open(temp_audio.name, "rb"),
                }
                
                # Add timestamp granularity if specified
                if timestamp_granularity:
                    transcription_params.update({
                        "response_format": "verbose_json",
                        "timestamp_granularities": timestamp_granularity
                    })
                
                # Get initial transcription
                transcription = client.audio.transcriptions.create(**transcription_params)
                
                # Apply post-processing if requested
                if post_process and system_prompt:
                    corrected_text = self._post_process_transcription(
                        transcription.text if hasattr(transcription, 'text') else transcription['text'],
                        system_prompt
                    )
                else:
                    corrected_text = transcription.text if hasattr(transcription, 'text') else transcription['text']
                
                # Prepare response
                result = {
                    "text": corrected_text,
                    "original_text": transcription.text if hasattr(transcription, 'text') else transcription['text'],
                    "model_used": "whisper-1",
                    "timestamp_granularity": timestamp_granularity,
                    "post_processed": post_process
                }
                
                # Add timestamps if available
                if timestamp_granularity:
                    if "word" in timestamp_granularity:
                        result["word_timestamps"] = transcription.words
                    if "segment" in timestamp_granularity:
                        result["segments"] = transcription.segments
                
                return result
                
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}", exc_info=True)
            raise

    def _post_process_transcription(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 0
    ) -> str:
        """
        Post-process transcription using GPT-4 for improved accuracy.
        
        Args:
            text: Original transcription text
            system_prompt: System prompt for GPT-4
            temperature: Temperature for GPT-4 response (0 for most deterministic)
        
        Returns:
            Corrected transcription text
        """
        try:
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4",
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {str(e)}. Using original text.")
            return text

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return data and sample rate."""
        audio_data, sample_rate = sf.read(file_path)
        return audio_data, sample_rate

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data."""
        return audio_data / np.max(np.abs(audio_data))

    def resample_audio(self, audio_data: np.ndarray, 
                      original_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        from scipy import signal
        return signal.resample(audio_data, 
                             int(len(audio_data) * target_rate / original_rate))

    def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Reduce noise in audio data."""
        # Implement noise reduction logic
        return audio_data

    def _extract_technical_terms(
        self,
        text: str,
        context: Optional[Dict]
    ) -> List[str]:
        """Extract subject-specific technical terms."""
        if not context or 'keywords' not in context:
            return []
        return [term for term in context['keywords'] 
                if term.lower() in text.lower()]

    def _generate_summary(self, text: str) -> str:
        """Generate concise summary of audio content."""
        sentences = text.split('.')
        summary_length = min(3, len(sentences))
        return '. '.join(sentences[:summary_length]) + '.'

    def detect_language(self, audio_path: str) -> str:
        """Detect the language of the audio."""
        audio_data, _ = self.load_audio(Path(audio_path))
        return self.model.detect_language(audio_data)

    def translate_to_english(self, audio_path: str) -> Dict:
        """Translate non-English audio to English."""
        return self.process_audio(audio_path, task_type="translate")

    @contextmanager
    def collect_metrics(self):
        """Context manager for collecting processing metrics."""
        import time
        start_time = time.time()
        metrics = {}
        try:
            yield metrics
        finally:
            metrics['processing_time'] = time.time() - start_time
            metrics['memory_usage'] = 0  # Implement memory usage tracking 