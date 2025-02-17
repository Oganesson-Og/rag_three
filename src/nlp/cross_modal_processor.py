"""
Cross-Modal Processing Module
-------------------------

Enhanced cross-modal reasoning system for educational content processing.

Key Features:
- Multi-modal processing
- Text analysis
- Image processing
- Audio handling
- Embedding fusion
- Confidence scoring
- Async operations

Technical Details:
- Model management
- Tensor operations
- Embedding generation
- Score calculation
- Error handling
- Resource cleanup
- Performance tracking

Dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- PIL>=9.0.0
- numpy>=1.24.0
- librosa>=0.10.0

Example Usage:
    # Initialize processor
    processor = CrossModalProcessor()
    
    # Process multi-modal query
    result = await processor.process_multimodal_query(
        text="Explain this diagram",
        image="diagram.jpg",
        audio="explanation.wav"
    )
    
    # Access results
    embeddings = result["embeddings"]
    confidence = result["confidence"]

Supported Models:
- Text: sentence-transformers/all-mpnet-base-v2
- Vision: openai/clip-vit-base-patch32
- Audio: openai/whisper-base

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoModel, AutoTokenizer, AutoImageProcessor, WhisperModel, 
    AutoProcessor, AutoModelForCausalLM
)
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import logging
from pathlib import Path
import os
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

class CrossModalProcessor:
    """Processes text and vision inputs using Qwen2.5-VL."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize cross-modal processor."""
        self.config = config
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        try:
            # Initialize model with Apple Silicon optimizations
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="mps",
                trust_remote_code=True
            )
            
            # Initialize processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Model settings
            self.max_length = config.get('max_length', 2048)
            self.temperature = config.get('temperature', 0.7)
            self.top_p = config.get('top_p', 0.9)
            
            logger.info(f"Successfully loaded Qwen2.5-VL on MPS")
            
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise ValueError(f"Failed to initialize Qwen2.5-VL model: {str(e)}")
    
    async def process(self, text: str, image: Optional[bytes] = None) -> Dict[str, Any]:
        """Process text and optional image input."""
        try:
            messages = [{"role": "user", "content": []}]
            
            # Add image if provided
            if image is not None:
                messages[0]["content"].append({
                    "type": "image",
                    "image": image if isinstance(image, str) else Image.open(image)
                })
            
            # Add text
            messages[0]["content"].append({
                "type": "text",
                "text": text
            })
            
            # Prepare inputs
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process only images, no video
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[chat_text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to("mps")
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return {
                "response": response,
                "model_name": self.config['model_name'],
                "modalities": ["text", "image"] if image is not None else ["text"]
            }
            
        except Exception as e:
            raise ValueError(f"Processing failed: {str(e)}")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            del self.model
            if torch.mps.is_available():
                torch.mps.empty_cache()
        except:
            pass

    async def process_multimodal_query(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        audio: Optional[Union[str, Path, np.ndarray]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Process multi-modal educational query."""
        try:
            results = {}
            
            # Default modality weights
            weights = weights or {
                "text": 0.4,
                "image": 0.3,
                "audio": 0.3
            }
            
            # Process each modality
            if text:
                results["text"] = await self._process_text(text)
            
            if image:
                results["image"] = await self._process_image(image)
                
            if audio:
                results["audio"] = await self._process_audio(audio)
                
            # Combine embeddings with weights
            combined_embedding = self._combine_embeddings(results, weights)
            
            return {
                "embeddings": combined_embedding,
                "modality_scores": self._calculate_modality_scores(results),
                "confidence": self._calculate_confidence(results)
            }
            
        except Exception as e:
            self.logger.error(f"Cross-modal processing error: {str(e)}")
            raise

    async def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text input."""
        try:
            # Tokenize and encode text
            inputs = self.text_processor(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                
            return {
                "embedding": outputs.last_hidden_state.mean(dim=1).cpu().numpy(),
                "attention": outputs.attentions[-1].cpu().numpy() if outputs.attentions else None,
                "confidence": self._calculate_text_confidence(outputs)
            }
            
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            raise

    async def _process_image(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """Process image input."""
        try:
            # Load and preprocess image
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
                
            # Process image
            inputs = self.vision_processor(
                image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
                
            return {
                "embedding": outputs.last_hidden_state.mean(dim=1).cpu().numpy(),
                "attention": outputs.attentions[-1].cpu().numpy() if outputs.attentions else None,
                "confidence": self._calculate_image_confidence(outputs)
            }
            
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            raise

    async def _process_audio(
        self,
        audio: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """Process audio input."""
        try:
            # Load audio if path provided
            if isinstance(audio, (str, Path)):
                audio = self._load_audio(audio)
                
            # Process audio
            with torch.no_grad():
                result = self.audio_model.transcribe(audio)
                
            return {
                "embedding": result.embeddings,
                "transcription": result.text,
                "confidence": result.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Audio processing error: {str(e)}")
            raise

    def _combine_embeddings(
        self,
        results: Dict[str, Dict[str, Any]],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Combine embeddings from different modalities."""
        combined = None
        total_weight = 0
        
        for modality, weight in weights.items():
            if modality in results:
                embedding = results[modality]["embedding"]
                if combined is None:
                    combined = embedding * weight
                else:
                    combined += embedding * weight
                total_weight += weight
                
        return combined / total_weight if total_weight > 0 else None

    def _calculate_modality_scores(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for each modality."""
        return {
            modality: data["confidence"]
            for modality, data in results.items()
            if "confidence" in data
        }

    def _calculate_confidence(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score."""
        scores = [
            data["confidence"]
            for data in results.values()
            if "confidence" in data
        ]
        return np.mean(scores) if scores else 0.0

    def _calculate_text_confidence(self, outputs: Any) -> float:
        """Calculate confidence score for text processing."""
        attention_weights = outputs.attentions[-1] if outputs.attentions else None
        if attention_weights is not None:
            return float(attention_weights.mean().cpu().numpy())
        return 0.8  # Default confidence

    def _calculate_image_confidence(self, outputs: Any) -> float:
        """Calculate confidence score for image processing."""
        attention_weights = outputs.attentions[-1] if outputs.attentions else None
        if attention_weights is not None:
            return float(attention_weights.mean().cpu().numpy())
        return 0.7  # Default confidence

    def _load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load audio file."""
        import librosa
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            return audio
        except Exception as e:
            self.logger.error(f"Audio loading error: {str(e)}")
            raise 