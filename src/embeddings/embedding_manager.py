
"""
Multi-Modal Embedding Manager Module
---------------------------------

Comprehensive embedding generation system supporting text, image, and audio modalities.

Key Features:
- Multi-modal embedding generation
- Text embeddings using MPNET
- Image embeddings using ViT
- Audio embeddings using Whisper
- Batch processing support
- Embedding combination utilities
- Device-aware processing (CPU/GPU)

Technical Details:
- Uses transformer models for each modality
- Handles various input formats
- Normalized embedding outputs
- Configurable model selection
- Memory-efficient processing
- Batch operation support

Dependencies:
- transformers>=4.30.0
- torch>=2.0.0
- numpy>=1.24.0
- Pillow>=9.5.0
- librosa>=0.10.0

Example Usage:
    # Initialize manager
    manager = EmbeddingManager()
    
    # Generate embeddings for different modalities
    text_emb = manager.get_text_embedding("sample text")
    image_emb = manager.get_image_embedding(image_data)
    audio_emb = manager.get_audio_embedding(audio_data)
    
    # Batch processing
    batch_emb = manager.get_batch_embeddings(
        texts=["text1", "text2"],
        images=[img1, img2]
    )

Models Used:
- Text: sentence-transformers/all-mpnet-base-v2
- Image: google/vit-base-patch16-224
- Audio: openai/whisper-base

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Optional, Union, List
import numpy as np
import io
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel
import librosa
from transformers import WhisperProcessor, WhisperModel

class EmbeddingManager:
    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        image_model_name: str = "google/vit-base-patch16-224",
        audio_model_name: str = "openai/whisper-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Initialize text models
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        
        # Initialize image models
        self.image_processor = ViTImageProcessor.from_pretrained(image_model_name)
        self.image_model = ViTModel.from_pretrained(image_model_name).to(device)
        
        # Initialize audio models
        self.audio_processor = WhisperProcessor.from_pretrained(audio_model_name)
        self.audio_model = WhisperModel.from_pretrained(audio_model_name).to(device)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text input."""
        # Tokenize and prepare input
        inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            
        # Use [CLS] token embedding as sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]  # Return the first embedding (batch size 1)

    def get_image_embedding(self, image_data: Union[bytes, Image.Image]) -> np.ndarray:
        """Generate embeddings for image input."""
        # Convert bytes to PIL Image if necessary
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
            
        # Prepare image input
        inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            
        # Use pooled output as image embedding
        embeddings = outputs.pooler_output.cpu().numpy()
        return embeddings[0]  # Return the first embedding (batch size 1)

    def get_audio_embedding(self, audio_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """Generate embeddings for audio input."""
        # Load and preprocess audio
        if isinstance(audio_data, bytes):
            audio_array, sampling_rate = librosa.load(io.BytesIO(audio_data), sr=16000)
        else:
            audio_array = audio_data
            
        # Prepare audio input
        inputs = self.audio_processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            
        # Use mean of last hidden states as audio embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings[0]  # Return the first embedding (batch size 1)

    def get_batch_embeddings(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Union[bytes, Image.Image]]] = None,
        audio_files: Optional[List[Union[bytes, np.ndarray]]] = None
    ) -> dict:
        """Generate embeddings for batches of different modalities."""
        embeddings = {}
        
        if texts:
            embeddings['text'] = [
                self.get_text_embedding(text)
                for text in texts
            ]
            
        if images:
            embeddings['image'] = [
                self.get_image_embedding(image)
                for image in images
            ]
            
        if audio_files:
            embeddings['audio'] = [
                self.get_audio_embedding(audio)
                for audio in audio_files
            ]
            
        return embeddings

    def combine_embeddings(
        self,
        embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Combine multiple embeddings with optional weights."""
        if weights is None:
            weights = [1.0] * len(embeddings)
            
        if len(embeddings) != len(weights):
            raise ValueError("Number of embeddings must match number of weights")
            
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Combine embeddings
        combined = sum(
            emb * weight
            for emb, weight in zip(embeddings, weights)
        )
        
        # Normalize combined embedding
        return combined / np.linalg.norm(combined) 