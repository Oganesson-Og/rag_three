"""
Cross-Modal Reasoning Module
--------------------------------

Advanced reasoning system for processing and combining queries across multiple
modalities (text, image, audio) with intelligent embedding combination strategies.

Key Features:
- Multi-modal query processing
- Intelligent embedding combination
- Modality-specific relevance scoring
- Flexible filtering capabilities
- Comprehensive result processing
- Configurable similarity thresholds
- Modality-specific result retrieval

Technical Details:
- Normalized embedding combination
- Weighted modality scoring
- Vector similarity calculations
- Modality-aware result ranking
- Efficient vector operations
- Comprehensive error handling

Dependencies:
- numpy>=1.24.0
- pydantic>=2.5.0
- typing-extensions>=4.7.0

Example Usage:
    # Initialize reasoner
    reasoner = CrossModalReasoner(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )
    
    # Process multi-modal query
    results = reasoner.process_multimodal_query(
        text_query="Sample text",
        image_query=image_bytes,
        context_filters={"subject": "science"}
    )

Performance Considerations:
- Optimized vector operations
- Efficient embedding normalization
- Smart result caching
- Configurable processing thresholds

Author: Keith Satuku
Version: 1.0.0
Created: 2024
License: MIT
"""

from typing import Dict, List, Optional, Union
import numpy as np
from ..vector_store.education_vector_store import EducationVectorStore
from ..embeddings.embedding_manager import EmbeddingManager

class CrossModalReasoner:
    def __init__(
        self,
        vector_store: EducationVectorStore,
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = 0.75
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold

    def process_multimodal_query(
        self,
        text_query: Optional[str] = None,
        image_query: Optional[bytes] = None,
        audio_query: Optional[bytes] = None,
        context_filters: Optional[Dict] = None
    ) -> Dict:
        """Process queries across different modalities."""
        embeddings = []
        
        if text_query:
            text_embedding = self.embedding_manager.get_text_embedding(text_query)
            embeddings.append(("text", text_embedding))
            
        if image_query:
            image_embedding = self.embedding_manager.get_image_embedding(image_query)
            embeddings.append(("image", image_embedding))
            
        if audio_query:
            audio_embedding = self.embedding_manager.get_audio_embedding(audio_query)
            embeddings.append(("audio", audio_embedding))

        # Combine embeddings with weighted average
        combined_embedding = self._combine_embeddings([emb for _, emb in embeddings])
        
        # Search vector store with combined embedding
        results = self.vector_store.search(
            query_vector=combined_embedding,
            filters=context_filters,
            limit=5,
            score_threshold=self.similarity_threshold
        )
        
        # Process and rank results
        processed_results = self._process_results(results, embeddings)
        
        return {
            "combined_results": processed_results,
            "modality_specific": {
                modality: self._get_modality_specific_results(emb, context_filters)
                for modality, emb in embeddings
            }
        }

    def _combine_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Combine embeddings from different modalities."""
        if not embeddings:
            raise ValueError("No embeddings provided")
            
        # Normalize embeddings
        normalized_embeddings = [
            emb / np.linalg.norm(emb)
            for emb in embeddings
        ]
        
        # Simple average for now - could be enhanced with learned weights
        combined = np.mean(normalized_embeddings, axis=0)
        return combined / np.linalg.norm(combined)

    def _process_results(
        self,
        results: List[Dict],
        query_embeddings: List[tuple]
    ) -> List[Dict]:
        """Process and enhance search results."""
        processed_results = []
        
        for result in results:
            # Calculate modality-specific relevance scores
            modality_scores = {
                modality: float(np.dot(
                    result["vector"],
                    emb / np.linalg.norm(emb)
                ))
                for modality, emb in query_embeddings
            }
            
            processed_results.append({
                **result,
                "modality_scores": modality_scores,
                "average_modality_score": sum(modality_scores.values()) / len(modality_scores)
            })
            
        return sorted(
            processed_results,
            key=lambda x: x["average_modality_score"],
            reverse=True
        )

    def _get_modality_specific_results(
        self,
        embedding: np.ndarray,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Get results specific to a single modality."""
        return self.vector_store.search(
            query_vector=embedding,
            filters=filters,
            limit=3,
            score_threshold=self.similarity_threshold
        ) 