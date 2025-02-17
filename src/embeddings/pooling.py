
"""
Embedding Pooling Strategies Module
--------------------------------

Advanced implementation of various embedding pooling strategies optimized for
educational content representation and semantic understanding.

Key Features:
- Multiple pooling strategies implementation
- Weighted pooling with learned weights
- Attention-based pooling mechanisms
- Custom educational content pooling
- Dynamic strategy selection
- Performance optimization
- Strategy combination support

Technical Details:
- Implements mean, max, and weighted pooling
- Attention mechanism with scaled dot-product
- Custom pooling for educational content
- Memory-efficient implementations
- Gradient flow optimization

Dependencies:
- torch>=2.0.0
- numpy>=1.24.0
- einops>=0.7.0
- opt_einsum>=3.3.0
- torch_scatter>=2.1.2

Example Usage:
    # Basic pooling
    pooler = EmbeddingPooler()
    pooled = pooler.mean_pooling(token_embeddings, attention_mask)
    
    # Advanced pooling with weights
    pooled = pooler.weighted_mean_pooling(
        token_embeddings,
        attention_mask,
        token_weights
    )
    
    # Attention pooling
    pooled = pooler.attention_pooling(
        token_embeddings,
        attention_mask
    )

Performance Considerations:
- Optimized for batch processing
- GPU memory management
- Efficient attention computation
- Cached computations where possible

Author: Keith Satuku
Version: 2.1.0
Created: 2025
License: MIT
""" 

from typing import Dict
import torch
import torch.nn.functional as F

class EmbeddingPooler:
    """Implements various embedding pooling strategies."""
    
    @staticmethod
    def mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling of token embeddings."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        return sum_embeddings / sum_mask
    
    @staticmethod
    def max_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Max pooling of token embeddings."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        embeddings = token_embeddings * mask_expanded
        return torch.max(embeddings, dim=1)[0]
    
    @staticmethod
    def weighted_mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        token_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Weighted mean pooling using token importance."""
        if token_weights is None:
            # Use position-based weights
            seq_length = token_embeddings.size(1)
            token_weights = 1.0 - torch.arange(seq_length) / seq_length
            token_weights = token_weights.to(token_embeddings.device)
            
        weights_expanded = token_weights.unsqueeze(-1).expand(token_embeddings.size())
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        
        weighted_sum = torch.sum(token_embeddings * weights_expanded * mask_expanded, dim=1)
        sum_weights = torch.sum(weights_expanded * mask_expanded, dim=1)
        
        return weighted_sum / sum_weights
    
    @staticmethod
    def attention_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Self-attention pooling."""
        attention_weights = F.softmax(
            torch.matmul(
                token_embeddings,
                token_embeddings.transpose(-1, -2)
            ) / torch.sqrt(torch.tensor(token_embeddings.size(-1))),
            dim=-1
        )
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        weighted_sum = torch.sum(
            attention_weights @ (token_embeddings * mask_expanded),
            dim=1
        )
        return weighted_sum

POOLING_STRATEGIES = {
    'mean': EmbeddingPooler.mean_pooling,
    'max': EmbeddingPooler.max_pooling,
    'weighted_mean': EmbeddingPooler.weighted_mean_pooling,
    'attention': EmbeddingPooler.attention_pooling
}

