"""
Educational Content Retrieval Module
--------------------------------

Advanced retrieval system specifically designed for educational content, implementing
sophisticated ranking and filtering strategies for academic material retrieval.

Key Features:
- Education-aware ranking algorithms
- Syllabus-based content filtering
- Complexity level matching
- Intent-based retrieval optimization
- Educational metadata utilization
- Multi-stage ranking pipeline
- Cross-reference handling
- Prerequisite tracking

Technical Details:
- Hybrid retrieval combining vector and semantic search
- Custom scoring functions for educational relevance
- Efficient metadata filtering with Qdrant
- Query result caching system
- Automatic query expansion for educational terms
- Learning path awareness
- Topic relationship mapping

Dependencies:
- numpy>=1.24.0
- qdrant-client>=1.7.0
- fastapi>=0.109.0
- pydantic>=2.5.0
- scipy>=1.11.0
- redis>=5.0.1

Example Usage:
    # Basic retrieval
    retriever = EducationRetriever(
        subject='physics',
        level='a-level'
    )
    results = retriever.retrieve(query)
    
    # Advanced retrieval with filters
    results = retriever.retrieve(
        query,
        filter_metadata={
            'difficulty': 'advanced',
            'topic': 'quantum_mechanics'
        },
        include_prerequisites=True
    )
    
    # Curriculum-aware retrieval
    results = retriever.retrieve(
        query,
        user_context={
            'current_topic': 'quantum_mechanics',
            'completed_topics': ['classical_mechanics']
        }
    )

Performance Considerations:
- Result caching with Redis
- Batch processing support
- Qdrant query optimization
- Connection pooling
- Asynchronous operations
- Index optimization

Author: Keith Satuku
Version: 2.5.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional, Union, Any
import numpy as np
from numpy.typing import NDArray
import logging
from datetime import datetime
from ..query.education_processor import EducationQueryProcessor
from ..storage.vector_store.qdrant_store import QdrantVectorStore
from ..embeddings.enhanced_generator import EnhancedEmbeddingGenerator
from ..config.domain_config import get_domain_config

# Type aliases
Vector = Union[List[float], NDArray[np.float32]]
RetrievalResult = Dict[str, Any]

class EducationRetriever:
    """Enhanced education content retriever using Qdrant with advanced educational features"""
    
    def __init__(
        self,
        subject: str,
        level: str,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.subject = subject
        self.level = level
        self.config = config or {}
        
        # Initialize vector store
        self.vector_store = vector_store or QdrantVectorStore(
            collection_name=f"education_{subject}_{level}"
        )
        
        # Initialize embedding generator
        self.embedding_generator = EnhancedEmbeddingGenerator(
            model_name=embedding_model
        )
        
        # Initialize query processor
        self.query_processor = EducationQueryProcessor(
            subject=subject,
            level=level
        )

    async def add_content(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add educational content to vector store with enhanced metadata"""
        embedding = self.embedding_generator.generate_embedding(content)
        
        enhanced_metadata = {
            **(metadata or {}),
            "subject": self.subject,
            "grade_level": self.level,
            "content_type": "text",
            "content_length": len(content),
            "embedding_model": self.embedding_generator.model_name,
            "prerequisites": metadata.get("prerequisites", []),
            "difficulty": metadata.get("difficulty", "intermediate"),
            "topics": metadata.get("topics", [])
        }
        
        return await self.vector_store.add_vector(
            vector=embedding,
            metadata=enhanced_metadata
        )

    async def retrieve(
        self,
        query: str,
        filter_metadata: Optional[Dict] = None,
        top_k: int = 5,
        include_prerequisites: bool = False,
        user_context: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve relevant educational content with advanced features"""
        # Process query with educational context
        processed_query = await self.query_processor.process_query(
            query=query,
            user_context=user_context
        )
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(processed_query['processed_query'])
        
        # Enhance filters with educational context
        search_filters = {
            "subject": self.subject,
            "grade_level": self.level,
            **(filter_metadata or {})
        }
        
        # Add difficulty adaptation if user context provided
        if user_context and 'user_level' in user_context:
            search_filters['difficulty'] = self._adapt_difficulty(user_context['user_level'])
        
        # Retrieve results
        results = await self.vector_store.search(
            query_vector=query_embedding,
            k=top_k,
            filters=search_filters
        )
        
        # Rerank results using educational criteria
        reranked_results = self._rerank_results(processed_query, results)
        
        # Include prerequisites if requested
        if include_prerequisites:
            reranked_results = await self._include_prerequisites(reranked_results)
        
        return [self._format_result(result) for result in reranked_results]

    def _rerank_results(
        self,
        query: Dict,
        results: List[Dict]
    ) -> List[Dict]:
        """Rerank results using educational criteria"""
        reranked = []
        
        for result in results:
            # Calculate educational relevance score
            edu_score = self._calculate_educational_relevance(
                query=query,
                result=result
            )
            
            # Calculate curriculum alignment score
            curriculum_score = self._calculate_curriculum_alignment(
                result=result,
                query_topics=query.get('topics', [])
            )
            
            # Combine scores
            final_score = (
                0.5 * result['score'] +  # Vector similarity
                0.3 * edu_score +        # Educational relevance
                0.2 * curriculum_score   # Curriculum alignment
            )
            
            reranked.append({
                **result,
                'score': final_score,
                'educational_score': edu_score,
                'curriculum_score': curriculum_score
            })
        
        # Sort by final score
        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked
    
    def _calculate_educational_relevance(
        self,
        query: Dict,
        result: Dict
    ) -> float:
        """Calculate educational relevance score"""
        score = 0.0
        
        # Check syllabus topic match
        query_topics = set(query.get('topics', []))
        result_topics = set(result['metadata'].get('topics', []))
        topic_overlap = len(query_topics & result_topics)
        score += 0.4 * (topic_overlap / max(len(query_topics), 1))
        
        # Check complexity level match
        query_level = query.get('complexity_level', 'intermediate')
        result_level = result['metadata'].get('complexity', 'intermediate')
        score += 0.3 * (1.0 if query_level == result_level else 0.5)
        
        # Check educational intent match
        query_intent = query.get('query_intent', 'general')
        result_type = result['metadata'].get('content_type', 'general')
        score += 0.3 * (1.0 if query_intent == result_type else 0.5)
        
        return score
    
    def _calculate_curriculum_alignment(
        self,
        result: Dict,
        query_topics: List[str]
    ) -> float:
        """Calculate curriculum alignment score"""
        # Implementation depends on curriculum structure
        # This is a simplified version
        result_topics = set(result['metadata'].get('topics', []))
        query_topics = set(query_topics)
        
        if not query_topics:
            return 0.5  # Neutral score if no topics provided
            
        overlap = len(result_topics & query_topics)
        return overlap / len(query_topics)
    
    def _adapt_difficulty(self, user_level: str) -> str:
        """Adapt difficulty based on user level"""
        # Implement difficulty adaptation logic
        difficulty_mapping = {
            'beginner': ['basic', 'intermediate'],
            'intermediate': ['intermediate', 'advanced'],
            'advanced': ['advanced', 'expert']
        }
        return difficulty_mapping.get(user_level, ['intermediate'])[0]
    
    async def _include_prerequisites(self, results: List[Dict]) -> List[Dict]:
        """Include prerequisites in results"""
        enhanced_results = []
        
        for result in results:
            prerequisites = result['metadata'].get('prerequisites', [])
            if prerequisites:
                prereq_results = await self.vector_store.search(
                    filters={"topics": {"$in": prerequisites}},
                    k=2
                )
                result['prerequisites'] = prereq_results
            enhanced_results.append(result)
            
        return enhanced_results
    
    def _format_result(self, result: Dict) -> Dict:
        """Format retrieval result for presentation"""
        return {
            'content': result['content'],
            'metadata': {
                'subject': result['metadata'].get('subject', self.subject),
                'level': result['metadata'].get('level', self.level),
                'topics': result['metadata'].get('topics', []),
                'complexity': result['metadata'].get('complexity', 'intermediate'),
                'content_type': result['metadata'].get('content_type', 'general'),
                'prerequisites': result['metadata'].get('prerequisites', [])
            },
            'relevance_score': result['score'],
            'educational_score': result.get('educational_score', 0.0),
            'curriculum_score': result.get('curriculum_score', 0.0)
        }

    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        **kwargs
    ) -> List[List[Dict]]:
        """Batch retrieve relevant educational content"""
        results = []
        for query in queries:
            results.append(await self.retrieve(query, top_k=top_k, **kwargs))
        return results 