"""
RAG (Retrieval Augmented Generation) System Module
-----------------------------------------------

Core system for document processing, embedding generation, and semantic search
with support for batch operations and asynchronous processing.

Key Features:
- Document processing and chunking
- Embedding generation
- Semantic search
- Batch processing
- Asynchronous operations
- In-memory document store
- Similarity scoring
- Metadata management

Technical Details:
- Async/await pattern implementation
- Cosine similarity calculations
- Chunking with overlap
- Vector embeddings
- Document ID generation
- Error handling and logging
- Memory management
- Batch size optimization

Dependencies:
- numpy
- asyncio
- logging
- datetime
- pathlib
- typing

Example Usage:
    # Initialize RAG system
    rag = RAGSystem(
        document_processor=DocumentProcessor(),
        embedding_generator=EmbeddingGenerator(),
        config={"chunk_size": 512}
    )
    
    # Process single document
    result = await rag.process_document(
        document="path/to/document.pdf",
        metadata={"source": "textbook"}
    )
    
    # Perform semantic search
    results = await rag.query(
        "quantum mechanics",
        top_k=5
    )
    
    # Batch process documents
    documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    results = await rag.batch_process(
        documents,
        batch_size=3
    )

Processing Pipeline:
1. Document Ingestion
    - File reading
    - Format detection
    - Preprocessing
    
2. Text Chunking
    - Semantic splitting
    - Overlap management
    - Size optimization
    
3. Embedding Generation
    - Vector creation
    - Batch processing
    - Caching
    
4. Search and Retrieval
    - Similarity calculation
    - Result ranking
    - Context assembly

Performance Considerations:
- Batch size optimization
- Memory management
- Async processing
- Caching strategies
- Vector similarity calculations

Error Handling:
- Document processing errors
- Embedding generation failures
- Search query issues
- Batch processing recovery
- Resource management

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
import logging
from datetime import datetime

# Remove the circular import
# from .embeddings.embedding_generator import EmbeddingGenerator
from .document_processing.processor import DocumentProcessor
from .models.query import Query, QueryResult, SearchMetadata
from .chunking.base import TextChunker

class RAGSystem:
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_generator: Any,  # Type hint as Any to avoid circular import
        config: Optional[Dict[str, Any]] = None
    ):
        self.document_processor = document_processor
        self.embedding_generator = embedding_generator
        self.config = config or {}
        self.chunker = TextChunker(
            chunk_size=self.config.get('chunk_size', 512),
            overlap=self.config.get('overlap', 50)
        )
        self.logger = logging.getLogger(__name__)
        self.document_store = {}  # Simple in-memory store for demo

    async def process_document(
        self, 
        document: Union[str, Path, bytes],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document and generate embeddings"""
        try:
            # Process document
            processed_doc = await self.document_processor.process(document)
            
            # Chunk the processed document
            chunks = self.chunker.chunk_text(processed_doc)
            
            # Generate embeddings for each chunk
            chunk_embeddings = []
            for chunk in chunks:
                embedding = await self.embedding_generator.generate(chunk)
                chunk_embeddings.append({
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': metadata or {}
                })
            
            # Store in document store
            doc_id = str(datetime.now().timestamp())
            self.document_store[doc_id] = {
                'chunks': chunk_embeddings,
                'metadata': metadata or {},
                'original': processed_doc
            }
            
            return {
                "doc_id": doc_id,
                "processed_document": processed_doc,
                "num_chunks": len(chunks),
                "embeddings": chunk_embeddings
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    async def query(
        self, 
        query: Union[str, Query],
        top_k: int = 5
    ) -> List[QueryResult]:
        """Execute a query and return relevant results"""
        try:
            # Convert string query to Query object if needed
            if isinstance(query, str):
                query = Query(text=query)
            
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate(query.text)
            
            # Search through document store
            results = []
            for doc_id, doc_data in self.document_store.items():
                for chunk in doc_data['chunks']:
                    similarity = self._calculate_similarity(
                        query_embedding, 
                        chunk['embedding']
                    )
                    
                    if similarity > self.config.get('similarity_threshold', 0.7):
                        results.append(
                            QueryResult(
                                text=chunk['text'],
                                score=float(similarity),
                                metadata=SearchMetadata(
                                    timestamp=datetime.now(),
                                    source=doc_id,
                                    confidence=float(similarity),
                                    relevance_score=float(similarity),
                                    additional_info=chunk['metadata']
                                ),
                                source_document=doc_data['original']
                            )
                        )
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    async def update_index(
        self, 
        documents: List[Union[str, Path, bytes]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Update the search index with new documents"""
        doc_ids = []
        metadata = metadata or [None] * len(documents)
        
        for doc, meta in zip(documents, metadata):
            result = await self.process_document(doc, meta)
            doc_ids.append(result['doc_id'])
            
        return doc_ids

    def _calculate_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between embeddings"""
        import numpy as np
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    async def batch_process(
        self,
        documents: List[Union[str, Path, bytes]],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple documents in batches"""
        results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            tasks = [self.process_document(doc) for doc in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results

    def clear_index(self) -> None:
        """Clear the document store"""
        self.document_store.clear() 