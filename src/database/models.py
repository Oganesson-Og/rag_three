"""
1. Educational Data Management
# Student session tracking
# Learning progress
# Assessment history
# Educational metadata
2. Content Organization
# Original content storage (optional)
# Educational standards mapping
# Content type classification
# Qdrant vector references
3. Relationship Management
# Student-content relationships
# Session tracking
# Progress monitoring
# Assessment history

------------------
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import qdrant_client
from qdrant_client.http import models as qdrant_models
from sqlalchemy.sql import func
import numpy as np

Base = declarative_base()

class DocumentDB(Base):
    """SQLAlchemy Document model for database storage."""
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True)
    title = Column(String)
    source = Column(String)
    content_type = Column(String)
    doc_info = Column(JSON)  # Stores content and processing metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    vectors = relationship("Vector", back_populates="document", cascade="all, delete-orphan")
    
    @property
    def content(self) -> Optional[Union[str, bytes]]:
        """Get document content from doc_info."""
        if not self.doc_info:
            return None
        return self.doc_info.get("content")
    
    @content.setter
    def content(self, value: Union[str, bytes]):
        """Set document content in doc_info."""
        if not self.doc_info:
            self.doc_info = {}
        self.doc_info["content"] = value
    
    @property
    def modality(self) -> Optional[str]:
        """Get document modality from doc_info."""
        return self.doc_info.get("modality") if self.doc_info else None
    
    def add_processing_event(self, event: Dict[str, Any]):
        """Add processing event to document history."""
        if "processing_history" not in self.doc_info:
            self.doc_info["processing_history"] = []
        self.doc_info["processing_history"].append({
            **event,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}')>"

class Chunk(Base):
    """Chunk model for storing document chunks."""
    __tablename__ = 'chunks'
    
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'))
    content = Column(String)
    chunk_info = Column(JSON)  # Renamed from metadata
    embedding_data = Column(JSON)  # Renamed from embedding
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationship with document
    document = relationship("DocumentDB", back_populates="chunks")

class Vector(Base):
    """Vector model."""
    __tablename__ = 'vectors'

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'))
    vector_data = Column(JSON, nullable=False)
    vector_info = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship('DocumentDB', back_populates='vectors')

class Cache(Base):
    """Cache model."""
    __tablename__ = 'cache'

    key = Column(String, primary_key=True)
    value = Column(JSON, nullable=False)
    cache_info = Column(JSON, default=dict)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EducationalSession(Base):
    """Tracks student learning sessions."""
    __tablename__ = 'educational_sessions'
    
    id = Column(String, primary_key=True)
    student_id = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    topic = Column(String)
    status = Column(String)  # active, completed, interrupted
    session_metrics = Column(JSON)  # Renamed from metrics

class LearningProgress(Base):
    """Student progress tracking."""
    __tablename__ = 'learning_progress'
    
    id = Column(String, primary_key=True)
    student_id = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    mastery_level = Column(Float)
    completed_modules = Column(JSON)
    assessment_history = Column(JSON)
    last_interaction = Column(DateTime)

class ContentMetadata(Base):
    """Original content metadata and references."""
    __tablename__ = 'content_metadata'
    
    id = Column(String, primary_key=True)
    qdrant_id = Column(String, nullable=False)
    original_content = Column(String)
    content_type = Column(String)
    educational_info = Column(JSON)
    created_at = Column(DateTime)

class QdrantVectorStore(Base):
    """Qdrant vector store metadata."""
    __tablename__ = 'vector_store'
    
    id = Column(String, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))  # Added foreign key
    collection_name = Column(String, unique=True)
    dimension = Column(Integer)
    store_info = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationship with document
    document = relationship("DocumentDB", backref="qdrant_vectors")
    
    def __init__(self, collection_name: str, dimension: int = 1536, **kwargs):
        """Initialize vector store."""
        super().__init__(
            collection_name=collection_name,
            dimension=dimension,
            store_info=kwargs.get('store_info', {}),
            document_id=kwargs.get('document_id')
        )

    def __repr__(self):
        return f"<QdrantVector(id={self.id}, collection={self.collection_name})>"

class DatabaseConnection:
    """Database connection manager."""
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.session = None

    def connect(self):
        """Establish database connection."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        self.engine = create_engine(self.connection_string)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)

    def close(self):
        """Close database connection."""
        if self.session:
            self.session.close() 