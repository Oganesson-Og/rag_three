"""
Educational RAG Feedback Module
-----------------------------

PostgreSQL-based feedback system for tracking, analyzing, and improving RAG performance
through continuous monitoring and evaluation.

Key Features:
- Feedback collection and storage
- Performance analytics
- Trend analysis
- Query performance tracking
- Statistical aggregation
- Real-time monitoring

Technical Details:
- PostgreSQL database integration
- JSON metadata storage
- Indexed query optimization
- Statistical aggregation functions
- Time-series analysis
- Performance benchmarking

Dependencies:
- psycopg2>=2.9.0
- python-dotenv>=0.19.0
- dataclasses>=0.6
- typing>=3.7.4

Example Usage:
    # Initialize feedback system
    feedback_system = FeedbackLoop()
    
    # Record feedback
    feedback_system.record_feedback(
        query_id="query123",
        result_id="result456",
        feedback_type="user_rating",
        score=0.85,
        metadata={"modality": "text-image"}
    )
    
    # Get performance stats
    stats = feedback_system.get_feedback_stats(
        feedback_type="user_rating",
        time_window_days=7
    )

Database Schema:
- query_id: Unique identifier for the query
- result_id: Associated result identifier
- feedback_type: Category of feedback
- score: Numerical evaluation (0-1)
- metadata: JSONB storage for additional data
- timestamps: Creation and update tracking

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import DictCursor
import os
from dotenv import load_dotenv

@dataclass
class FeedbackEntry:
    query_id: str
    result_id: str
    feedback_type: str
    score: float
    metadata: Dict[str, Any]
    timestamp: datetime

class FeedbackLoop:
    def __init__(self):
        load_dotenv()
        self.db_params = {
            'dbname': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'host': os.getenv('POSTGRES_SERVER'),
            'port': os.getenv('POSTGRES_PORT')
        }
        self._init_db()
        
    def _init_db(self) -> None:
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                # Create feedback table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rag_feedback (
                        id SERIAL PRIMARY KEY,
                        query_id VARCHAR(255) NOT NULL,
                        result_id VARCHAR(255) NOT NULL,
                        feedback_type VARCHAR(50) NOT NULL,
                        score FLOAT NOT NULL,
                        metadata JSONB NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Create indexes for better query performance
                    CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON rag_feedback(query_id);
                    CREATE INDEX IF NOT EXISTS idx_feedback_type ON rag_feedback(feedback_type);
                    CREATE INDEX IF NOT EXISTS idx_feedback_score ON rag_feedback(score);
                """)
                conn.commit()
            
    def record_feedback(
        self,
        query_id: str,
        result_id: str,
        feedback_type: str,
        score: float,
        metadata: Dict[str, Any]
    ) -> None:
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_feedback 
                    (query_id, result_id, feedback_type, score, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (query_id, result_id, feedback_type, score, json.dumps(metadata))
                )
                conn.commit()
            
    def get_feedback_stats(
        self,
        feedback_type: Optional[str] = None,
        time_window_days: Optional[int] = None
    ) -> Dict[str, Any]:
        query = """
            SELECT 
                feedback_type,
                AVG(score) as avg_score,
                COUNT(*) as count,
                MIN(score) as min_score,
                MAX(score) as max_score,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as median_score
            FROM rag_feedback
        """
        params = []
        conditions = []
        
        if feedback_type:
            conditions.append("feedback_type = %s")
            params.append(feedback_type)
            
        if time_window_days:
            conditions.append("timestamp >= NOW() - INTERVAL %s DAY")
            params.append(time_window_days)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " GROUP BY feedback_type"
        
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                
        return {
            row['feedback_type']: {
                'avg_score': float(row['avg_score']),
                'median_score': float(row['median_score']),
                'min_score': float(row['min_score']),
                'max_score': float(row['max_score']),
                'count': int(row['count'])
            }
            for row in results
        }
        
    def get_low_performing_queries(
        self,
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT 
                query_id,
                AVG(score) as avg_score,
                COUNT(*) as feedback_count,
                jsonb_agg(metadata) as metadata_samples
            FROM rag_feedback
            GROUP BY query_id
            HAVING AVG(score) < %s
            ORDER BY avg_score ASC
            LIMIT %s
        """
        
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (threshold, limit))
                results = cur.fetchall()
                
        return [
            {
                'query_id': row['query_id'],
                'avg_score': float(row['avg_score']),
                'feedback_count': int(row['feedback_count']),
                'metadata_samples': row['metadata_samples']
            }
            for row in results
        ]
    
    def get_feedback_trends(
        self,
        days: int = 30,
        feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get daily trends of feedback scores"""
        query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(score) as avg_score,
                COUNT(*) as count
            FROM rag_feedback
            WHERE timestamp >= NOW() - INTERVAL %s DAY
        """
        params = [days]
        
        if feedback_type:
            query += " AND feedback_type = %s"
            params.append(feedback_type)
            
        query += """
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp)
        """
        
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                
        return [
            {
                'date': row['date'].isoformat(),
                'avg_score': float(row['avg_score']),
                'count': int(row['count'])
            }
            for row in results
        ]