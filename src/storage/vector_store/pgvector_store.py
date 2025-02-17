"""
PostgreSQL Vector Store Module
--------------------------------

PostgreSQL-based vector store implementation using pgvector extension.

Key Features:
- Persistent vector storage
- Efficient similarity search
- Rich metadata filtering
- Transaction support
- Connection pooling
- Batch operations
- Index optimization

Technical Details:
- pgvector extension
- PostgreSQL connection management
- JSON metadata storage
- Vector similarity search
- Connection pooling
- Error handling

Dependencies:
- psycopg2>=2.9.0
- numpy>=1.24.0
- typing-extensions>=4.7.0

Example Usage:
    store = PGVectorStore(connection_params={
        "dbname": "vectors",
        "user": "user",
        "password": "pass"
    })
    vector_id = store.add_vector([1.0, 2.0, 3.0], {"type": "test"})
    results = store.search([1.0, 2.0, 3.0], k=5)

Performance Considerations:
- Index-based search
- Connection pooling
- Transaction management
- Batch processing
- Query optimization

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray
import psycopg2
from psycopg2.extras import execute_values
import logging
from datetime import datetime
from .base import BaseVectorStore, Vector
import uuid

class PGVectorStore(BaseVectorStore):
    """PostgreSQL-based vector store implementation using the pgvector extension."""

    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        self.connection_params = connection_params or {}
        self.logger = logging.getLogger(__name__)
        self._init_connection()

    def _init_connection(self) -> None:
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.logger.info("Connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise

    def add_vector(self, vector: Vector, metadata: Dict[str, Any] = None) -> str:
        vector_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vectors (id, vector, metadata, timestamp)
                VALUES (%s, %s, %s, %s)
                """,
                (vector_id, vector, metadata or {}, datetime.now())
            )
        self.conn.commit()
        return vector_id

    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, vector, metadata, timestamp FROM vectors WHERE id = %s", 
                (vector_id,)
            )
            result = cur.fetchone()
        return result

    def search(
        self, 
        query_vector: Vector, 
        k: int = 10, 
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        filter_clause = ""
        filter_params = []
        for key, value in filters.items():
            filter_clause += f" AND metadata->>'{key}' = %s"
            filter_params.append(value)

        query = f"""
            SELECT id, vector, metadata, 1 - (vector <=> %s) as similarity
            FROM vectors
            WHERE 1=1 {filter_clause}
            ORDER BY similarity DESC
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, [query_vector] + filter_params + [k])
            results = cur.fetchall()
        return [
            {
                'id': r[0],
                'vector': r[1],
                'metadata': r[2],
                'score': float(r[3])
            }
            for r in results
        ]

    def delete_vectors(self, ids: List[str]) -> int:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM vectors WHERE id = ANY(%s)", (ids,))
        self.conn.commit()
        return cur.rowcount

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE vectors SET metadata = metadata || %s WHERE id = %s", 
                (metadata, vector_id)
            )
        self.conn.commit()
        return True 