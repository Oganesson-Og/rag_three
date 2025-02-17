"""
Database Connection Module
----------------------
"""

from typing import Dict, Optional, Any, Generator
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

class DatabaseConnection:
    """Database connection manager."""
    
    def __init__(
        self,
        connection_params: Optional[Dict[str, Any]] = None,
        pool_size: int = 5
    ) -> None:
        self.connection_params = connection_params or {}
        self.pool_size = pool_size
        self.logger = logging.getLogger(__name__)
        self._init_pool()

    def _init_pool(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = SimpleConnectionPool(
                1, self.pool_size,
                **self.connection_params,
                cursor_factory=RealDictCursor
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Pool initialization error: {str(e)}")
            raise

    @contextmanager
    def get_connection(self) -> Generator:
        """Get database connection from pool."""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self) -> Generator:
        """Get database cursor."""
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Cursor error: {str(e)}")
                raise
            finally:
                cursor.close()

    def close(self) -> None:
        """Close connection pool."""
        try:
            if hasattr(self, 'pool'):
                self.pool.closeall()
                self.logger.info("Database connection pool closed")
        except Exception as e:
            self.logger.error(f"Pool closing error: {str(e)}")
            raise 