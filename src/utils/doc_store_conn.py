"""
Document Store Connection Module
----------------------------

Abstract base class and implementations for document store connections.

Key Features:
- Multiple store support
- Vector operations
- CRUD operations
- Search capabilities
- Metadata management
- Health checks
- Index management

Technical Details:
- Abstract interfaces
- Connection pooling
- Query optimization
- Vector operations
- Index management
- Error handling
- Type validation

Dependencies:
- numpy>=1.24.0
- typing-extensions>=4.7.0
- datetime
- logging

Example Usage:
    # Initialize connection
    conn = PostgresDocStore(
        host="localhost",
        port=5432,
        database="documents"
    )
    
    # Search documents
    results = conn.search(
        match_expr=MatchTextExpr("content", "query"),
        order_by=OrderByExpr("timestamp", ascending=False),
        limit=10
    )
    
    # Insert documents
    doc_ids = conn.insert([{
        "title": "Document",
        "content": "Text content",
        "metadata": {"type": "article"}
    }])
    
    # Update documents
    conn.update(doc_ids, {"status": "published"})
    
    # Delete documents
    conn.delete(doc_ids)

Performance Considerations:
- Connection pooling
- Query optimization
- Batch operations
- Index utilization
- Cache management
- Error handling
- Type validation

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Any, TypeVar, Generic
from typing_extensions import TypeAlias
import numpy as np
from numpy.typing import NDArray
import logging
from datetime import datetime

DEFAULT_MATCH_VECTOR_TOPN = 10
DEFAULT_MATCH_SPARSE_TOPN = 10
VEC = Union[List, np.ndarray]

# Type aliases
Vector = Union[List[float], NDArray[np.float32]]
QueryVector = Union[Vector, Dict[str, float]]
OrderDirection = Union[str, bool]
ExtraOptions = Optional[Dict[str, Any]]

@dataclass
class SparseVectorData:
    """Sparse vector representation."""
    indices: List[int]
    values: List[Union[float, int]]
    size: Optional[int] = None

    def to_dict(self) -> Dict[str, float]:
        if self.values is None:
            raise ValueError("values cannot be None")
        return {str(i): v for i, v in zip(self.indices, self.values)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SparseVectorData':
        indices = [int(k) for k in d.keys()]
        values = list(d.values())
        return cls(indices=indices, values=values)


class MatchTextExpr:
    """Text matching expression."""
    def __init__(self, field: str, query: str):
        self.field = field
        self.query = query


# First define the base class
class MatchTensorExpr(ABC):
    """Base class for tensor matching expressions."""
    
    def __init__(
        self,
        field_name: str,
        tensor: Vector,
        extra_option: ExtraOptions = None
    ) -> None:
        self.field_name = field_name
        self.tensor = tensor
        self.extra_option = extra_option or {}
        self.logger = logging.getLogger(__name__)


# Then define the derived class
class MatchDenseExpr(MatchTensorExpr):
    """Dense vector matching expression."""
    def __init__(
        self,
        field_name: str,
        tensor: Vector,
        extra_option: ExtraOptions = None,
        top_k: int = 10
    ) -> None:
        super().__init__(field_name, tensor, extra_option)
        self.top_k = top_k


class MatchSparseExpr(ABC):
    """Sparse vector matching expression."""
    def __init__(
        self,
        field: str,
        sparse_data: SparseVectorData,
        top_k: int = 10
    ):
        self.field = field
        self.sparse_data = sparse_data
        self.top_k = top_k


class FusionExpr:
    """Expression for fusing multiple search results."""
    
    def __init__(
        self,
        match_exprs: List[MatchTensorExpr],
        weights: Optional[List[float]] = None
    ) -> None:
        self.match_exprs = match_exprs
        self.weights = weights or [1.0] * len(match_exprs)
        self.logger = logging.getLogger(__name__)


class OrderByExpr:
    """Expression for ordering search results."""
    
    def __init__(
        self,
        field_name: str,
        ascending: bool = True
    ) -> None:
        self.field_name = field_name
        self.ascending = ascending
        self.logger = logging.getLogger(__name__)


# Combined match expression type
MatchExpr: TypeAlias = Union[
    MatchTextExpr,
    MatchDenseExpr,
    MatchSparseExpr,
    FusionExpr
]

class DocStoreConnection(ABC):
    """Abstract base class for document store connections."""
    
    @abstractmethod
    def search(
        self,
        match_expr: MatchExpr,
        order_by: Optional[OrderByExpr] = None,
        offset: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents using match expression."""
        pass

    @abstractmethod
    def insert(
        self,
        docs: List[Dict[str, Any]],
        knowledgebase_id: Optional[str] = None
    ) -> List[str]:
        """Insert documents into store."""
        pass

    @abstractmethod
    def update(
        self,
        doc_ids: List[str],
        updates: Dict[str, Any]
    ) -> int:
        """Update documents by ID."""
        pass

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def get_by_ids(
        self,
        doc_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get documents by ID."""
        pass

    def store_vector(
        self,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict] = None
    ) -> str:
        """Store vector in document store."""
        raise NotImplementedError("Not implemented")
        
    def get_vector(
        self,
        id: str
    ) -> Optional[Union[List[float], np.ndarray]]:
        """Retrieve vector from document store."""
        raise NotImplementedError("Not implemented")
        
    def search_vectors(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        filter_expr: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors."""
        raise NotImplementedError("Not implemented")
        
    def delete_vector(self, id: str) -> bool:
        """Delete vector from store."""
        raise NotImplementedError("Not implemented")
        
    def update_metadata(
        self,
        id: str,
        metadata: Dict
    ) -> bool:
        """Update vector metadata."""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def dbType(self) -> str:
        """
        Return the type of the database.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def health(self) -> dict:
        """
        Return the health status of the database.
        """
        raise NotImplementedError("Not implemented")

    """
    Table operations
    """

    @abstractmethod
    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        """
        Create an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        """
        Delete an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:
        """
        Check if an index with given name exists
        """
        raise NotImplementedError("Not implemented")

    """
    CRUD operations
    """

    @abstractmethod
    def search(
        self,
        select_fields: List[str],
        highlight_fields: List[str],
        condition: Dict[str, Any],
        match_exprs: List[MatchExpr],
        order_by: OrderByExpr,
        offset: int,
        limit: int,
        index_names: Union[str, List[str]],
        knowledgebase_ids: List[str],
        agg_fields: Optional[List[str]] = None,
        rank_feature: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search with given conditions."""
        raise NotImplementedError

    @abstractmethod
    def get(self, chunkId: str, indexName: str, knowledgebaseIds: List[str]) -> Optional[Dict]:
        """Get document by chunk ID."""
        pass

    @abstractmethod
    def insert(self, rows: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        """
        Update or insert a bulk of rows
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        """
        Update rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        """
        Delete rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")

    """
    Helper functions for search result
    """

    @abstractmethod
    def getTotal(self, res):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def getChunkIds(self, res):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def getAggregation(self, res, fieldnm: str):
        raise NotImplementedError("Not implemented")

    """
    SQL
    """
    @abstractmethod
    def sql(self, sql: str, fetch_size: int, format: str) -> Dict[str, Any]:
        """Run SQL query."""
        raise NotImplementedError