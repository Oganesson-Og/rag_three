"""
Monitoring Models Module
----------------------

Core monitoring models for tracking system performance, logging, and metrics
in educational RAG applications.

Key Features:
- Log level management
- Metrics configuration
- Event tracking
- System monitoring
- Performance metrics
- Query analytics
- Type validation

Technical Details:
- Dataclass models
- Enum handling
- Timestamp tracking
- Metadata support
- Metric aggregation
- Log configuration
- Type validation

Dependencies:
- dataclasses (standard library)
- enum (standard library)
- datetime (standard library)
- typing (standard library)

Example Usage:
    # Configure logging
    log_config = LogConfig(
        log_dir="/var/log/rag",
        file_name="rag.log",
        level=LogLevel.INFO
    )
    
    # Create metrics config
    metrics_config = MetricsConfig(
        enabled=True,
        collection_interval=60,
        retention_days=7
    )
    
    # Track query metrics
    query_metrics = QueryMetrics(
        timestamp=datetime.now(),
        query_text="What is quantum mechanics?",
        duration_ms=150.5
    )

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""


from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40

@dataclass
class LogConfig:
    log_dir: str
    file_name: str
    level: LogLevel = LogLevel.INFO
    max_size_mb: int = 10
    backup_count: int = 3

@dataclass
class LogEvent:
    level: LogLevel
    message: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class MetricsConfig:
    enabled: bool = True
    collection_interval: int = 60
    retention_days: int = 7

@dataclass
class QueryMetrics:
    timestamp: datetime
    query_text: str
    duration_ms: float = 0.0
    num_chunks: int = 0
    num_tokens: int = 0
    cache_hit: bool = False
    labels: Dict[str, str] = None

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    operation_name: str
    duration_ms: float 