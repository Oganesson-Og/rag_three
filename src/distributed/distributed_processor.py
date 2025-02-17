"""
Distributed Processing Module
--------------------------

Distributed task processing system for managing parallel educational content
processing across multiple worker nodes.

Key Features:
- Task distribution
- Priority-based processing
- Worker management
- Task dependency handling
- Resource monitoring
- Error recovery
- Load balancing
- State persistence

Technical Details:
- Ray distributed framework
- Priority queue system
- Async processing
- Worker pooling
- Task scheduling
- Resource tracking
- Error handling
- Performance monitoring

Dependencies:
- ray>=2.5.0
- typing (standard library)
- datetime (standard library)
- logging (standard library)
- pathlib (standard library)
- queue (standard library)
- threading (standard library)
- dataclasses (standard library)
- numpy>=1.24.0
- enum (standard library)
- asyncio>=3.4.3
- json (standard library)

Example Usage:
    # Initialize processor
    processor = DistributedProcessor(
        num_workers=4,
        config_path="config/distributed.json"
    )
    
    # Submit task
    task_id = await processor.submit_task(
        content="Educational content",
        task_type="embedding",
        priority=ProcessingPriority.HIGH
    )
    
    # Get result
    result = await processor.get_result(task_id)

Processing Features:
- Priority scheduling
- Task dependencies
- Resource allocation
- Worker monitoring
- Error recovery
- Result caching
- Performance metrics
- Load distribution

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Any, Callable
import ray
from datetime import datetime
import logging
from pathlib import Path
import queue
import threading
from dataclasses import dataclass
import numpy as np
from enum import Enum
import asyncio
import json
import time

class ProcessingPriority(Enum):
    """
    Task processing priority levels.
    
    Attributes:
        LOW: Background processing
        MEDIUM: Standard processing
        HIGH: Priority processing
        CRITICAL: Immediate processing
    """
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class ProcessingTask:
    """
    Represents a processing task in the distributed system.
    
    Attributes:
        task_id (str): Unique task identifier
        task_type (str): Type of processing task
        content (Any): Content to process
        priority (ProcessingPriority): Task priority level
        metadata (Dict): Task metadata
        created_at (datetime): Task creation timestamp
        dependencies (List[str]): Required task dependencies
        timeout (int): Task timeout in seconds
    """
    task_id: str
    task_type: str
    content: Any
    priority: ProcessingPriority
    metadata: Dict
    created_at: datetime
    dependencies: List[str] = None
    timeout: int = 3600  # 1 hour default

@ray.remote
class WorkerNode:
    """
    Remote worker for distributed processing.
    
    Attributes:
        worker_id (str): Unique worker identifier
        processor_configs (Dict): Processor configurations
        logger (logging.Logger): Logger instance
        processors (Dict): Processing components
        status (str): Worker status
        current_task (ProcessingTask): Current processing task
    
    Methods:
        process_task: Process assigned task
        get_status: Get worker status
    """
    
    def __init__(
        self,
        worker_id: str,
        processor_configs: Dict
    ):
        """
        Initialize worker node.
        
        Args:
            worker_id: Worker identifier
            processor_configs: Processor configurations
        """
        self.worker_id = worker_id
        self.processor_configs = processor_configs
        self.logger = logging.getLogger(f"worker_{worker_id}")
        self.processors = self._initialize_processors()
        self.status = "idle"
        self.current_task = None
        
    def _initialize_processors(self) -> Dict:
        """Initialize processing components."""
        return {
            "embedding": ray.get_actor("embedding_processor"),
            "chunking": ray.get_actor("chunking_processor"),
            "math": ray.get_actor("math_processor"),
            "vector_store": ray.get_actor("vector_store")
        }
        
    async def process_task(self, task: ProcessingTask) -> Dict:
        """
        Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            ValueError: If processor not found
        """
        self.status = "processing"
        self.current_task = task
        
        try:
            # Select appropriate processor
            processor = self.processors.get(task.task_type)
            if not processor:
                raise ValueError(f"No processor found for task type: {task.task_type}")
                
            # Process content
            result = await processor.process.remote(
                content=task.content,
                metadata=task.metadata
            )
            
            return {
                "task_id": task.task_id,
                "status": "completed",
                "result": result,
                "worker_id": self.worker_id,
                "processing_time": time.time() - task.created_at.timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Task processing error: {str(e)}")
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": str(e),
                "worker_id": self.worker_id
            }
        finally:
            self.status = "idle"
            self.current_task = None
            
    def get_status(self) -> Dict:
        """
        Get worker status.
        
        Returns:
            Dictionary containing worker status information
        """
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "current_task": self.current_task,
            "processor_configs": self.processor_configs
        }

class DistributedProcessor:
    """Manages distributed processing of educational content."""
    
    def __init__(
        self,
        num_workers: int = 4,
        config_path: Optional[Path] = None,
        redis_address: Optional[str] = None
    ):
        # Initialize Ray
        ray.init(
            address=redis_address,
            ignore_reinit_error=True,
            logging_level=logging.INFO
        )
        
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.task_queue = queue.PriorityQueue()
        self.results_cache = {}
        self.active_tasks = {}
        self.workers = self._initialize_workers(num_workers)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_workers)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration settings."""
        default_config = {
            "max_retries": 3,
            "task_timeout": 3600,
            "batch_size": 10,
            "priority_weights": {
                "CRITICAL": 0,
                "HIGH": 1,
                "MEDIUM": 2,
                "LOW": 3
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
                
        return default_config

    def _initialize_workers(self, num_workers: int) -> List[WorkerNode]:
        """Initialize worker nodes."""
        workers = []
        for i in range(num_workers):
            worker = WorkerNode.remote(
                worker_id=f"worker_{i}",
                processor_configs=self.config.get("processor_configs", {})
            )
            workers.append(worker)
        return workers

    async def submit_task(
        self,
        content: Any,
        task_type: str,
        priority: ProcessingPriority = ProcessingPriority.MEDIUM,
        metadata: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Submit a task for processing."""
        task = ProcessingTask(
            task_id=f"task_{time.time()}_{task_type}",
            task_type=task_type,
            content=content,
            priority=priority,
            metadata=metadata or {},
            created_at=datetime.now(),
            dependencies=dependencies or []
        )
        
        # Add to priority queue
        self.task_queue.put(
            (priority.value, task)
        )
        
        self.logger.info(f"Task submitted: {task.task_id}")
        return task.task_id

    async def process_batch(
        self,
        contents: List[Any],
        task_type: str,
        priority: ProcessingPriority = ProcessingPriority.MEDIUM,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """Submit a batch of tasks for processing."""
        task_ids = []
        for content in contents:
            task_id = await self.submit_task(
                content=content,
                task_type=task_type,
                priority=priority,
                metadata=metadata
            )
            task_ids.append(task_id)
        return task_ids

    async def get_result(
        self,
        task_id: str,
        wait: bool = True,
        timeout: Optional[int] = None
    ) -> Optional[Dict]:
        """Get task result."""
        start_time = time.time()
        
        while True:
            if task_id in self.results_cache:
                return self.results_cache[task_id]
                
            if not wait:
                return None
                
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
                
            await asyncio.sleep(0.1)

    def _monitor_workers(self) -> None:
        """Monitor worker status and manage task distribution."""
        while True:
            try:
                # Check for completed tasks
                for worker in self.workers:
                    status = ray.get(worker.get_status.remote())
                    
                    if status["status"] == "idle":
                        # Assign new task if available
                        if not self.task_queue.empty():
                            _, task = self.task_queue.get()
                            
                            # Check dependencies
                            if self._are_dependencies_met(task):
                                ray.get(worker.process_task.remote(task))
                                
                # Clean up old results
                self._cleanup_old_results()
                
            except Exception as e:
                self.logger.error(f"Monitor error: {str(e)}")
                
            time.sleep(1)

    def _are_dependencies_met(self, task: ProcessingTask) -> bool:
        """Check if task dependencies are met."""
        if not task.dependencies:
            return True
            
        return all(
            dep in self.results_cache and
            self.results_cache[dep]["status"] == "completed"
            for dep in task.dependencies
        )

    def _cleanup_old_results(self, max_age: int = 3600) -> None:
        """Clean up old results from cache."""
        current_time = time.time()
        
        to_remove = [
            task_id for task_id, result in self.results_cache.items()
            if current_time - result.get("timestamp", 0) > max_age
        ]
        
        for task_id in to_remove:
            del self.results_cache[task_id]

    async def shutdown(self) -> None:
        """Shutdown the distributed processor."""
        # Wait for remaining tasks
        while not self.task_queue.empty():
            await asyncio.sleep(1)
            
        # Shutdown workers
        for worker in self.workers:
            ray.kill(worker)
            
        # Shutdown Ray
        ray.shutdown()
        
        self.logger.info("Distributed processor shutdown complete") 