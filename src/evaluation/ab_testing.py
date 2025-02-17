"""
A/B Testing Framework Module
--------------------------

Core A/B testing framework for evaluating different RAG system components
and strategies through controlled experiments.

Key Features:
- Variant assignment and tracking
- Traffic splitting
- Metric collection
- Statistical analysis
- Result persistence
- Experiment management
- Winner determination

Technical Details:
- Consistent user assignment
- JSON-based result storage
- Statistical computations
- Configurable metrics
- Experiment versioning
- Data persistence
- Type safety

Dependencies:
- numpy>=1.24.0
- typing (standard library)
- datetime (standard library)
- json (standard library)
- pathlib (standard library)
- random (standard library)
- dataclasses (standard library)
- enum (standard library)

Example Usage:
    # Create experiment variants
    variants = [
        Variant(
            id="v1",
            name="baseline",
            config={"k": 5},
            implementation=baseline_retrieval
        ),
        Variant(
            id="v2",
            name="enhanced",
            config={"k": 5, "rerank": True},
            implementation=enhanced_retrieval
        )
    ]
    
    # Initialize framework
    framework = ABTestingFramework(
        experiment_name="retrieval_comparison",
        experiment_type=ExperimentType.RETRIEVAL_STRATEGY,
        variants=variants,
        metrics=["relevance", "latency"]
    )
    
    # Run experiment
    variant = framework.assign_variant(user_id)
    results = variant.implementation(**variant.config)
    framework.record_result(variant.id, results, user_id)

Experiment Types:
- Retrieval Strategy Testing
- Chunking Method Comparison
- Embedding Model Evaluation
- Ranking Algorithm Assessment

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Callable, Any
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import random
from dataclasses import dataclass
import uuid
from enum import Enum

class ExperimentType(Enum):
    RETRIEVAL_STRATEGY = "retrieval_strategy"
    CHUNKING_METHOD = "chunking_method"
    EMBEDDING_MODEL = "embedding_model"
    RANKING_ALGORITHM = "ranking_algorithm"

@dataclass
class Variant:
    id: str
    name: str
    config: Dict[str, Any]
    implementation: Callable

@dataclass
class ExperimentResult:
    variant_id: str
    metrics: Dict[str, float]
    timestamp: str
    user_id: str
    query_id: str

class ABTestingFramework:
    def __init__(
        self,
        experiment_name: str,
        experiment_type: ExperimentType,
        variants: List[Variant],
        metrics: List[str],
        traffic_split: Optional[List[float]] = None,
        results_path: Optional[Path] = None
    ):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.variants = variants
        self.metrics = metrics
        self.traffic_split = traffic_split or [1.0/len(variants)] * len(variants)
        self.results_path = results_path or Path("experiment_results")
        self.results_path.mkdir(exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        self._load_existing_results()

    def _load_existing_results(self) -> None:
        """Load existing experiment results."""
        result_file = self.results_path / f"{self.experiment_name}_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                self.results = [
                    ExperimentResult(**result)
                    for result in data
                ]

    def assign_variant(self, user_id: str) -> Variant:
        """Assign a variant to a user based on traffic split."""
        # Ensure consistent assignment for the same user
        random.seed(hash(user_id))
        assignment = random.random()
        
        cumulative_prob = 0
        for prob, variant in zip(self.traffic_split, self.variants):
            cumulative_prob += prob
            if assignment <= cumulative_prob:
                return variant
                
        return self.variants[-1]  # Fallback to last variant

    def record_result(
        self,
        variant_id: str,
        metrics: Dict[str, float],
        user_id: str,
        query_id: str
    ) -> None:
        """Record experiment result."""
        result = ExperimentResult(
            variant_id=variant_id,
            metrics=metrics,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            query_id=query_id
        )
        
        self.results.append(result)
        self._save_results()

    def _save_results(self) -> None:
        """Save experiment results to file."""
        result_file = self.results_path / f"{self.experiment_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(
                [vars(result) for result in self.results],
                f,
                indent=2
            )

    def get_experiment_stats(self) -> Dict[str, Any]:
        """Calculate experiment statistics."""
        stats = {}
        
        for variant in self.variants:
            variant_results = [
                r for r in self.results
                if r.variant_id == variant.id
            ]
            
            if not variant_results:
                continue
                
            variant_stats = {
                "sample_size": len(variant_results),
                "metrics": {}
            }
            
            for metric in self.metrics:
                values = [
                    r.metrics[metric]
                    for r in variant_results
                    if metric in r.metrics
                ]
                
                if values:
                    variant_stats["metrics"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "median": np.median(values),
                        "95th_percentile": np.percentile(values, 95)
                    }
                    
            stats[variant.name] = variant_stats
            
        return stats

    def get_winning_variant(self, primary_metric: str) -> Optional[str]:
        """Determine the winning variant based on primary metric."""
        stats = self.get_experiment_stats()
        
        best_score = float('-inf')
        winning_variant = None
        
        for variant_name, variant_stats in stats.items():
            if primary_metric in variant_stats["metrics"]:
                score = variant_stats["metrics"][primary_metric]["mean"]
                if score > best_score:
                    best_score = score
                    winning_variant = variant_name
                    
        return winning_variant 