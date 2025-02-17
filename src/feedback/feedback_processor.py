"""
Educational Feedback Processing Module
-----------------------------------

Comprehensive feedback analysis system for educational content with support for
multiple feedback types, learning outcomes, and automated improvement suggestions.

Key Features:
- Multi-dimensional feedback processing
- Learning outcome analysis
- Engagement tracking
- Automated suggestions
- Performance metrics
- Trend analysis
- Alert system

Technical Details:
- Weighted scoring system
- Time-series analysis
- Statistical computations
- JSON-based persistence
- Trend detection
- Alert thresholds
- Batch processing

Dependencies:
- numpy>=1.24.0
- pandas>=1.5.0
- sklearn>=1.0.0
- typing (standard library)
- pathlib (standard library)
- json (standard library)
- logging (standard library)
- datetime (standard library)

Example Usage:
    # Initialize processor
    processor = FeedbackProcessor(
        config_path="config/feedback.json",
        feedback_dir="data/feedback"
    )
    
    # Process new feedback
    feedback_entry = FeedbackEntry(
        feedback_id="fb123",
        user_id="user456",
        content_id="content789",
        feedback_type=FeedbackType.COMPREHENSION,
        rating=0.85,
        learning_outcome=LearningOutcome.GOOD
    )
    processor.process_feedback(feedback_entry)
    
    # Get content analysis
    analysis = processor.get_content_analysis("content789")

Feedback Types:
- Relevance (content alignment)
- Comprehension (understanding)
- Difficulty (complexity level)
- Completeness (coverage)
- Accuracy (correctness)
- Engagement (interaction)

Learning Outcomes:
- Excellent
- Good
- Fair
- Needs Improvement

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""


from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from collections import defaultdict

class FeedbackType(Enum):
    RELEVANCE = "relevance"
    COMPREHENSION = "comprehension"
    DIFFICULTY = "difficulty"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    ENGAGEMENT = "engagement"

class LearningOutcome(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    NEEDS_IMPROVEMENT = "needs_improvement"

@dataclass
class FeedbackEntry:
    """Represents a single feedback entry."""
    feedback_id: str
    user_id: str
    content_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    rating: float  # 0.0 to 1.0
    learning_outcome: Optional[LearningOutcome]
    comments: Optional[str]
    metadata: Dict[str, Any]

class FeedbackProcessor:
    """Processes and analyzes feedback for educational RAG system."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        feedback_dir: Optional[Path] = None,
        min_feedback_threshold: int = 5
    ):
        self.config = self._load_config(config_path)
        self.feedback_dir = feedback_dir or Path("data/feedback")
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_feedback_threshold = min_feedback_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize feedback storage
        self.feedback_entries: List[FeedbackEntry] = []
        self.content_feedback: Dict[str, List[FeedbackEntry]] = defaultdict(list)
        self.user_feedback: Dict[str, List[FeedbackEntry]] = defaultdict(list)
        
        # Load existing feedback
        self._load_existing_feedback()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load feedback processing configuration."""
        default_config = {
            "rating_weights": {
                "relevance": 0.3,
                "comprehension": 0.2,
                "accuracy": 0.2,
                "engagement": 0.15,
                "difficulty": 0.15
            },
            "improvement_threshold": 0.7,
            "alert_threshold": 0.4
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
                
        return default_config

    def process_feedback(
        self,
        feedback: Union[FeedbackEntry, List[FeedbackEntry]]
    ) -> None:
        """Process new feedback entries."""
        if isinstance(feedback, FeedbackEntry):
            feedback = [feedback]
            
        for entry in feedback:
            # Store feedback
            self.feedback_entries.append(entry)
            self.content_feedback[entry.content_id].append(entry)
            self.user_feedback[entry.user_id].append(entry)
            
            # Analyze feedback
            self._analyze_feedback_impact(entry)
            
            # Check for alerts
            self._check_feedback_alerts(entry)
            
            # Update content scores
            self._update_content_scores(entry.content_id)

    def _analyze_feedback_impact(self, entry: FeedbackEntry) -> None:
        """Analyze the impact of new feedback."""
        content_feedback = self.content_feedback[entry.content_id]
        
        if len(content_feedback) >= self.min_feedback_threshold:
            # Calculate metrics
            metrics = self._calculate_content_metrics(content_feedback)
            
            # Check for significant changes
            if self._is_significant_change(metrics):
                self.logger.info(
                    f"Significant feedback change detected for content {entry.content_id}"
                )
                self._trigger_content_review(entry.content_id, metrics)

    def get_content_analysis(
        self,
        content_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive content analysis based on feedback."""
        feedback = self.content_feedback.get(content_id, [])
        
        if not feedback:
            return {"error": "No feedback available"}
            
        return {
            "metrics": self._calculate_content_metrics(feedback),
            "learning_outcomes": self._analyze_learning_outcomes(feedback),
            "engagement_trends": self._analyze_engagement_trends(feedback),
            "improvement_suggestions": self._generate_improvement_suggestions(feedback)
        }

    def _calculate_content_metrics(
        self,
        feedback: List[FeedbackEntry]
    ) -> Dict[str, float]:
        """Calculate various metrics from feedback."""
        metrics = {}
        
        for feedback_type in FeedbackType:
            type_feedback = [
                f.rating for f in feedback
                if f.feedback_type == feedback_type
            ]
            
            if type_feedback:
                metrics[feedback_type.value] = {
                    "mean": np.mean(type_feedback),
                    "std": np.std(type_feedback),
                    "count": len(type_feedback)
                }
                
        # Calculate weighted average
        weights = self.config["rating_weights"]
        weighted_scores = []
        weight_sum = 0
        
        for feedback_type, weight in weights.items():
            if feedback_type in metrics:
                weighted_scores.append(
                    metrics[feedback_type]["mean"] * weight
                )
                weight_sum += weight
                
        if weighted_scores:
            metrics["overall_score"] = sum(weighted_scores) / weight_sum
            
        return metrics

    def _analyze_learning_outcomes(
        self,
        feedback: List[FeedbackEntry]
    ) -> Dict[str, Any]:
        """Analyze learning outcomes from feedback."""
        outcomes = [f.learning_outcome for f in feedback if f.learning_outcome]
        
        if not outcomes:
            return {}
            
        outcome_counts = pd.Series(outcomes).value_counts()
        
        return {
            "distribution": outcome_counts.to_dict(),
            "most_common": outcome_counts.index[0].value,
            "improvement_rate": self._calculate_improvement_rate(feedback)
        }

    def _analyze_engagement_trends(
        self,
        feedback: List[FeedbackEntry]
    ) -> Dict[str, Any]:
        """Analyze user engagement trends."""
        engagement_scores = [
            f.rating for f in feedback
            if f.feedback_type == FeedbackType.ENGAGEMENT
        ]
        
        if not engagement_scores:
            return {}
            
        df = pd.DataFrame({
            'timestamp': [f.timestamp for f in feedback],
            'engagement': engagement_scores
        })
        
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        return {
            "trend": self._calculate_trend(df['engagement']),
            "peak_engagement": df['engagement'].max(),
            "recent_engagement": df['engagement'].iloc[-5:].mean()
        }

    def _calculate_improvement_rate(
        self,
        feedback: List[FeedbackEntry]
    ) -> float:
        """Calculate the rate of improvement in learning outcomes."""
        outcomes = pd.Series([
            f.learning_outcome for f in feedback
            if f.learning_outcome
        ])
        
        if len(outcomes) < 2:
            return 0.0
            
        outcome_scores = {
            LearningOutcome.EXCELLENT: 4,
            LearningOutcome.GOOD: 3,
            LearningOutcome.FAIR: 2,
            LearningOutcome.NEEDS_IMPROVEMENT: 1
        }
        
        scores = [outcome_scores[o] for o in outcomes]
        return (scores[-1] - scores[0]) / len(scores)

    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction from time series data."""
        if len(series) < 2:
            return "insufficient_data"
            
        slope = np.polyfit(range(len(series)), series, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        return "stable"

    def _generate_improvement_suggestions(
        self,
        feedback: List[FeedbackEntry]
    ) -> List[str]:
        """Generate improvement suggestions based on feedback."""
        suggestions = []
        metrics = self._calculate_content_metrics(feedback)
        
        # Check each feedback type
        for feedback_type in FeedbackType:
            if feedback_type.value in metrics:
                score = metrics[feedback_type.value]["mean"]
                
                if score < self.config["improvement_threshold"]:
                    suggestions.extend(
                        self._get_type_specific_suggestions(feedback_type, score)
                    )
                    
        return suggestions

    def _get_type_specific_suggestions(
        self,
        feedback_type: FeedbackType,
        score: float
    ) -> List[str]:
        """Get type-specific improvement suggestions."""
        suggestions = []
        
        if feedback_type == FeedbackType.RELEVANCE:
            suggestions.extend([
                "Review content alignment with learning objectives",
                "Add more context-specific examples",
                "Improve content organization"
            ])
        elif feedback_type == FeedbackType.COMPREHENSION:
            suggestions.extend([
                "Simplify complex explanations",
                "Add more visual aids",
                "Include step-by-step breakdowns"
            ])
        elif feedback_type == FeedbackType.DIFFICULTY:
            suggestions.extend([
                "Adjust content difficulty level",
                "Add more scaffolding",
                "Include prerequisite reviews"
            ])
            
        return suggestions

    def _update_content_scores(self, content_id: str) -> None:
        """Update content scores based on feedback."""
        feedback = self.content_feedback[content_id]
        metrics = self._calculate_content_metrics(feedback)
        
        # Implement content score update logic
        # This could involve:
        # - Updating recommendation weights
        # - Adjusting content difficulty ratings
        # - Modifying content visibility

    def _check_feedback_alerts(self, entry: FeedbackEntry) -> None:
        """Check for feedback patterns that require alerts."""
        content_feedback = self.content_feedback[entry.content_id]
        metrics = self._calculate_content_metrics(content_feedback)
        
        if metrics.get("overall_score", 1.0) < self.config["alert_threshold"]:
            self.logger.warning(
                f"Low feedback score alert for content {entry.content_id}"
            )
            self._trigger_content_review(entry.content_id, metrics)

    def _trigger_content_review(
        self,
        content_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Trigger a content review based on feedback."""
        self.logger.info(
            f"Content review triggered for {content_id}\n"
            f"Metrics: {json.dumps(metrics, indent=2)}"
        )
        # Implement review triggering logic

    def save_feedback(self) -> None:
        """Save feedback data to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_file = self.feedback_dir / f"feedback_{timestamp}.json"
        
        with open(feedback_file, 'w') as f:
            json.dump(
                [vars(entry) for entry in self.feedback_entries],
                f,
                indent=2,
                default=str
            )

    def _load_existing_feedback(self) -> None:
        """Load existing feedback data."""
        for feedback_file in self.feedback_dir.glob("feedback_*.json"):
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = FeedbackEntry(**entry_data)
                        self.feedback_entries.append(entry)
                        self.content_feedback[entry.content_id].append(entry)
                        self.user_feedback[entry.user_id].append(entry)
            except Exception as e:
                self.logger.error(f"Error loading feedback file {feedback_file}: {str(e)}") 