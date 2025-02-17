"""
Curriculum Progression Management Module
------------------------------------

Comprehensive system for managing educational progression paths and tracking
student advancement through curriculum topics.

Key Features:
- Topic dependency management
- Learning path generation
- Progress tracking
- Difficulty assessment
- Subject area organization
- Prerequisites validation
- Achievement monitoring
- Resource management

Technical Details:
- Graph-based progression
- JSON configuration
- Progress persistence
- Metadata handling
- Path optimization
- Validation rules
- Resource tracking
- Performance metrics

Dependencies:
- typing (standard library)
- datetime (standard library)
- json (standard library)
- pathlib (standard library)
- dataclasses (standard library)
- enum (standard library)

Example Usage:
    # Initialize manager
    manager = CurriculumProgressionManager("curriculum.json")
    
    # Get next topics
    next_topics = manager.get_next_topics(
        user_id="student_123",
        completed_topics=["algebra_1", "geometry_1"]
    )
    
    # Update progress
    manager.update_progress(
        user_id="student_123",
        topic_id="algebra_2",
        status="completed",
        assessment_score=0.85
    )

Progression Features:
- Topic sequencing
- Prerequisite checking
- Progress reporting
- Path customization
- Resource suggestions
- Performance tracking
- Difficulty scaling

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class DifficultyLevel(Enum):
    """
    Educational content difficulty levels.
    
    Attributes:
        BEGINNER: Entry-level content
        INTERMEDIATE: Mid-level content
        ADVANCED: Advanced-level content
    """
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class SubjectArea(Enum):
    """
    Academic subject areas.
    
    Attributes:
        MATHEMATICS: Math-related topics
        SCIENCE: Science-related topics
        LANGUAGE: Language-related topics
        SOCIAL_STUDIES: Social studies topics
    """
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    LANGUAGE = "language"
    SOCIAL_STUDIES = "social_studies"

@dataclass
class TopicNode:
    """
    Educational topic node with metadata.
    
    Attributes:
        id (str): Unique topic identifier
        name (str): Topic name/title
        prerequisites (List[str]): Required prerequisite topics
        difficulty (DifficultyLevel): Topic difficulty level
        subject (SubjectArea): Academic subject area
        learning_objectives (List[str]): Learning goals
        estimated_duration (int): Estimated completion time
        resources (List[Dict]): Learning resources
    """
    id: str
    name: str
    prerequisites: List[str]
    difficulty: DifficultyLevel
    subject: SubjectArea
    learning_objectives: List[str]
    estimated_duration: int  # in minutes
    resources: List[Dict]

class CurriculumProgressionManager:
    """
    Manager for educational progression and topic sequencing.
    
    Attributes:
        curriculum_graph (Dict[str, TopicNode]): Topic dependency graph
        learning_paths (Dict[str, List[str]]): Predefined learning paths
        progress_tracking (Dict[str, Dict]): User progress data
    
    Methods:
        load_curriculum: Load curriculum from JSON
        get_next_topics: Get available next topics
        update_progress: Update user progress
        get_learning_path: Generate learning path
        get_progress_report: Generate progress report
    """
    
    def __init__(self, curriculum_path: Optional[Path] = None):
        """
        Initialize progression manager.
        
        Args:
            curriculum_path: Optional path to curriculum JSON
        """
        self.curriculum_graph: Dict[str, TopicNode] = {}
        self.learning_paths: Dict[str, List[str]] = {}
        self.progress_tracking: Dict[str, Dict] = {}
        
        if curriculum_path:
            self.load_curriculum(curriculum_path)

    def load_curriculum(self, path: Path) -> None:
        """
        Load curriculum structure from JSON file.
        
        Args:
            path: Path to curriculum JSON file
            
        Raises:
            ValueError: If curriculum loading fails
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            for topic_data in data['topics']:
                topic = TopicNode(
                    id=topic_data['id'],
                    name=topic_data['name'],
                    prerequisites=topic_data['prerequisites'],
                    difficulty=DifficultyLevel(topic_data['difficulty']),
                    subject=SubjectArea(topic_data['subject']),
                    learning_objectives=topic_data['learning_objectives'],
                    estimated_duration=topic_data['estimated_duration'],
                    resources=topic_data['resources']
                )
                self.curriculum_graph[topic.id] = topic

            self.learning_paths = data.get('learning_paths', {})
        except Exception as e:
            raise ValueError(f"Failed to load curriculum: {str(e)}")

    def get_next_topics(self, user_id: str, completed_topics: List[str]) -> List[TopicNode]:
        """
        Get next available topics based on prerequisites.
        
        Args:
            user_id: User identifier
            completed_topics: List of completed topic IDs
            
        Returns:
            List of available topic nodes
        """
        available_topics = []
        
        for topic_id, topic in self.curriculum_graph.items():
            if topic_id not in completed_topics:
                prerequisites_met = all(
                    prereq in completed_topics 
                    for prereq in topic.prerequisites
                )
                if prerequisites_met:
                    available_topics.append(topic)
                    
        return available_topics

    def update_progress(
        self,
        user_id: str,
        topic_id: str,
        status: str,
        assessment_score: Optional[float] = None
    ) -> None:
        """
        Update user's progress for a specific topic.
        
        Args:
            user_id: User identifier
            topic_id: Topic identifier
            status: Progress status
            assessment_score: Optional assessment score
        """
        if user_id not in self.progress_tracking:
            self.progress_tracking[user_id] = {}
            
        self.progress_tracking[user_id][topic_id] = {
            "status": status,
            "completion_date": datetime.utcnow().isoformat(),
            "assessment_score": assessment_score
        }

    def get_learning_path(
        self,
        user_id: str,
        subject: SubjectArea,
        difficulty: DifficultyLevel
    ) -> List[TopicNode]:
        """
        Generate personalized learning path.
        
        Args:
            user_id: User identifier
            subject: Subject area
            difficulty: Difficulty level
            
        Returns:
            List of topic nodes in recommended order
        """
        completed_topics = set(
            topic_id 
            for topic_id, data in self.progress_tracking.get(user_id, {}).items()
            if data["status"] == "completed"
        )
        
        # Filter topics by subject and difficulty
        available_topics = [
            topic for topic in self.curriculum_graph.values()
            if topic.subject == subject and topic.difficulty == difficulty
        ]
        
        # Sort topics by prerequisites
        sorted_topics = []
        while available_topics:
            for topic in available_topics[:]:
                if all(prereq in completed_topics or prereq in {t.id for t in sorted_topics}
                      for prereq in topic.prerequisites):
                    sorted_topics.append(topic)
                    available_topics.remove(topic)
                    
        return sorted_topics

    def get_progress_report(self, user_id: str) -> Dict:
        """
        Generate comprehensive progress report.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing progress statistics
        """
        if user_id not in self.progress_tracking:
            return {"error": "User not found"}
            
        completed_topics = []
        in_progress_topics = []
        available_topics = []
        
        user_progress = self.progress_tracking[user_id]
        
        for topic_id, topic in self.curriculum_graph.items():
            if topic_id in user_progress:
                if user_progress[topic_id]["status"] == "completed":
                    completed_topics.append(topic)
                else:
                    in_progress_topics.append(topic)
            elif all(prereq in [t.id for t in completed_topics] for prereq in topic.prerequisites):
                available_topics.append(topic)
                
        return {
            "completed": [t.__dict__ for t in completed_topics],
            "in_progress": [t.__dict__ for t in in_progress_topics],
            "available": [t.__dict__ for t in available_topics],
            "completion_rate": len(completed_topics) / len(self.curriculum_graph) * 100
        } 