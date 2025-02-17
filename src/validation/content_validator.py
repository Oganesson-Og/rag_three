"""
Content Validator Module
----------------------

Comprehensive validation system for educational content quality and standards.

Key Features:
- Content quality validation
- Structure validation
- Reference checking
- Educational standards
- Technical validation
- Metadata validation
- Readability analysis
- Fact checking

Technical Details:
- NLP-based validation
- Transformer models
- Spacy integration
- Pattern matching
- Metric calculation
- Error handling
- Configurable rules

Dependencies:
- spacy>=3.5.0
- transformers>=4.30.0
- numpy>=1.24.0
- bs4>=4.12.0
- validators>=0.20.0
- typing-extensions>=4.7.0

Example Usage:
    validator = ContentValidator(
        config_path="path/to/config.json",
        validation_level=ValidationLevel.STRICT,
        spacy_model="en_core_web_lg"
    )
    
    # Validate content
    results = validator.validate_content(
        content="Educational content text...",
        content_type="article",
        metadata={
            "author": "John Doe",
            "grade_level": "high_school",
            "subject": "physics"
        }
    )
    
    # Check validation results
    for result in results:
        if not result.is_valid:
            print(f"Validation failed: {result.category}")
            print(f"Issues: {result.issues}")
            print(f"Suggestions: {result.suggestions}")

Performance Considerations:
- Efficient NLP operations
- Model loading optimization
- Memory management
- Batch processing
- Cache utilization
- Error handling
- Validation order

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import re
from enum import Enum
import logging
import json
from pathlib import Path
import numpy as np
from transformers import pipeline
import spacy
from bs4 import BeautifulSoup
import validators
from datetime import datetime

class ValidationLevel(Enum):
    """Validation level enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"

class ValidationCategory(Enum):
    """Validation category enumeration."""
    CONTENT = "content"
    STRUCTURE = "structure"
    REFERENCES = "references"
    EDUCATIONAL = "educational"
    TECHNICAL = "technical"

@dataclass
class ValidationResult:
    """Represents the result of content validation.
    
    Attributes:
        is_valid: Whether the content passed validation
        category: Category of validation
        level: Level of validation performed
        score: Validation score (0.0 to 1.0)
        issues: List of validation issues found
        suggestions: List of improvement suggestions
        metadata: Additional validation metadata
    """
    is_valid: bool
    category: ValidationCategory
    level: ValidationLevel
    score: float
    issues: List[Dict]
    suggestions: List[str]
    metadata: Dict

class ContentValidator:
    """Comprehensive validator for educational content."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        spacy_model: str = "en_core_web_lg",
        device: str = "cuda"
    ):
        self.logger = logging.getLogger(__name__)
        self.validation_level = validation_level
        self.config = self._load_config(config_path)
        
        # Initialize NLP components
        self.nlp = spacy.load(spacy_model)
        self.fact_checker = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=device
        )
        
        # Load validation rules
        self.rules = self._load_validation_rules()
        
        # Initialize specialized validators
        self.math_validator = self._initialize_math_validator()
        self.reference_validator = self._initialize_reference_validator()
        self.educational_validator = self._initialize_educational_validator()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load validator configuration."""
        default_config = {
            "min_content_length": 100,
            "max_content_length": 10000,
            "readability_threshold": 0.7,
            "fact_check_threshold": 0.8,
            "reference_required": True,
            "allowed_domains": [
                "edu", "org", "gov"
            ]
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
                
        return default_config

    def _load_validation_rules(self) -> Dict:
        """Load validation rules for different categories."""
        return {
            ValidationCategory.CONTENT: {
                "min_length": self.config["min_content_length"],
                "max_length": self.config["max_content_length"],
                "required_sections": [
                    "introduction",
                    "content",
                    "summary"
                ],
                "banned_words": set([
                    "obviously",
                    "clearly",
                    "everyone knows"
                ])
            },
            ValidationCategory.EDUCATIONAL: {
                "required_elements": [
                    "learning_objectives",
                    "prerequisites",
                    "difficulty_level"
                ],
                "max_complexity_score": 0.8,
                "min_examples": 2
            }
        }

    def validate_content(
        self,
        content: Union[str, Dict],
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> List[ValidationResult]:
        """Validate educational content comprehensively."""
        results = []
        
        # Content validation
        content_result = self._validate_content_quality(content, content_type)
        results.append(content_result)
        
        # Structure validation
        structure_result = self._validate_structure(content, content_type)
        results.append(structure_result)
        
        # Educational validation
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            edu_result = self._validate_educational_aspects(content, metadata)
            results.append(edu_result)
            
        # Reference validation
        if self.validation_level == ValidationLevel.STRICT:
            ref_result = self._validate_references(content)
            results.append(ref_result)
            
        # Technical validation
        tech_result = self._validate_technical_aspects(content, content_type)
        results.append(tech_result)
        
        return results

    def _validate_content_quality(
        self,
        content: Union[str, Dict],
        content_type: str
    ) -> ValidationResult:
        """Validate content quality and accuracy."""
        issues = []
        suggestions = []
        
        # Convert dict content to string if necessary
        text_content = content if isinstance(content, str) else json.dumps(content)
        
        # Check content length
        if len(text_content) < self.rules[ValidationCategory.CONTENT]["min_length"]:
            issues.append({
                "type": "content_length",
                "severity": "high",
                "message": "Content is too short"
            })
            
        # Check readability
        readability_score = self._calculate_readability(text_content)
        if readability_score < self.config["readability_threshold"]:
            issues.append({
                "type": "readability",
                "severity": "medium",
                "message": "Content readability is below threshold"
            })
            suggestions.append("Consider simplifying the language")
            
        # Check for banned words
        banned_words = self._find_banned_words(text_content)
        if banned_words:
            issues.append({
                "type": "banned_words",
                "severity": "low",
                "message": f"Found banned words: {', '.join(banned_words)}"
            })
            
        # Fact checking for educational claims
        if self.validation_level == ValidationLevel.STRICT:
            fact_check_issues = self._fact_check_content(text_content)
            issues.extend(fact_check_issues)
            
        return ValidationResult(
            is_valid=len(issues) == 0,
            category=ValidationCategory.CONTENT,
            level=self.validation_level,
            score=self._calculate_validation_score(issues),
            issues=issues,
            suggestions=suggestions,
            metadata={"readability_score": readability_score}
        )

    def _validate_structure(
        self,
        content: Union[str, Dict],
        content_type: str
    ) -> ValidationResult:
        """Validate content structure."""
        issues = []
        suggestions = []
        
        # Check required sections
        missing_sections = self._check_required_sections(content)
        if missing_sections:
            issues.append({
                "type": "missing_sections",
                "severity": "high",
                "message": f"Missing required sections: {', '.join(missing_sections)}"
            })
            
        # Validate hierarchical structure
        structure_issues = self._validate_hierarchy(content)
        issues.extend(structure_issues)
        
        # Check formatting consistency
        formatting_issues = self._check_formatting(content)
        issues.extend(formatting_issues)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            category=ValidationCategory.STRUCTURE,
            level=self.validation_level,
            score=self._calculate_validation_score(issues),
            issues=issues,
            suggestions=suggestions,
            metadata={}
        )

    def _validate_educational_aspects(
        self,
        content: Union[str, Dict],
        metadata: Optional[Dict]
    ) -> ValidationResult:
        """Validate educational aspects of content."""
        issues = []
        suggestions = []
        
        # Check learning objectives
        if metadata and "learning_objectives" not in metadata:
            issues.append({
                "type": "missing_objectives",
                "severity": "high",
                "message": "Learning objectives not specified"
            })
            
        # Validate difficulty level
        if metadata and "difficulty_level" in metadata:
            if not self._is_valid_difficulty(metadata["difficulty_level"]):
                issues.append({
                    "type": "invalid_difficulty",
                    "severity": "medium",
                    "message": "Invalid difficulty level specification"
                })
                
        # Check examples and exercises
        example_count = self._count_examples(content)
        if example_count < self.rules[ValidationCategory.EDUCATIONAL]["min_examples"]:
            issues.append({
                "type": "insufficient_examples",
                "severity": "medium",
                "message": f"Insufficient examples (found {example_count})"
            })
            
        return ValidationResult(
            is_valid=len(issues) == 0,
            category=ValidationCategory.EDUCATIONAL,
            level=self.validation_level,
            score=self._calculate_validation_score(issues),
            issues=issues,
            suggestions=suggestions,
            metadata={"example_count": example_count}
        )

    def _validate_references(self, content: Union[str, Dict]) -> ValidationResult:
        """Validate references and citations."""
        issues = []
        suggestions = []
        
        # Extract references
        references = self._extract_references(content)
        
        # Validate each reference
        for ref in references:
            if not self._is_valid_reference(ref):
                issues.append({
                    "type": "invalid_reference",
                    "severity": "medium",
                    "message": f"Invalid reference format: {ref}"
                })
                
        # Check reference freshness
        outdated_refs = self._check_reference_dates(references)
        if outdated_refs:
            issues.append({
                "type": "outdated_references",
                "severity": "low",
                "message": f"Found {len(outdated_refs)} outdated references"
            })
            
        return ValidationResult(
            is_valid=len(issues) == 0,
            category=ValidationCategory.REFERENCES,
            level=self.validation_level,
            score=self._calculate_validation_score(issues),
            issues=issues,
            suggestions=suggestions,
            metadata={"reference_count": len(references)}
        )

    def _validate_technical_aspects(
        self,
        content: Union[str, Dict],
        content_type: str
    ) -> ValidationResult:
        """Validate technical aspects of content."""
        issues = []
        suggestions = []
        
        # Validate mathematical content
        if content_type in ["math", "science"]:
            math_issues = self.math_validator.validate(content)
            issues.extend(math_issues)
            
        # Check code snippets if present
        code_issues = self._validate_code_snippets(content)
        issues.extend(code_issues)
        
        # Validate media elements
        media_issues = self._validate_media_elements(content)
        issues.extend(media_issues)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            category=ValidationCategory.TECHNICAL,
            level=self.validation_level,
            score=self._calculate_validation_score(issues),
            issues=issues,
            suggestions=suggestions,
            metadata={}
        )

    def _calculate_validation_score(self, issues: List[Dict]) -> float:
        """Calculate overall validation score."""
        if not issues:
            return 1.0
            
        severity_weights = {
            "high": 0.5,
            "medium": 0.3,
            "low": 0.2
        }
        
        total_weight = sum(
            severity_weights[issue["severity"]]
            for issue in issues
        )
        
        return max(0.0, 1.0 - total_weight)

    def _calculate_readability(self, text: str) -> float:
        """Calculate text readability score."""
        doc = self.nlp(text)
        
        # Simple readability metric based on sentence length and complexity
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
        avg_word_length = np.mean([len(token.text) for token in doc])
        
        # Normalize scores
        sentence_score = max(0, 1 - (avg_sentence_length / 40))
        word_score = max(0, 1 - (avg_word_length / 10))
        
        return (sentence_score + word_score) / 2

    def _fact_check_content(self, text: str) -> List[Dict]:
        """Check factual accuracy of educational content."""
        issues = []
        sentences = [sent.text for sent in self.nlp(text).sents]
        
        for sentence in sentences:
            # Use fact-checking model
            result = self.fact_checker(sentence)
            if result[0]["score"] < self.config["fact_check_threshold"]:
                issues.append({
                    "type": "potential_inaccuracy",
                    "severity": "high",
                    "message": f"Potential inaccuracy: {sentence}"
                })
                
        return issues

    def generate_report(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            "overall_score": np.mean([r.score for r in validation_results]),
            "validation_level": self.validation_level.value,
            "results": [
                {
                    "category": r.category.value,
                    "score": r.score,
                    "issues": r.issues,
                    "suggestions": r.suggestions
                }
                for r in validation_results
            ],
            "summary": self._generate_summary(validation_results),
            "timestamp": datetime.now().isoformat()
        } 