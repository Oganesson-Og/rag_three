"""
Prompt Engineering Module
--------------------------------

Advanced prompt management system with dynamic template generation,
context awareness, and educational content optimization.

Key Features:
- Dynamic template management
- Context-aware prompt generation
- Template versioning and validation
- Performance tracking
- Educational content optimization
- Metadata management
- Template caching

Technical Details:
- Template-based prompt generation
- Variable validation system
- Usage tracking and analytics
- Metadata-driven template selection
- Performance optimization features
- Error handling and validation

Dependencies:
- string (standard library)
- datetime (standard library)
- logging>=2.0.0
- pydantic>=2.5.0

Example Usage:
    # Initialize prompt generator
    generator = PromptGenerator(config={})
    
    # Generate search prompt
    search_prompt = await generator.generate_search_prompt(
        query="Sample query",
        context={"subject": "math"}
    )
    
    # Add custom template
    generator.add_template(
        name="custom_template",
        template="Custom ${variable} template",
        required_vars=["variable"]
    )

Performance Considerations:
- Template caching
- Efficient variable substitution
- Optimized validation checks
- Metadata indexing

Author: Keith Satuku
Version: 1.0.0
Created: 2024
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from string import Template
import json

from .models import Chunk, Document
from src.utils.cache import AdvancedCache

logger = logging.getLogger(__name__)

class PromptTemplate(Template):
    """Enhanced template with metadata and validation."""
    
    def __init__(
        self,
        template: str,
        metadata: Optional[Dict[str, Any]] = None,
        required_vars: Optional[List[str]] = None
    ):
        super().__init__(template)
        self.metadata = metadata or {}
        self.required_vars = required_vars or []
        self.last_used = datetime.now()
        self.usage_count = 0
    
    def safe_substitute(self, *args, **kwargs) -> str:
        """Enhanced substitution with validation."""
        missing_vars = [var for var in self.required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        self.last_used = datetime.now()
        self.usage_count += 1
        
        return super().safe_substitute(*args, **kwargs)

class PromptGenerator:
    """
    Advanced prompt generation system with context awareness and optimization.
    
    Features:
    - Dynamic template management
    - Context-aware prompt generation
    - Educational content optimization
    - Performance tracking
    - Template versioning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = AdvancedCache()
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load and initialize prompt templates."""
        templates = {
            # Search templates
            "search_base": PromptTemplate(
                template="""
                Given the following query, identify key concepts and relevant information:
                
                Query: ${query}
                
                Subject Area: ${subject}
                Educational Level: ${level}
                Prior Context: ${context}
                
                Focus on identifying:
                - Core concepts
                - Related topics
                - Key terminology
                - Educational objectives
                """,
                required_vars=["query", "subject", "level"]
            ),
            
            # Response templates
            "response_educational": PromptTemplate(
                template="""
                Generate an educational response using the following information:
                
                Query: ${query}
                
                Retrieved Context:
                ${chunks}
                
                Educational Parameters:
                - Subject: ${subject}
                - Level: ${level}
                - Learning Objectives: ${objectives}
                
                Response Guidelines:
                1. Start with key concepts
                2. Provide clear explanations
                3. Include relevant examples
                4. Address common misconceptions
                5. Conclude with summary points
                
                Additional Context: ${context}
                """,
                required_vars=["query", "chunks", "subject", "level"]
            ),
            
            # Chunk integration templates
            "chunk_integration": PromptTemplate(
                template="""
                Synthesize the following chunks into a coherent response:
                
                ${chunks}
                
                Key Points to Address:
                - Main concepts from each chunk
                - Relationships between chunks
                - Supporting evidence
                - Practical applications
                
                Integration Guidelines:
                1. Maintain logical flow
                2. Highlight connections
                3. Resolve contradictions
                4. Ensure completeness
                """,
                required_vars=["chunks"]
            )
        }
        
        # Load custom templates from config
        custom_templates = self.config.get('custom_templates', {})
        for name, template_data in custom_templates.items():
            templates[name] = PromptTemplate(
                template=template_data['template'],
                metadata=template_data.get('metadata'),
                required_vars=template_data.get('required_vars', [])
            )
        
        return templates
    
    async def generate_search_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate optimized search prompt."""
        context = context or {}
        
        try:
            template = self.templates["search_base"]
            return template.safe_substitute(
                query=query,
                subject=context.get('subject', 'General'),
                level=context.get('level', 'Intermediate'),
                context=context.get('prior_context', 'None')
            )
        except Exception as e:
            logger.error(f"Search prompt generation failed: {str(e)}")
            # Fallback to basic prompt
            return f"Query: {query}\nContext: {json.dumps(context)}"
    
    async def generate_response_prompt(
        self,
        query: str,
        chunks: List[Chunk],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate context-aware response prompt."""
        context = context or {}
        
        try:
            # Format chunks
            formatted_chunks = self._format_chunks(chunks)
            
            template = self.templates["response_educational"]
            return template.safe_substitute(
                query=query,
                chunks=formatted_chunks,
                subject=context.get('subject', 'General'),
                level=context.get('level', 'Intermediate'),
                objectives=context.get('objectives', 'Understanding of the topic'),
                context=context.get('additional_context', '')
            )
        except Exception as e:
            logger.error(f"Response prompt generation failed: {str(e)}")
            # Fallback to basic prompt
            return f"Query: {query}\nChunks: {formatted_chunks}"
    
    def _format_chunks(self, chunks: List[Chunk]) -> str:
        """Format chunks for prompt inclusion."""
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            formatted_chunks.append(f"Chunk {i}:\n{chunk.text}\n")
        return "\n".join(formatted_chunks)
    
    async def optimize_prompt(
        self,
        prompt: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimize prompt based on performance metrics."""
        # TODO: Implement prompt optimization based on metrics
        return prompt
    
    def add_template(
        self,
        name: str,
        template: str,
        metadata: Optional[Dict[str, Any]] = None,
        required_vars: Optional[List[str]] = None
    ):
        """Add new prompt template."""
        self.templates[name] = PromptTemplate(
            template=template,
            metadata=metadata,
            required_vars=required_vars
        )
    
    async def get_template_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get template usage statistics."""
        return {
            name: {
                "last_used": template.last_used,
                "usage_count": template.usage_count,
                "metadata": template.metadata
            }
            for name, template in self.templates.items()
        } 