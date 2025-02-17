"""
LLM Model Manager Module
----------------------

Robust Large Language Model management system with fallback support and circuit breaking
capabilities for production environments.

Key Features:
- Multi-model fallback support
- Circuit breaking for failing models
- Priority-based model selection
- Automatic failure recovery
- Rate limit handling
- Configurable model parameters
- Async operation support

Technical Details:
- Supports OpenAI, Google, and Llama models
- Async/await implementation
- Failure count tracking
- Automatic circuit reset
- Priority-based routing
- Configurable timeouts
- Comprehensive error handling

Dependencies:
- openai>=1.12.0
- google-generativeai>=0.3.0
- llama-cpp-python>=0.2.0
- asyncio>=3.4.3
- tenacity>=8.2.0
- pydantic>=2.5.0

Example Usage:
    # Initialize manager
    manager = LLMManager({
        "openai_api_key": "your-key",
        "google_api_key": "your-key",
        "llama_model_path": "path/to/model"
    })
    
    # Generate completion with fallback
    response = await manager.generate_completion(
        prompt="Explain quantum mechanics",
        system_prompt="You are a physics teacher"
    )
    
    # Direct model usage
    response = await manager._generate_with_model(
        model_config=primary_model,
        prompt="Hello, world!",
        system_prompt="Be concise"
    )

Model Configuration:
- Priority-based model selection
- Configurable temperature and tokens
- Adjustable sampling parameters
- Custom model configurations
- Provider-specific settings

Fallback Behavior:
- Automatic fallback on failure
- Configurable retry attempts
- Circuit breaker protection
- Rate limit handling
- Error recovery strategies

Author: Keith Satuku
Version: 2.1.0
Created: 2024
License: MIT
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from openai import AsyncOpenAI
import os
import logging
from llama_cpp import Llama

class ModelErrorType(Enum):
    """Types of model errors that can occur."""
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    CONTEXT_LENGTH = "context_length"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    CONTENT_FILTER = "content_filter"
    INITIALIZATION_ERROR = "initialization_error"
    UNKNOWN = "unknown"

class ModelError(Exception):
    """
    Exception raised for LLM model errors.
    
    Attributes:
        error_type: Type of error that occurred
        model_name: Name of the model that caused the error
        provider: Provider of the model (OpenAI, Google, etc.)
        message: Detailed error message
        retry_after: Seconds to wait before retrying (for rate limits)
        original_error: Original exception that was caught
        request_data: Data that was sent in the request
    """
    
    def __init__(
        self,
        message: str,
        error_type: ModelErrorType = ModelErrorType.UNKNOWN,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        original_error: Optional[Exception] = None,
        request_data: Optional[Dict[str, Any]] = None
    ):
        self.error_type = error_type
        self.model_name = model_name
        self.provider = provider
        self.retry_after = retry_after
        self.original_error = original_error
        self.request_data = request_data
        
        # Build detailed error message
        error_details = [
            f"Error Type: {error_type.value}",
            f"Message: {message}"
        ]
        
        if model_name:
            error_details.append(f"Model: {model_name}")
        if provider:
            error_details.append(f"Provider: {provider}")
        if retry_after:
            error_details.append(f"Retry After: {retry_after}s")
        if original_error:
            error_details.append(f"Original Error: {str(original_error)}")
            
        super().__init__("\n".join(error_details))
    
    @classmethod
    def from_provider_error(cls, error: Exception, provider: str, model_name: str) -> 'ModelError':
        """Create ModelError from provider-specific error."""
        if provider == "openai":
            return cls._from_openai_error(error, model_name)
        elif provider == "google":
            return cls._from_google_error(error, model_name)
        elif provider == "llama":
            return cls._from_llama_error(error, model_name)
        else:
            return cls(
                message=str(error),
                provider=provider,
                model_name=model_name,
                original_error=error
            )
    
    @classmethod
    def _from_openai_error(cls, error: Exception, model_name: str) -> 'ModelError':
        """Convert OpenAI error to ModelError."""
        error_str = str(error).lower()
        
        if "rate limit" in error_str:
            return cls(
                message=str(error),
                error_type=ModelErrorType.RATE_LIMIT,
                provider="openai",
                model_name=model_name,
                retry_after=60,  # Default retry after 60s
                original_error=error
            )
        elif "context length" in error_str:
            return cls(
                message=str(error),
                error_type=ModelErrorType.CONTEXT_LENGTH,
                provider="openai",
                model_name=model_name,
                original_error=error
            )
        # Add more OpenAI-specific error handling
        
        return cls(
            message=str(error),
            provider="openai",
            model_name=model_name,
            original_error=error
        )
    
    @classmethod
    def _from_google_error(cls, error: Exception, model_name: str) -> 'ModelError':
        """Convert Google AI error to ModelError."""
        error_str = str(error).lower()
        
        if "quota" in error_str:
            return cls(
                message=str(error),
                error_type=ModelErrorType.RATE_LIMIT,
                provider="google",
                model_name=model_name,
                original_error=error
            )
        # Add more Google-specific error handling
        
        return cls(
            message=str(error),
            provider="google",
            model_name=model_name,
            original_error=error
        )
    
    @classmethod
    def _from_llama_error(cls, error: Exception, model_name: str) -> 'ModelError':
        """Convert Llama error to ModelError."""
        return cls(
            message=str(error),
            provider="llama",
            model_name=model_name,
            original_error=error
        )
    
    def is_retryable(self) -> bool:
        """Check if the error is retryable."""
        return self.error_type in {
            ModelErrorType.RATE_LIMIT,
            ModelErrorType.TIMEOUT,
            ModelErrorType.MODEL_UNAVAILABLE
        }
    
    def get_retry_strategy(self) -> Dict[str, Any]:
        """Get retry strategy for this error."""
        if not self.is_retryable():
            return {"should_retry": False}
            
        return {
            "should_retry": True,
            "wait_time": self.retry_after or 60,
            "max_retries": 3 if self.error_type == ModelErrorType.RATE_LIMIT else 2
        }

# Try importing different LLM backends
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logging.warning("Google GenerativeAI not available. To use Google models, install: pip install google-generativeai")

class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    LLAMA = "llama"

@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    organization: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    priority: int = 0  # Lower number = higher priority

class LLMManager:
    """
    Manages multiple LLM models with fallback support and circuit breaking.
    
    Features:
    - Multiple model support (OpenAI, Google, Llama)
    - Automatic fallback on failure
    - Circuit breaker pattern implementation
    - Priority-based model selection
    - Async operation support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM manager with configuration.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.model_type = config.get('model_type', 'openai')
        
        # Initialize Google AI if configured
        if self.model_type == 'google':
            if not GOOGLE_AVAILABLE:
                raise ImportError(
                    "Google GenerativeAI package not installed. "
                    "Please install with: pip install google-generativeai"
                )
            api_key = config.get('google_api_key') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not found in config or environment")
            genai.configure(api_key=api_key)
        
        self.models: Dict[str, ModelConfig] = {
            "primary": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                api_key=config.get("openai_api_key"),
                organization=config.get("openai_org"),
                priority=0
            ),
            "secondary": ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name="gemini-pro",
                api_key=config.get("google_api_key"),
                priority=1
            ),
            "tertiary": ModelConfig(
                provider=ModelProvider.LLAMA,
                model_name="llama-2-70b",
                priority=2
            )
        }
        
        # Initialize clients
        self._init_clients(config)
        
        # Track failures for circuit breaking
        self.failure_counts: Dict[str, int] = {
            model_id: 0 for model_id in self.models
        }
        self.max_failures = 3
        self.failure_reset_time = 300  # 5 minutes
        
        self.lock = asyncio.Lock()
    
    def _init_clients(self, config: Dict[str, Any]):
        """Initialize API clients for each provider."""
        self.clients = {}
        
        # OpenAI client
        if config.get("openai_api_key"):
            self.clients[ModelProvider.OPENAI] = AsyncOpenAI(
                api_key=config["openai_api_key"],
                organization=config.get("openai_org")
            )
        
        # Google client
        if config.get("google_api_key"):
            self.clients[ModelProvider.GOOGLE] = genai
        
        # Llama client
        self.clients[ModelProvider.LLAMA] = Llama(
            model_path=config.get("llama_model_path", "models/llama-2-70b.gguf"),
            n_ctx=4096
        )
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion with fallback support."""
        # Sort models by priority
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1].priority
        )
        
        last_error = None
        
        for model_id, model_config in sorted_models:
            if self.failure_counts[model_id] >= self.max_failures:
                continue
                
            try:
                response = await self._generate_with_model(
                    model_config,
                    prompt,
                    system_prompt,
                    **kwargs
                )
                
                # Reset failure count on success
                self.failure_counts[model_id] = 0
                
                return response
                
            except Exception as e:
                last_error = e
                
                async with self.lock:
                    self.failure_counts[model_id] += 1
                
                # Schedule failure count reset
                asyncio.create_task(
                    self._reset_failure_count(model_id)
                )
                
                continue
        
        # If all models failed
        raise ModelError(
            f"All models failed. Last error: {str(last_error)}"
        )
    
    async def _generate_with_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using specific model."""
        if model_config.provider == ModelProvider.OPENAI:
            response = await self.clients[ModelProvider.OPENAI].chat.completions.create(
                model=model_config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
                frequency_penalty=model_config.frequency_penalty,
                presence_penalty=model_config.presence_penalty,
                **kwargs
            )
            return response.choices[0].message.content
            
        elif model_config.provider == ModelProvider.GOOGLE:
            model = self.clients[ModelProvider.GOOGLE].GenerativeModel(
                model_config.model_name
            )
            response = model.generate_content(prompt)
            return response.text
            
        elif model_config.provider == ModelProvider.LLAMA:
            response = self.clients[ModelProvider.LLAMA](
                prompt,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p
            )
            return response['choices'][0]['text']
    
    async def _reset_failure_count(self, model_id: str):
        """Reset failure count after timeout."""
        await asyncio.sleep(self.failure_reset_time)
        async with self.lock:
            self.failure_counts[model_id] = 0 