"""
LLM client with support for multiple providers and retry policies
"""
import asyncio
import logging
import time
from typing import AsyncIterator, Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

# Import provider implementations
from .providers import (
    BaseLLMProvider,
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider
)

logger = logging.getLogger(__name__)


# ============== Enums and Configuration ==============

class RetryStrategy(Enum):
    """Retry strategies for LLM calls"""
    CONSTANT_DELAY = "constant_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    GOOGLE = "google"  # Alias for Gemini
    ANTHROPIC = "anthropic"
    CLAUDE = "claude"  # Alias for Anthropic


@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 0.3  # seconds
    multiplier: float = 1.5
    max_delay: float = 10.0
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        if self.strategy == RetryStrategy.CONSTANT_DELAY:
            return self.initial_delay
        else:  # EXPONENTIAL_BACKOFF
            delay = self.initial_delay * (self.multiplier ** attempt)
            return min(delay, self.max_delay)


# ============== Unified LLM Client ==============

class LLMClient:
    """
    Unified LLM client with support for multiple providers
    """
    
    PROVIDER_MAP = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.GEMINI: GeminiProvider,
        LLMProvider.GOOGLE: GeminiProvider,  # Alias
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.CLAUDE: AnthropicProvider,  # Alias
    }
    
    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        enable_circuit_breaker: bool = True
    ):
        # Convert string to enum if needed
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Supported: {', '.join([p.value for p in LLMProvider])}"
                )
        
        self.provider_type = provider
        self.model = model
        self.retry_policy = retry_policy or RetryPolicy()
        
        # Initialize the appropriate provider
        provider_class = self.PROVIDER_MAP.get(provider)
        if not provider_class:
            raise ValueError(f"Provider {provider} not implemented")
        
        self.provider = provider_class(api_key=api_key, model=model)
        
        # Initialize circuit breaker if enabled
        self.circuit_breaker: Optional[CircuitBreaker] = None
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                reset_timeout=60.0
            )
        
        # For backward compatibility
        self.client = self.provider.client
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Any:
        """
        Complete a chat with retry logic and circuit breaker
        """
        # Wrap with circuit breaker if enabled
        if self.circuit_breaker:
            return await self.circuit_breaker.call(
                self._complete_with_retry,
                messages, temperature, max_tokens, tools, stream
            )
        else:
            return await self._complete_with_retry(
                messages, temperature, max_tokens, tools, stream
            )
    
    async def _complete_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Any:
        """
        Internal method: Complete with retry logic
        """
        last_error = None
        
        for attempt in range(self.retry_policy.max_retries):
            try:
                if stream:
                    # Return the generator directly
                    return await self._stream_with_retry(
                        messages, temperature, max_tokens, tools, attempt
                    )
                else:
                    return await self.provider.complete(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        stream=False
                    )
            
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.retry_policy.max_retries - 1:
                    delay = self.retry_policy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
        
        raise last_error or Exception("Failed after retries")
    
    async def _stream_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]],
        attempt: int
    ) -> AsyncIterator:
        """Wrap streaming with retry capability"""
        try:
            async for chunk in await self.provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True
            ):
                yield chunk
        except Exception as e:
            if attempt < self.retry_policy.max_retries - 1:
                logger.warning(f"Stream failed, retrying: {e}")
                delay = self.retry_policy.get_delay(attempt)
                await asyncio.sleep(delay)
                # Retry by calling complete again
                async for chunk in await self.complete(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    stream=True
                ):
                    yield chunk
            else:
                raise


# ============== Convenience Functions ==============

def constant_retry(max_retries: int = 3, delay: float = 0.2) -> RetryPolicy:
    """Create a constant delay retry policy"""
    return RetryPolicy(
        max_retries=max_retries,
        strategy=RetryStrategy.CONSTANT_DELAY,
        initial_delay=delay
    )


def exponential_retry(
    max_retries: int = 3,
    initial_delay: float = 0.3,
    multiplier: float = 1.5,
    max_delay: float = 10.0
) -> RetryPolicy:
    """Create an exponential backoff retry policy"""
    return RetryPolicy(
        max_retries=max_retries,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=initial_delay,
        multiplier=multiplier,
        max_delay=max_delay
    )


# ============== Circuit Breaker ==============

class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance
    Prevents cascading failures when services are unavailable
    """
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        States:
        - closed: Normal operation, requests pass through
        - open: Circuit is tripped, requests fail fast
        - half-open: Testing if service has recovered
        """
        # Check if circuit is open
        if self.state == "open":
            if self.last_failure_time and time.time() - self.last_failure_time > self.reset_timeout:
                # Try to recover - move to half-open state
                self.state = "half-open"
                self.failure_count = 0
                logger.info("Circuit breaker moving to half-open state")
            else:
                raise Exception(
                    f"Circuit breaker is open (failures: {self.failure_count}). "
                    f"Will retry in {self.reset_timeout - (time.time() - self.last_failure_time):.1f}s"
                )
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Reset on success
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                if self.state != "open":
                    self.state = "open"
                    logger.warning(
                        f"Circuit breaker opened after {self.failure_count} failures. "
                        f"Will reset after {self.reset_timeout}s"
                    )
            
            raise e
    
    def reset(self):
        """Manually reset the circuit breaker"""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "threshold": self.failure_threshold,
            "time_until_reset": (
                self.reset_timeout - (time.time() - self.last_failure_time)
                if self.last_failure_time and self.state == "open"
                else 0
            )
        }