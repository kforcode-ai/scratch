"""
LLM client with support for multiple providers and retry policies
"""
import asyncio
import time
from typing import AsyncIterator, Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

# Import provider implementations
from .logging import logger
from .providers import (
    BaseLLMProvider,
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider
)


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
        self._ensure_credentials()
        logger.debug(
            "llm.call.start",
            provider=self.provider_type.value if isinstance(self.provider_type, LLMProvider) else str(self.provider_type),
            model=self.model,
            stream=stream,
            message_count=len(messages),
            tools=len(tools) if tools else 0,
        )
        # Wrap with circuit breaker if enabled
        if self.circuit_breaker:
            return await self.circuit_breaker.call(
                self._complete_with_retry,
                messages, temperature, max_tokens, tools, stream
            )
        return await self._complete_with_retry(
            messages, temperature, max_tokens, tools, stream
        )
    
    def _ensure_credentials(self) -> None:
        api_key = getattr(self.provider, "api_key", None)
        if api_key:
            return
        hint = {
            LLMProvider.OPENAI: "Set OPENAI_API_KEY in your environment.",
            LLMProvider.GEMINI: "Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment.",
            LLMProvider.GOOGLE: "Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment.",
            LLMProvider.ANTHROPIC: "Set ANTHROPIC_API_KEY in your environment.",
            LLMProvider.CLAUDE: "Set ANTHROPIC_API_KEY in your environment.",
        }.get(self.provider_type, "Provide credentials for the selected LLM provider.")
        logger.error(
            "llm.complete.missing_api_key",
            provider=self.provider_type.value if isinstance(self.provider_type, LLMProvider) else str(self.provider_type),
            hint=hint,
        )
        raise RuntimeError(hint)
    
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
                    # Return the async generator without awaiting
                    return self._stream_with_retry(
                        messages, temperature, max_tokens, tools, attempt
                    )
                else:
                    try:
                        result = await self.provider.complete(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,
                            stream=False
                        )
                        logger.debug("llm.complete.success", stream=False)
                        self._record_attempt(attempt)
                        return result
                    except Exception as err:
                        logger.error(
                            "llm.complete.error",
                            error=str(err),
                            attempt=attempt,
                            stream=False,
                            exc_type=type(err).__name__,
                        )
                        logger.exception("llm.complete.exception", stream=False)
                        raise

            except Exception as e:
                last_error = e
                logger.warning(
                    "llm.complete.retry",
                    attempt=attempt,
                    remaining=self.retry_policy.max_retries - attempt - 1,
                    error=str(e),
                    exc_type=type(e).__name__,
                )
                logger.exception("llm.complete.retry_trace")

                if attempt < self.retry_policy.max_retries - 1:
                    delay = self.retry_policy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
        
        raise last_error or Exception("Failed after retries")

    def consume_last_metrics(self) -> Dict[str, Any]:
        """Return provider-supplied metrics for the most recent call, if any."""
        if hasattr(self.provider, "last_metadata"):
            metadata = getattr(self.provider, "last_metadata") or {}
            if isinstance(metadata, dict):
                self.provider.last_metadata = {}
                return dict(metadata)
        return {}

    def _record_attempt(self, attempt: int) -> None:
        """Attach retry metadata to the provider's last_metadata store."""
        if not hasattr(self.provider, "last_metadata"):
            return
        existing = getattr(self.provider, "last_metadata") or {}
        metadata = dict(existing)
        provider_name = self.provider_type.value if isinstance(self.provider_type, LLMProvider) else str(self.provider_type)
        metadata.setdefault("provider", provider_name)
        metadata["retries"] = attempt
        setattr(self.provider, "last_metadata", metadata)

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
            stream_result = await self.provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True
            )
            self._record_attempt(attempt)
            logger.debug("llm.complete.success", stream=True)
            async for chunk in stream_result:
                yield chunk
        except Exception as e:
            if attempt < self.retry_policy.max_retries - 1:
                logger.warning(
                    "llm.stream.retry",
                    attempt=attempt,
                    error=str(e),
                    exc_type=type(e).__name__,
                )
                logger.exception("llm.stream.retry_trace")
                delay = self.retry_policy.get_delay(attempt)
                await asyncio.sleep(delay)
                # Retry using the next attempt
                async for chunk in self._stream_with_retry(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    attempt=attempt + 1
                ):
                    yield chunk
            else:
                logger.error(
                    "llm.stream.failed",
                    attempt=attempt,
                    error=str(e),
                    exc_type=type(e).__name__,
                )
                logger.exception("llm.stream.failed_trace")
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
