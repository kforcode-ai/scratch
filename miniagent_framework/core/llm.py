"""
LLM client with support for multiple providers and retry policies
"""
import asyncio
import logging
from typing import AsyncIterator, Dict, Any, List, Optional, Union
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
        retry_policy: Optional[RetryPolicy] = None
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
        Complete a chat with retry logic
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