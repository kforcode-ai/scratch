"""
MiniAgent - A simple, powerful agent framework
Inspired by HICA and BAML, focusing on simplicity and readability
"""

from .core import Agent, Thread, AgentConfig, Message
from .tools import Tool, ToolRegistry, ToolResult
from .llm import LLMClient, LLMProvider, RetryPolicy, RetryStrategy
from .events import Event, EventType, StreamCallback
from .providers import BaseLLMProvider, OpenAIProvider, GeminiProvider, AnthropicProvider

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Thread", 
    "AgentConfig",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "LLMClient",
    "RetryPolicy",
    "RetryStrategy",
    "Event",
    "EventType",
    "StreamCallback"
]
