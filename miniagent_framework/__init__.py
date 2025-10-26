"""
MiniAgent Framework - Simple, powerful agent framework
"""

# Core imports
from miniagent_framework.core import (
    Agent,
    AgentConfig,
    Thread,
    Message,
    StreamCallback,
    Event,
    EventType,
)

from miniagent_framework.core.tools import (
    # Basic tools
    Tool,
    SimpleTool,
    ToolRegistry,
    ToolResult,
    # Enhanced tools
    EnhancedTool,
    ToolParameter,
    ParamType,
    APITool,
    # Built-in tools
    KnowledgeBaseTool,
    WebSearchTool,
)

from miniagent_framework.core.llm import (
    LLMClient,
    LLMProvider,
    RetryPolicy,
    RetryStrategy,
    constant_retry,
    exponential_retry,
)

from miniagent_framework.core.providers import (
    BaseLLMProvider,
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider,
)

# Extension imports (optional)
try:
    from miniagent_framework.extensions.session import (
        SessionAgent,
        SessionManager,
        SessionThread,
    )
    from miniagent_framework.extensions.redis_client import (
        RedisConfig,
        RedisSessionManager,
    )
    _has_redis = True
except ImportError:
    _has_redis = False

__version__ = "0.3.0"

__all__ = [
    # Core
    "Agent",
    "AgentConfig", 
    "Thread",
    "Message",
    "StreamCallback",
    "Event",
    "EventType",
    # Tools (basic)
    "Tool",
    "SimpleTool",
    "ToolRegistry",
    "ToolResult",
    # Tools (enhanced)
    "EnhancedTool",
    "ToolParameter",
    "ParamType",
    "APITool",
    # Built-in tools
    "KnowledgeBaseTool",
    "WebSearchTool",
    # LLM
    "LLMClient",
    "LLMProvider",
    "RetryPolicy",
    "RetryStrategy",
    "constant_retry",
    "exponential_retry",
    # Providers
    "BaseLLMProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "AnthropicProvider",
]

# Add extension exports if available
if _has_redis:
    __all__.extend([
        "SessionAgent",
        "SessionManager",
        "SessionThread",
        "RedisConfig",
        "RedisSessionManager",
    ])