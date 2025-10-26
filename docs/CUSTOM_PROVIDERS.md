# 🔌 Creating Custom LLM Providers

This guide explains how to add support for new LLM providers to the MiniAgent framework.

## 📁 File Structure

After refactoring, the LLM code is organized as:

```
miniagent_framework/core/
├── llm.py         # Core LLMClient, retry logic, enums
└── providers.py   # Provider implementations (OpenAI, Gemini, Anthropic)
```

## 🛠️ Adding a New Provider

To add support for a new LLM provider, follow these steps:

### 1. Create Your Provider Class

Add your provider class to `miniagent_framework/core/providers.py`:

```python
from .providers import BaseLLMProvider

class MyCustomProvider(BaseLLMProvider):
    """Custom LLM provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("MY_PROVIDER_API_KEY")
        self.model = model or "my-default-model"
        super().__init__(self.api_key, self.model)
    
    def _initialize_client(self):
        """Initialize your provider's client"""
        try:
            # Import and initialize your client library
            from my_provider import AsyncClient
            self.client = AsyncClient(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.warning("MyProvider library not installed")
            self.client = None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Union[str, Dict, AsyncIterator]:
        """Implement the completion logic"""
        # Your implementation here
        pass
```

### 2. Register Your Provider

Add your provider to the enum and mapping in `miniagent_framework/core/llm.py`:

```python
# Add to LLMProvider enum
class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    MY_PROVIDER = "my_provider"  # Add your provider

# Add to LLMClient.PROVIDER_MAP
class LLMClient:
    PROVIDER_MAP = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.GEMINI: GeminiProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.MY_PROVIDER: MyCustomProvider,  # Add mapping
    }
```

### 3. Export Your Provider (Optional)

If you want users to directly access your provider class:

```python
# In miniagent_framework/__init__.py
from miniagent_framework.core.providers import (
    # ... existing providers ...
    MyCustomProvider
)

__all__ = [
    # ... existing exports ...
    "MyCustomProvider"
]
```

## 🔑 Key Methods to Implement

Your provider must implement these methods from `BaseLLMProvider`:

### `_initialize_client()`
Initialize your provider's client library.

### `complete()`
The main method for handling completions. Must support:
- Regular text completions
- Tool/function calling (if supported)
- Streaming responses (if `stream=True`)

### Helper Methods (Optional)

You may need to implement helper methods for:
- `_convert_messages()`: Convert OpenAI format to your provider's format
- `_convert_tools()`: Convert tool definitions to your provider's format
- `_parse_response()`: Parse provider response to standard format
- `_stream_response()`: Handle streaming responses

## 📝 Example: Adding Cohere Provider

Here's a complete example of adding Cohere support:

```python
class CohereProvider(BaseLLMProvider):
    """Cohere LLM provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model or "command"
        super().__init__(self.api_key, self.model)
    
    def _initialize_client(self):
        """Initialize Cohere client"""
        try:
            import cohere
            self.client = cohere.AsyncClient(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.warning("Cohere library not installed. Install with: pip install cohere")
            self.client = None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Union[str, Dict, AsyncIterator]:
        """Complete using Cohere API"""
        if not self.client:
            raise RuntimeError("Cohere client not initialized. Please provide COHERE_API_KEY.")
        
        # Convert messages to Cohere format
        prompt = self._messages_to_prompt(messages)
        
        # Call Cohere API
        response = await self.client.generate(
            prompt=prompt,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        else:
            return response.text
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to Cohere prompt format"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"Instructions: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        return prompt + "Assistant: "
    
    async def _stream_response(self, stream) -> AsyncIterator:
        """Stream Cohere response"""
        async for event in stream:
            if hasattr(event, 'text'):
                yield event.text
```

## 🧪 Testing Your Provider

Test your provider with this simple script:

```python
from miniagent_framework.core import Agent, AgentConfig

async def test_provider():
    config = AgentConfig(
        provider="my_provider",
        api_key="your-api-key",  # or use env var
        model="your-model"
    )
    
    agent = Agent(config=config)
    response = await agent.run("Hello, can you hear me?")
    print(response)

# Run the test
import asyncio
asyncio.run(test_provider())
```

## 🎯 Best Practices

1. **Error Handling**: Always handle missing API keys and network errors gracefully
2. **Streaming**: Implement streaming support if your provider supports it
3. **Tool Calling**: Map tool/function calling to your provider's format
4. **Message Format**: Convert OpenAI's message format to your provider's expected format
5. **Response Parsing**: Ensure responses are parsed to the standard format expected by the framework

## 📦 Distribution

If you want to share your provider:

1. Create a separate package (e.g., `miniagent-cohere-provider`)
2. Import and register it dynamically:

```python
# In your package
from miniagent_framework.core.providers import BaseLLMProvider

class CohereProvider(BaseLLMProvider):
    # ... implementation ...

# Register with the framework
def register():
    from miniagent_framework.core.llm import LLMClient, LLMProvider
    
    # Add to enum (if possible)
    # Or just use string provider name
    
    # Add to provider map
    LLMClient.PROVIDER_MAP["cohere"] = CohereProvider
```

## 🔗 Resources

- [Base Provider Interface](../miniagent_framework/core/providers.py)
- [Example Providers](../miniagent_framework/core/providers.py)
- [LLM Client Implementation](../miniagent_framework/core/llm.py)
- [Multi-Provider Demo](../demos/demo_multi_llm_providers.py)


