# MiniAgent Framework 🤖

A **simple, powerful agent framework** for building conversational AI agents with tool use, streaming, and robust retry logic.

## 🎯 Philosophy

**Simplicity over SOLID** - We prioritize clean, readable code that's easy to understand and modify over complex design patterns.

## ✨ Key Features

### 1. **LLM Clients with Retry Policies**
- Support for OpenAI (Gemini, Anthropic coming soon)
- Configurable retry strategies:
  - Constant delay
  - Exponential backoff with customizable parameters
- Automatic error handling and recovery

### 2. **Real-time Streaming**
- Stream LLM responses word-by-word
- Show users what's happening behind the scenes:
  - Agent thinking process
  - Tool selection and execution
  - Tool outputs
  - Final responses
- Everything is streamed, even structured responses

### 3. **Tool System**
- Simple decorator-based tool registration
- Automatic parameter inference
- Built-in error handling and retry logic
- Structured tool results with human and LLM-friendly outputs
- Built-in tools:
  - Web search (Tavily)
  - Knowledge base (mockable)

### 4. **Conversation Management**
- Event-sourced conversation threads
- Full conversation history and context
- Stateful and stateless operation modes
- External context injection

### 5. **Event-Driven Architecture**
- Inspired by HICA's event sourcing
- Complete audit trail of all actions
- Flexible callback system for monitoring

## 📁 Project Structure

```
miniagent/              # The framework
├── __init__.py        # Package exports
├── core.py            # Agent, Thread, AgentConfig
├── llm.py             # LLM client with retry logic
├── tools.py           # Tool system and registry
├── events.py          # Event system and callbacks
└── README.md          # Framework documentation

example_miniagent.py   # Complete usage examples
test_miniagent.py      # Basic tests
```

## 🚀 Quick Start

```python
from miniagent import Agent, AgentConfig, ToolRegistry

# Create tools
registry = ToolRegistry()

@registry.tool()
async def calculate(expression: str):
    return eval(expression)

# Configure agent
config = AgentConfig(
    system_prompt="You are a helpful assistant.",
    stream_by_default=True
)

# Create and run
agent = Agent(config=config, tools=registry)
response = await agent.run("What's 2 + 2?")
```

## 🔄 Retry Policies

```python
# Exponential backoff
from miniagent.llm import exponential_retry

config.retry_policy = exponential_retry(
    max_retries=3,
    initial_delay=0.3,
    multiplier=1.5,
    max_delay=10.0
)

# Constant delay
from miniagent.llm import constant_retry

config.retry_policy = constant_retry(
    max_retries=3,
    delay=0.2
)
```

## 🛠️ Creating Custom Tools

```python
from miniagent.tools import ToolResult

@registry.tool(description="Search database")
async def search(query: str) -> ToolResult:
    # Your implementation
    results = await database.search(query)
    
    return ToolResult(
        success=True,
        data=results,
        display_content=f"Found {len(results)} results",  # For humans
        llm_content=str(results)  # For LLM
    )
```

## 📡 Event Callbacks

```python
from miniagent import StreamCallback, EventType

callbacks = StreamCallback()

# Monitor what's happening
callbacks.on(EventType.AGENT_THINKING, lambda e: print(f"🤔 {e.content}"))
callbacks.on(EventType.TOOL_EXECUTION, lambda e: print(f"🔧 {e.content}"))
callbacks.on(EventType.STREAM_CHUNK, lambda e: print(e.content, end=""))
```

## 💬 Conversation Threads

```python
from miniagent import Thread

# Stateful conversations
thread = Thread()
await agent.run("Hello!", thread=thread)
await agent.run("What did I say?", thread=thread)  # Remembers context

# Stateless queries
await agent.run("What's the weather?")  # No thread, no memory
```

## 🎨 Customization

Everything is customizable:
- System prompts
- Temperature and model settings
- Retry strategies
- Tool selection logic
- Event handlers
- Streaming behavior

## 📚 Inspiration

This framework is inspired by:
- [HICA](https://github.com/sandipan1/hica) - Event-sourced architecture and human-in-the-loop design
- [BAML](https://github.com/BoundaryML/baml) - Type-safe LLM interactions
- Focus on **simplicity and readability** over complex abstractions

## 🚦 Getting Started

1. Install dependencies:
```bash
pip install openai python-dotenv
pip install tavily-python  # Optional, for web search
```

2. Set up environment variables:
```bash
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here  # Optional
```

3. Run the example:
```bash
python example_miniagent.py
```

## 🎯 Design Principles

1. **Simplicity First** - Code should be easy to read and understand
2. **Explicit over Implicit** - Clear, obvious behavior
3. **Modular** - Each component does one thing well
4. **Extensible** - Easy to add new tools and capabilities
5. **Production-Ready** - Built-in retry, error handling, and monitoring

## 📄 License

MIT

---

Built with ❤️ for creators who want to build AI agents without complexity.