# 🤖 MiniAgent Framework

A **simple, powerful agent framework** for building conversational AI agents with tool use, streaming, Redis-backed sessions, and robust retry logic.

## 📁 Project Structure

```
miniagent_framework/
├── core/                    # Core agent framework
│   ├── __init__.py         # Package exports
│   ├── core.py             # Agent, Thread, AgentConfig
│   ├── llm.py              # LLM client with retry logic
│   ├── tools.py            # Tool system and registry
│   └── events.py           # Event system and callbacks
│
├── extensions/             # Extended functionality
│   ├── redis_client.py     # Redis session manager
│   ├── session.py          # Session-aware agent
│   └── events_enhanced.py  # Enhanced events for sessions
│
└── tools/                  # Enhanced tool implementations
    └── tools_enhanced.py   # Token-optimized tools

demos/                      # Example usage and demos
├── demo_miniagent.py       # Basic framework demo
├── demo_redis_session.py   # Redis sessions demo
└── demo_enhanced_tools.py  # Token optimization demo

tests/                      # Test suites
└── test_comprehensive.py   # Comprehensive framework tests

docs/                       # Documentation
├── README_FRAMEWORK.md     # Core framework docs
├── README_REDIS.md         # Redis session docs
└── improvements.md         # Future improvements

config/                     # Configuration files
└── docker-compose.yml      # Redis setup
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core framework only (choose your LLM provider)
pip install python-dotenv

# For OpenAI/GPT:
pip install openai

# For Google Gemini:
pip install google-generativeai

# For Anthropic Claude:
pip install anthropic

# Full installation with all providers:
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Create .env file with your provider's API key
echo "OPENAI_API_KEY=your-key-here" >> .env      # For OpenAI
echo "GOOGLE_API_KEY=your-key-here" >> .env       # For Gemini
echo "ANTHROPIC_API_KEY=your-key-here" >> .env    # For Claude
echo "TAVILY_API_KEY=your-key-here" >> .env       # Optional for web search
```

### 3. Run Demos

#### Basic Agent Demo
```bash
python demos/demo_miniagent.py
```

#### Redis Session Management
```bash
# Start Redis first
docker-compose -f config/docker-compose.yml up -d

# Run session demo
python demos/demo_redis_session.py
```

#### Token-Optimized Tools
```bash
python demos/demo_enhanced_tools.py
```

#### Multi-LLM Provider Demo
```bash
python demos/demo_multi_llm_providers.py
```

## 🔍 Observability
- Structured JSON logging now binds `session_id`, `request_id`, and `operation_id` context by default (see `miniagent_framework/core/logging.py`).
- LLM call events now include provider token usage metrics and latency budgets when available.
- Tool executions record duration and flag low-signal outputs to surface weak tool responses quickly.

## ✨ Key Features

### Core Framework
- 🤖 **Multi-LLM Support** - OpenAI, Google Gemini, Anthropic Claude
- 🔧 **Tool System** - Easy tool creation with decorators
- 📡 **Real-time Streaming** - Stream responses word-by-word
- 🔄 **Retry Logic** - Configurable retry strategies
- 📝 **Event System** - Complete audit trail of all actions

### Extensions
- 💾 **Redis Sessions** - Persistent conversations across restarts
- 🚀 **Token Optimization** - Send only required params to LLM
- 🔐 **Session Management** - User-based session isolation
- ⚡ **Caching** - Cache tool results and LLM responses
- 🔄 **Distributed** - Multiple agent instances can share sessions

## 🤖 Multi-LLM Provider Support

The framework supports multiple LLM providers out of the box:

### Supported Providers

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| OpenAI | GPT-4, GPT-3.5 | `OPENAI_API_KEY` |
| Google Gemini | Gemini Pro, Gemini Flash | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Anthropic | Claude 3 (Opus, Sonnet, Haiku) | `ANTHROPIC_API_KEY` |

### Usage Example

```python
from miniagent_framework.core import Agent, AgentConfig

# Using OpenAI (default)
agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4o-mini"  # Optional, uses default if not specified
))

# Using Google Gemini
agent = Agent(AgentConfig(
    provider="gemini",
    model="gemini-1.5-flash"
))

# Using Anthropic Claude
agent = Agent(AgentConfig(
    provider="anthropic",
    model="claude-3-haiku-20240307"
))

# The API is the same regardless of provider
response = await agent.run("Hello, how are you?")
```

### Provider-Specific Features

All providers support:
- ✅ Streaming responses
- ✅ Tool/function calling
- ✅ Retry policies
- ✅ Temperature and max_tokens control

### Switching Providers

You can easily switch providers without changing your code:

```python
# Configure via environment variable
config = AgentConfig(
    provider=os.getenv("LLM_PROVIDER", "openai")  # Flexible provider selection
)
```

## 🏗️ Architecture

The framework follows a modular architecture:

1. **Core Layer** - Basic agent functionality
2. **Extensions Layer** - Optional enhancements (Redis, caching)
3. **Tools Layer** - Reusable tool implementations
4. **Application Layer** - Your custom implementation

### Agent Loop Internals
- `Thread` stores the conversation, telemetry events, and optional execution plan metadata.
- `_prepare_iteration_messages` composes the system prompts, plan guidance, and context before each LLM call.
- `_call_llm` unifies streaming and non-streaming responses and reports them through the `LLMCallResult` dataclass.
- `_execute_tools` runs functions requested by the model and automatically advances plan progress.
- `_complete_remaining_plan_steps` finalizes any outstanding plan items once a user-facing answer is produced.

### Telemetry & Observability Primitives

Every interaction carries structured identifiers so you can trace the agent end-to-end:

| Concept | Purpose | Where to Inspect |
|---------|---------|------------------|
| `session_id` / `thread_id` | Conversation identifier spanning multiple user turns. `session_id` is bound into logs; `thread_id` is the persisted value on the `Thread`. | `Thread.id`, event metadata, structured logs |
| `request_id` | Unique to a single `agent.run(...)` call. Binds plan generation, tool calls, and the final response for that user message. | Event metadata, structlog context (auto-bound) |
| `operation_id` / `parent_operation_id` | Identifies individual operations within a request (e.g., decision LLM call, streaming pass, tool execution) and the parent that scheduled them. | Event metadata, log entries emitted inside the operation scope |
| `Event` | Structured record (type, timestamp, content, metadata) emitted for every lifecycle change—planning, tool usage, streaming, clarifications, etc. | `Thread.events`, `StreamCallback` handlers |
| Session | Higher-level conversation container. By default the `Thread` acts as the in-memory session; Redis/SQLite extensions persist it across processes. | `miniagent_framework/core/core.py`, `extensions/session.py` |

Tip: register handlers on `StreamCallback` (or read `Thread.events`) to export telemetry to Prometheus, OpenTelemetry, or your logging pipeline.

## 🤝 Contributing

1. Keep code simple and readable
2. Add tests for new features
3. Update documentation
4. Follow existing patterns

## 📄 License

MIT

---

Built with ❤️ for creators who want powerful AI agents without complexity.
