"""
Interactive demo showcasing the lean Agent runtime.
"""
import argparse
import asyncio
import json
import os
import sys
import textwrap
from typing import Optional

from dotenv import load_dotenv

# Allow running the demo without installing the package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from miniagent_framework.core import Agent, AgentConfig, AgentFactory, Thread  # noqa: E402
from miniagent_framework.core.tools import (  # noqa: E402
    ToolRegistry,
    KnowledgeBaseTool,
    WebSearchTool,
    DateTimeTool,
    Tool,
    ToolResult,
)
from miniagent_framework.core.events import EventType  # noqa: E402


DEFAULT_CREATOR_PROMPT = (
    "You are the MiniAgent demo assistant. Be concise, cite tool outputs when relevant, "
    "and keep responses user friendly."
)


class SampleGlossaryTool(Tool):
    """
    Simple illustrative tool that returns a short glossary entry.
    Demonstrates how to plug a custom tool into Agent.
    """

    def __init__(self) -> None:
        self.name = "glossary_lookup"
        self.description = "Fetch concise definitions for common AI agent terms."
        self.parameters = {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "Glossary term to look up (e.g., 'tool call').",
                }
            },
            "required": ["term"],
        }
        self._entries = {
            "tool call": "A structured request for the assistant to call a registered function.",
            "plan": "A sequence of steps outlining how an agent will approach a task.",
            "thread": "Ordered history of messages shared between user and assistant.",
        }

    async def execute(self, term: str) -> ToolResult:
        normalized = term.strip().lower()
        if not normalized:
            return ToolResult(
                success=False,
                error="Please provide a term to look up.",
                display_content="❌ Missing glossary term.",
            )
        definition = self._entries.get(normalized)
        if not definition:
            return ToolResult(
                success=False,
                error=f"No glossary entry for '{term}'.",
                display_content=f"📓 No glossary entry found for '{term}'.",
            )

        payload = {"term": term, "definition": definition}
        return ToolResult(
            success=True,
            data=payload,
            display_content=f"📓 {term.strip().title()}: {definition}",
            llm_content=json.dumps(payload),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agent interactive demo.")
    parser.add_argument("--model", help="LLM model identifier to use.")
    parser.add_argument("--provider", help="LLM provider name (e.g., openai, anthropic).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature override.")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for LLM completions.")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable streaming responses for the assistant.",
    )
    parser.add_argument("--tool-timeout", type=float, help="Timeout in seconds for tool execution.")
    parser.add_argument("--llm-timeout", type=float, help="Timeout in seconds for LLM calls.")
    return parser.parse_args()


def build_toolbox() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(KnowledgeBaseTool(knowledge={
        "slim agent": "A simplified runtime that uses a single loop to decide between tools and answers.",
        "miniagent": "The full-featured runtime with planning and observability knobs.",
        "tool registry": "Keeps track of the callable tools available to the agent.",
    }))
    registry.register(DateTimeTool())
    registry.register(SampleGlossaryTool())
    registry.register(WebSearchTool())  # Gracefully reports if TAVILY_API_KEY is missing
    return registry


def build_agent(args: argparse.Namespace) -> tuple[Agent, ToolRegistry]:
    toolbox = build_toolbox()
    config = AgentConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.tool_timeout is not None:
        config.tool_timeout = args.tool_timeout
    if args.llm_timeout is not None:
        config.llm_timeout = args.llm_timeout
    if args.stream is not None:
        config.stream_by_default = args.stream

    factory = AgentFactory(
        base_config=config,
        base_prompt=config.system_prompt,
        creator_prompt=DEFAULT_CREATOR_PROMPT,
    )
    agent = factory.create(tool_registry=toolbox)
    return agent, toolbox


def attach_console_observers(agent: Agent) -> None:
    """Display key agent events in the console for transparency."""
    streaming_state = {"active": False}

    def emit_event(event_type: str, **payload) -> None:
        record = {"event_type": event_type, "payload": payload}
        print(json.dumps(record, ensure_ascii=False))

    def handle_tool_selection(event):
        payload = event.content or {}
        tool = payload.get("tool")
        arguments = payload.get("arguments", {})
        emit_event("tool_selection", tool=tool, arguments=arguments)

    def handle_tool_start(event):
        payload = event.content or {}
        tool = payload.get("tool")
        arguments = payload.get("arguments", {})
        emit_event("tool_execution_start", tool=tool, arguments=arguments)

    def handle_tool_result(event):
        payload = event.content or {}
        tool = payload.get("tool")
        success = payload.get("success")
        result = payload.get("result") or {}
        summary = result.get("display") or result.get("llm") or result.get("data") or result
        emit_event("tool_result", tool=tool, success=success, summary=summary)

    def handle_tool_error(event):
        payload = event.content or {}
        tool = payload.get("tool")
        error = payload.get("error")
        emit_event("tool_error", tool=tool, error=error)

    def handle_stream_chunk(text: str):
        if not text:
            return
        if not streaming_state["active"]:
            print("\nAgent ▸ ", end="", flush=True)
            streaming_state["active"] = True
        print(text, end="", flush=True)

    def handle_stream_end():
        if streaming_state["active"]:
            print("")
            streaming_state["active"] = False

    agent.callbacks.on(EventType.TOOL_SELECTION, handle_tool_selection)
    agent.callbacks.on(EventType.TOOL_EXECUTION_START, handle_tool_start)
    agent.callbacks.on(EventType.TOOL_RESULT, handle_tool_result)
    agent.callbacks.on(EventType.TOOL_ERROR, handle_tool_error)
    agent.on_stream_chunk(handle_stream_chunk)
    agent.on_stream_end(handle_stream_end)


def show_intro() -> None:
    print("=" * 60)
    print("🤖 Agent Demo")
    print("=" * 60)
    print(
        textwrap.dedent(
            """
            Type a message and press enter to talk to the agent.
            Commands:
              :history      → show the last few turns
              :tools        → list registered tools
              :context TEXT → set optional extra context for the agent
              :quit         → exit the demo
            """
        ).strip()
    )
    print("=" * 60)


def render_history(thread: Thread, limit: int = 6) -> None:
    if not thread.messages:
        print("No history yet—say hello!")
        return
    print("🕑 Recent turns:")
    for message in thread.messages[-limit:]:
        prefix = "You" if message.role == "user" else "Agent"
        snippet = message.content.strip() or "<tool call>"
        print(f"{prefix:>10}: {snippet}")


def list_tools(registry: ToolRegistry) -> None:
    schemas = registry.get_full_schemas()
    if not schemas:
        print("No tools registered.")
        return
    print("🛠️ Tools available to the agent:")
    for name, schema in schemas.items():
        description = schema.get("description", "No description provided.")
        print(f"   • {name}: {description}")


async def main() -> None:
    load_dotenv()
    args = parse_args()
    show_intro()

    agent, toolbox = build_agent(args)
    attach_console_observers(agent)
    thread = Thread()
    extra_context: Optional[str] = None

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting demo.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {":quit", ":exit"}:
            print("Goodbye!")
            break
        if lowered == ":history":
            render_history(thread)
            continue
        if lowered == ":tools":
            list_tools(toolbox)
            continue
        if lowered.startswith(":context"):
            extra_context = user_input.partition(" ")[2].strip() or None
            if extra_context:
                print(f"Context updated:\n{textwrap.indent(extra_context, '   ')}")
            else:
                print("Cleared extra context.")
            continue

        response = await agent.run(
            user_input,
            thread=thread,
            context=extra_context,
            stream=agent.config.stream_by_default,
        )
        if agent.config.stream_by_default:
            if response:
                print(f"\nAgent (final) > {response}")
        else:
            print(f"Agent > {response}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
