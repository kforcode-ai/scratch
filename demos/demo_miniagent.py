"""
Interactive demo of MiniAgent framework
"""
import argparse
import asyncio
import json
import os
import sys
import textwrap
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add parent directory to path to import framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent_framework.core import Agent, AgentConfig, Thread
from miniagent_framework.core.tools import (
    ToolRegistry,
    KnowledgeBaseTool,
    WebSearchTool,
    DateTimeTool,
    Tool,
    ToolResult,
)
from miniagent_framework.core.events import StreamCallback, EventType
from miniagent_framework.core.llm import constant_retry

# Load environment variables
load_dotenv()


class MarketInsightsTool(Tool):
    """Fetch recent market commentary for a ticker using web search."""

    def __init__(self, web_search_tool: WebSearchTool):
        self.name = "market_insights"
        self.description = (
            "Collect recent price/insight snippets for a ticker symbol using web search."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol (e.g., 'AAPL', 'NIFTY BANK', 'ABBOTINDIA').",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional geographic or market scope to bias the results.",
                },
            },
            "required": ["symbol"],
        }
        self._web_search = web_search_tool

    async def execute(self, symbol: str, scope: str = "") -> ToolResult:
        cleaned = symbol.strip()
        if not cleaned:
            return ToolResult(
                success=False,
                error="Ticker symbol is required.",
                display_content="❌ Please provide a symbol, e.g., `ABBOTINDIA`.",
            )

        scope_hint = scope.strip() or "stock market"
        query = f"{cleaned} stock latest performance {scope_hint}"

        search_result = await self._web_search.execute(query=query)
        if not search_result.success:
            # Bubble up the failure so the agent can handle it.
            return ToolResult(
                success=False,
                error=search_result.error,
                display_content=search_result.display_content,
                metadata={"source_tool": "web_search"},
            )

        sources: List[Dict[str, Any]] = search_result.data or []
        top_sources = sources[:3]

        display_lines = [f"🔍 Market insights for {cleaned.upper()}:"]
        for idx, src in enumerate(top_sources, start=1):
            title = src.get("title") or "Untitled"
            content = (src.get("content") or "").strip()
            brief = (content[:220] + "...") if len(content) > 220 else content
            url = src.get("url") or "N/A"
            display_lines.append(f"{idx}. {title}\n   {brief}\n   {url}")

        if len(display_lines) == 1:
            display_lines.append("No recent market commentary found.")

        payload = {
            "symbol": cleaned.upper(),
            "sources": top_sources,
            "query": query,
        }

        return ToolResult(
            success=True,
            data=payload,
            display_content="\n".join(display_lines),
            llm_content=json.dumps(payload),
            metadata={"source_tool": "web_search"},
        )


def print_divider(char: str = "-") -> None:
    print(char * 60)


def render_final_response(raw_response: str, show_message: bool = True) -> None:
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError):
        if show_message and raw_response:
            print(raw_response)
        return

    message = payload.get("message")
    summary = payload.get("summary")
    clarification = payload.get("clarification")

    if show_message:
        if isinstance(message, (dict, list)):
            print(json.dumps(message, indent=2))
        else:
            print(message or "")

    if clarification:
        print("\nℹ️ Clarification required. Please respond to continue.")

    if summary is not None:
        print("\n🔎 Summary:")
        if isinstance(summary, (dict, list)):
            print(json.dumps(summary, indent=2))
        else:
            print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniAgent interactive demo.")
    parser.add_argument("--model", help="LLM model identifier to use.")
    parser.add_argument("--provider", help="LLM provider name (e.g., openai, anthropic).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature override.")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for LLM completions.")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable streaming by default.",
    )
    parser.add_argument("--agent-name", default="MiniBot", help="Display name for the agent.")
    parser.add_argument("--tool-timeout", type=float, help="Timeout in seconds for tool execution.")
    parser.add_argument("--llm-timeout", type=float, help="Timeout in seconds for LLM calls.")
    return parser.parse_args()


def render_plan(thread: Thread) -> None:
    plan = thread.metadata.get("plan")
    if not plan:
        print("ℹ️ No active plan yet. Ask for a breakdown to trigger planning.")
        return
    print("🗺️ Current plan:")
    print(textwrap.indent(plan, "   "))


def render_history(thread: Thread, limit: int = 6) -> None:
    if not thread.messages:
        print("History is empty—start chatting!")
        return
    print("🕑 Recent turns:")
    for message in thread.messages[-limit:]:
        prefix = "You " if message.role == "user" else "MiniBot "
        content = message.content.strip()
        print(f"{prefix:>8}: {content}")


def show_toolbox(registry: ToolRegistry) -> None:
    schemas = registry.get_full_schemas()
    print("🛠️ Available tools:")
    for name, schema in schemas.items():
        description = schema.get("description", "No description provided.")
        print(f"   • {name}: {description}")


def show_settings(config: AgentConfig, stream_enabled: bool) -> None:
    print("⚙️ Session settings:")
    print(f"   • Model: {config.model}")
    print(f"   • Streaming: {'on' if stream_enabled else 'off'}")
    print(f"   • Planning enabled: {'yes' if config.planning_enabled else 'no'}")
    print(f"   • Temperature: {config.temperature}")


async def main(args: argparse.Namespace):
    """Interactive demo of the framework"""

    print_divider("=")
    print("🤖 MiniAgent Framework - Interactive Demo")
    print_divider("=")

    # 1. Set up tools
    registry = ToolRegistry()

    # Add knowledge base with more content
    registry.register(KnowledgeBaseTool({
        "pricing": """📊 **Pricing Plans:**
• Starter: $9/month (10 users, 100GB storage)
• Professional: $29/month (50 users, 1TB storage)
• Enterprise: Custom pricing (unlimited users, custom features)
All plans include SSL, daily backups, and 99.9% uptime guarantee.""",

        "features": """✨ **Key Features:**
• Real-time collaboration and document sharing
• Advanced security with 2FA and SSO
• Analytics dashboard with custom reports
• 1000+ integrations (Slack, Teams, Google, etc.)
• Mobile apps for iOS and Android
• API access (Pro and Enterprise)""",

        "support": """🎧 **Support Options:**
• 24/7 email support (all plans)
• Live chat support (Professional+)
• Phone support (Enterprise)
• Dedicated account manager (Enterprise)
• Community forum and knowledge base""",

        "refund": """💰 **Refund Policy:**
• 30-day money-back guarantee
• No questions asked
• Full refund for first-time customers
• Pro-rated refunds for annual plans"""
    }))

    # Additional curated tools for richer chat scenarios
    web_search_tool = WebSearchTool()
    registry.register(web_search_tool)
    registry.register(DateTimeTool())
    registry.register(MarketInsightsTool(web_search_tool))

    # Add a simple custom tool
    @registry.tool(description="Calculate math expressions")
    async def calculate(expression: str):
        try:
            # Safe eval for simple math
            allowed_names = {
                k: v for k, v in __builtins__.items()
                if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"📐 Calculation: {expression} = {result}"
        except Exception:
            return "❌ Invalid expression. Try something like: 2+2, 10*5, 100/4"

    @registry.tool(name="summarize_notes", description="Condense notes or bullet points into a concise TL;DR.")
    async def summarize_notes(raw_notes: str):
        if not raw_notes.strip():
            return "Please share the notes you want me to condense."
        chunks = [
            line.strip("•- ").strip()
            for line in raw_notes.splitlines()
            if line.strip()
        ]
        segments = chunks or [raw_notes.strip()]
        summary = "; ".join(segments[:3])
        if len(segments) > 3:
            summary += f"; plus {len(segments) - 3} more highlights."
        return f"📝 TL;DR: {summary}"

    @registry.tool(name="action_plan", description="Break a goal into three focused steps.")
    async def action_plan(goal: str):
        cleaned = goal.strip()
        if not cleaned:
            return "Let me know the goal you'd like a quick action plan for."
        return (
            f"🚀 Action plan for '{cleaned}':\n"
            "1. Clarify the desired outcome and define what success looks like.\n"
            "2. Identify the stakeholders/resources required for the first milestone.\n"
            "3. Schedule a checkpoint to review progress and adjust scope."
        )

    # 2. Set up callbacks to show what's happening
    callbacks = StreamCallback()
    stream_state = {"chunks": False}

    # Show when agent is thinking
    def thinking_handler(event):
        content = event.content
        if isinstance(content, dict):
            if "plan" in content:
                print("🗺️ Plan:")
                print(textwrap.indent(content["plan"], "   "))
                return
            if "plan_progress" in content:
                total = content.get("total_steps")
                print(f"✅ Plan progress: completed step {content['plan_progress']} of {total}")
                return
            if "tool_calls" in content:
                print("💭 Thinking: analysing tool requirements...")
                return
            if "action" in content:
                print(f"💭 Thinking: {content['action']}")
                return
        if isinstance(content, str):
            print(f"💭 Thinking: {content}")
        else:
            print("💭 Thinking: processing...")

    callbacks.on(EventType.AGENT_THINKING, thinking_handler)

    def plan_event_handler(event):
        info = event.content or {}
        status = info.get("status")
        if status == "starting":
            print("🧭 Generating a fresh plan...")
        elif status == "completed":
            print(f"🧭 Plan ready with {info.get('steps', 0)} steps.")

    def plan_step_handler(event):
        step = (event.content or {}).get("step")
        if not step:
            return
        position = step.get("position")
        description = step.get("description", "").strip()
        tool_hint = step.get("tool")
        suffix = f" (tool: {tool_hint})" if tool_hint else ""
        print(f"   • Step {position}: {description}{suffix}")

    def tool_start_handler(event):
        data = event.content or {}
        tool = data.get("tool")
        args = data.get("arguments", {})
        print(f"🔧 Starting {tool} with {args}")

    def tool_end_handler(event):
        data = event.content or {}
        tool = data.get("tool")
        success = data.get("success")
        icon = "✅" if success else "⚠️"
        print(f"{icon} {tool} finished")

    def tool_progress_handler(event):
        data = event.content or {}
        tool = data.get("tool")
        progress = data.get("progress")
        if progress is not None:
            print(f"   ↳ {tool} progress: {progress}")

    def llm_start_handler(event):
        info = event.content or {}
        phase = info.get("phase", "llm")
        print(f"🧠 Calling LLM ({phase})...")

    def llm_end_handler(event):
        info = event.content or {}
        phase = info.get("phase", "llm")
        print(f"🧠 LLM ({phase}) complete.")

    def tool_result_handler(event):
        payload = event.content or {}
        success = payload.get("success")
        display = payload.get("display") or payload.get("llm")
        icon = "   ✓" if success else "   ✗"
        print(f"{icon} Tool completed" if success else f"{icon} Tool failed")
        if not success and payload.get("error"):
            print(textwrap.indent(f"Error: {payload['error']}", "      "))
        if display:
            printable = display if isinstance(display, str) else json.dumps(display, indent=2)
            print(textwrap.indent(printable, "      "))

    callbacks.on(EventType.PLAN_GENERATING, plan_event_handler)
    callbacks.on(EventType.PLAN_STEP, plan_step_handler)
    callbacks.on(EventType.TOOL_EXECUTION_START, tool_start_handler)
    callbacks.on(EventType.TOOL_EXECUTION_END, tool_end_handler)
    callbacks.on(EventType.TOOL_PROGRESS, tool_progress_handler)
    callbacks.on(EventType.LLM_CALL_START, llm_start_handler)
    callbacks.on(EventType.LLM_CALL_END, llm_end_handler)
    callbacks.on(EventType.TOOL_RESULT, tool_result_handler)

    # Show streaming (print each chunk inline)
    def stream_start_handler(event):
        stream_state["chunks"] = False

    def stream_chunk_handler(event):
        stream_state["chunks"] = True
        chunk = event.content or ""
        print(chunk, end="", flush=True)

    def stream_end_handler(event):
        if stream_state["chunks"]:
            print()

    callbacks.on(EventType.STREAM_START, stream_start_handler)
    callbacks.on(EventType.STREAM_CHUNK, stream_chunk_handler)
    callbacks.on(EventType.STREAM_END, stream_end_handler)

    # 3. Configure agent
    config_kwargs: Dict[str, Any] = {
        "name": args.agent_name,
        "system_prompt": """You are a helpful AI assistant with access to various tools.

Available tools:
- web_search: Search the web for current information
- knowledge_base: Access internal knowledge about pricing, features, support
- market_insights: Gather recent market commentary for a ticker symbol
- calculate: Perform mathematical calculations
- summarize_notes: Condense notes into a concise TL;DR
- action_plan: Break a goal into action steps
- get_datetime: Get current date and time

IMPORTANT: When you receive tool results, use the actual data provided to give accurate answers. Do not say you don't have access to data if tool results are provided. Extract and summarize the relevant information from tool results to answer user questions directly.

Use the appropriate tools to provide accurate, helpful responses. For product information (pricing, features, support), use the knowledge_base tool. For web searches, use web_search. For calculations, use calculate. For time/date, use get_datetime.""",
        "retry_policy": constant_retry(max_retries=2, delay=0.5),
        "planning_enabled": True,
    }

    if args.model:
        config_kwargs["model"] = args.model
    if args.provider:
        config_kwargs["provider"] = args.provider
    if args.temperature is not None:
        config_kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        config_kwargs["max_tokens"] = args.max_tokens
    if args.tool_timeout is not None:
        config_kwargs["tool_timeout"] = args.tool_timeout
    if args.llm_timeout is not None:
        config_kwargs["llm_timeout"] = args.llm_timeout
    if args.stream is not None:
        config_kwargs["stream_by_default"] = args.stream
    else:
        config_kwargs["stream_by_default"] = True

    config = AgentConfig(**config_kwargs)

    # 4. Create agent and thread for conversation
    agent = Agent(config=config, tools=registry, callbacks=callbacks)
    thread = Thread()  # Maintain conversation history

    # 5. Welcome message
    print("\n🎉 Welcome! I'm MiniBot, your AI assistant.")
    print_divider("-")

    stream_enabled = config.stream_by_default

    # 6. Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! Thanks for using MiniAgent!")
                break

            if user_input.lower() == 'reset':
                thread = Thread()
                stream_state["chunks"] = False
                print("🔄 Conversation history cleared!")
                continue

            if user_input.lower() == 'help':
                print("\n📚 Available commands:")
                print("  • Ask about pricing, features, support, weather, markets, planning, or brainstorming.")
                print("  • Type 'reset' to clear history or 'quit' to exit.")
                print("  • Toggle streaming with 'stream on' / 'stream off'.")
                print("  • Advanced shortcuts:")
                print("       :plan      show the current execution plan")
                print("       :tools     list integrated tools")
                print("       :history   print recent turns")
                print("       :settings  display session settings")
                continue

            if user_input.startswith(":"):
                cmd = user_input[1:].strip().lower()
                if cmd == "plan":
                    render_plan(thread)
                elif cmd == "tools":
                    show_toolbox(registry)
                elif cmd == "history":
                    render_history(thread)
                elif cmd in {"setting", "settings", "config"}:
                    show_settings(agent.config, stream_enabled)
                elif cmd in {"reset", "clear"}:
                    thread = Thread()
                    stream_state["chunks"] = False
                    print("🔄 Conversation history cleared!")
                else:
                    print(f"🤔 Unknown command ':{cmd}'. Type 'help' to see supported shortcuts.")
                continue

            if user_input.lower() == 'stream on':
                stream_enabled = True
                print("✅ Streaming enabled")
                continue

            if user_input.lower() == 'stream off':
                stream_enabled = False
                print("✅ Streaming disabled")
                continue

            if not user_input:
                continue

            # Process with agent
            print("-" * 40)
            print("🤖 MiniBot: ", end="")

            stream_state["chunks"] = False
            response = await agent.run(
                user_input=user_input,
                thread=thread,  # Maintain conversation context
                stream=stream_enabled
            )

            # Always display the final response with structure awareness
            if not stream_enabled:
                render_final_response(response, show_message=True)
            elif not stream_state["chunks"]:
                render_final_response(response, show_message=True)
            else:
                render_final_response(response, show_message=False)

            request_marker = thread.metadata.get("last_request_id")
            if request_marker:
                print(f"\n📎 Request ID: {request_marker}")

            stream_state["chunks"] = False

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as exc:
            print(f"\n❌ Error: {exc}")
            print("Please try again or type 'help' for assistance.")

    print()
    print_divider("=")
    print("✨ Thanks for trying MiniAgent Framework!")
    print_divider("=")


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
