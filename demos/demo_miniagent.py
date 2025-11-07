"""
Interactive demo of MiniAgent framework
"""
import argparse
import asyncio
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Add parent directory to path to import framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent_framework.core import Thread
from miniagent_framework.core.plan_agent import Agent, AgentConfig
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
    """Fetch recent market commentary for a ticker using the shared WebSearchTool."""

    def __init__(self, web_search_tool: WebSearchTool):
        self.name = "market_insights"
        self.description = (
            "Summarize the latest context for an index or ticker with date, region, and research links."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol or index (e.g., AAPL, RELIANCE, NIFTY 50).",
                },
                "country": {
                    "type": "string",
                    "description": "Jurisdiction to focus on (e.g., India, United States, Europe).",
                },
                "as_of_date": {
                    "type": "string",
                    "description": (
                        "Date/time the user cares about (ISO format or natural phrases like 'today'). "
                        "Always set this when the user requests recent data."
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": "Emphasis for the search (price action, sentiment, macro outlook, etc.).",
                },
                "include_resources": {
                    "type": "boolean",
                    "description": "Include curated research/news sources for ongoing tracking.",
                },
            },
            "required": ["symbol"],
        }
        self._web_search = web_search_tool

    async def execute(
        self,
        symbol: str,
        country: str = "",
        as_of_date: Optional[str] = None,
        focus: str = "",
        include_resources: bool = True,
    ) -> ToolResult:
        cleaned = symbol.strip()
        if not cleaned:
            return ToolResult(
                success=False,
                error="Symbol is required.",
                display_content="❌ Provide a ticker symbol like AAPL, RELIANCE, or NIFTY 50.",
            )

        country_hint = country.strip() or "global"
        focus_hint = focus.strip() or "price action"
        date_info = self._resolve_date(as_of_date)
        query_parts = [
            cleaned,
            "market insights",
            country_hint,
            date_info["label"],
            focus_hint,
        ]
        query = " ".join(part for part in query_parts if part)

        search_result = await self._web_search.execute(query=query)
        if not search_result.success:
            return ToolResult(
                success=False,
                error=search_result.error,
                display_content=search_result.display_content,
                metadata={"source_tool": "web_search"},
            )

        sources = (search_result.data or [])[:3]
        timestamp_label = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        display_lines = [
            f"🔍 Market insights for {cleaned.upper()}",
            f"• Scope: {country_hint.title()} · As of: {date_info['label']} · Focus: {focus_hint}",
            f"• Generated: {timestamp_label}",
            "",
        ]

        for idx, src in enumerate(sources, start=1):
            title = src.get("title") or "Untitled"
            snippet = (src.get("content") or "").strip()
            brief = (snippet[:220] + "...") if len(snippet) > 220 else snippet
            url = src.get("url") or "N/A"
            display_lines.append(f"{idx}. {title}\n   {brief}\n   {url}")

        if len(display_lines) == 4:
            display_lines.append("No recent commentary detected.")

        resource_links = []
        if include_resources:
            resource_links = self._reference_links_for(country_hint)
            if resource_links:
                display_lines.append("\n📚 Suggested follow-up sources:")
                for ref in resource_links:
                    display_lines.append(f"- {ref['name']}: {ref['url']} ({ref['description']})")

        payload = {
            "symbol": cleaned.upper(),
            "sources": sources,
            "query": query,
            "country": country_hint,
            "as_of_date": date_info,
            "focus": focus_hint,
            "generated_at": timestamp_label,
            "resources": resource_links,
        }
        return ToolResult(
            success=True,
            data=payload,
            display_content="\n".join(display_lines),
            llm_content=json.dumps(payload),
            metadata={"source_tool": "web_search"},
        )

    def _reference_links_for(self, country_hint: str) -> List[Dict[str, str]]:
        country_hint = country_hint.lower()
        global_links = [
            {
                "name": "Investing.com",
                "url": "https://www.investing.com/",
                "description": "Quotes, earnings calendar, analyst sentiment.",
            },
            {
                "name": "Trading Economics",
                "url": "https://tradingeconomics.com/",
                "description": "Macro releases, FX, and rates dashboards.",
            },
            {
                "name": "SEC EDGAR",
                "url": "https://www.sec.gov/edgar/search/",
                "description": "US filings and disclosures.",
            },
        ]
        india_links = [
            {
                "name": "NSE Announcements",
                "url": "https://www.nseindia.com/companytracker/corporateAnnouncements",
                "description": "Official Indian exchange disclosures.",
            },
            {
                "name": "BSE Corporate Filings",
                "url": "https://www.bseindia.com/corporates/ann.aspx",
                "description": "BSE announcements and board updates.",
            },
            {
                "name": "RBI Press Releases",
                "url": "https://rbi.org.in/scripts/bs_viewcontent.aspx?Id=2009",
                "description": "Policy commentary impacting Indian markets.",
            },
        ]
        europe_links = [
            {
                "name": "ESMA Register",
                "url": "https://registers.esma.europa.eu/publication/",
                "description": "Regulatory notices for EU-listed issuers.",
            },
            {
                "name": "ECB SDW",
                "url": "https://sdw.ecb.europa.eu/",
                "description": "Euro-area macro indicators and rates.",
            },
        ]

        if "india" in country_hint or "nifty" in country_hint:
            return global_links + india_links
        if any(x in country_hint for x in ("europe", "uk", "eu")):
            return global_links + europe_links
        if any(x in country_hint for x in ("us", "usa", "america", "nasdaq", "dow", "s&p", "sp500")):
            return global_links
        return global_links

    def _resolve_date(self, raw: Optional[str]) -> Dict[str, str]:
        """Normalize user-provided timestamps to a UTC label + ISO string."""

        def _format(dt: datetime) -> Dict[str, str]:
            dt_utc = dt.astimezone(timezone.utc)
            return {
                "label": dt_utc.strftime("%Y-%m-%d %H:%M UTC"),
                "iso": dt_utc.isoformat(),
                "source": raw or "auto_now",
            }

        if not raw:
            return _format(datetime.now(timezone.utc))

        normalized = raw.strip().lower()
        if normalized in {"today", "now", "latest", "current"}:
            return _format(datetime.now(timezone.utc))

        parse_attempts = [raw, raw.replace("Z", "+00:00") if "Z" in raw else raw]
        for candidate in parse_attempts:
            try:
                parsed = datetime.fromisoformat(candidate)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return _format(parsed)
            except ValueError:
                continue

        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                parsed = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                return _format(parsed)
            except ValueError:
                continue

        return _format(datetime.now(timezone.utc))


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
    registry.register(KnowledgeBaseTool(
            knowledge={
                "miniagent_overview": textwrap.dedent(
                    """\
                    MiniAgent is a lightweight yet production-ready agent runtime. It ships two execution modes:
                    (1) Slim Agent – a single-loop executor focused on fast tool decisions.
                    (2) Planner Agent – a structured planner that drafts steps, executes tools, and summarizes.
                    The framework is written in Python, exposes pluggable tools, and integrates tightly with
                    StreamCallback-based observability hooks."""
                ),
                "miniagent_objective": textwrap.dedent(
                    """\
                    Objective: make it simple for product teams to embed autonomous reasoning without scaffolding an entire LLM stack.
                    Core promises:
                    • ergonomic API surface (Agent, Thread, ToolRegistry)
                    • deterministic control over tool usage and retries
                    • observability out of the box (event stream, metrics snapshot)
                    • drop-in demos plus Streamlit playground for experimentation."""
                ),
                "miniagent_principles": textwrap.dedent(
                    """\
                    Design principles:
                    1. Human-first transparency — every tool call and LLM decision is emitted as an event.
                    2. Composability — agents, tools, and LLM providers are swappable dataclasses.
                    3. Production empathy — clear timeouts, retry policies, and thread serialization support.
                    4. Minimal magic — default prompts are explicit and easy to override."""
                ),
                "thread_definition": textwrap.dedent(
                    """\
                    A Thread captures the ordered history of Message objects (user, assistant, and tool replies).
                    MiniAgent stores events on the thread as well, enabling resumable conversations and debugging.
                    Threads expose metadata fields for plan summaries, last request IDs, and other runtime hints."""
                ),
                "telemetry_observability": textwrap.dedent(
                    """\
                    Observability is driven by StreamCallback + Observability helper:
                    • StreamCallback lets you register handlers for AGENT_THINKING, TOOL_RESULT, STREAM_CHUNK, etc.
                    • Observability scopes events per session/request and appends them to the thread.
                    • Plan Agent also emits PLAN_GENERATING and PLAN_STEP events for UI visualizations."""
                ),
                "pricing": textwrap.dedent(
                    """\
                    This playground assumes three SaaS tiers:
                    • Starter $9/mo (10 seats, 100GB, community support)
                    • Professional $29/mo (50 seats, 1 TB, live chat, API access)
                    • Enterprise — custom pricing, unlimited seats, SSO + dedicated TAM."""
                ),
                "support": textwrap.dedent(
                    """\
                    Support matrix:
                    • Starter – 24/7 email + public docs
                    • Professional – adds live chat and quarterly success reviews
                    • Enterprise – phone hotline, dedicated account manager, private Slack channel, and on-call escalation."""
                ),
            }
        ))
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
        arg_display = json.dumps(args, indent=2, ensure_ascii=False) if args else "{}"
        print(f"🔧 Starting {tool} with:")
        print(textwrap.indent(arg_display, "      "))

    def tool_end_handler(event):
        data = event.content or {}
        tool = data.get("tool")
        success = data.get("success")
        duration = data.get("duration_ms")
        icon = "✅" if success else "⚠️"
        timing = f" in {duration} ms" if duration is not None else ""
        print(f"{icon} {tool} finished{timing}")

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
        metadata = payload.get("metadata") or {}
        icon = "   ✓" if success else "   ✗"
        details: List[str] = []
        if metadata.get("duration_ms") is not None:
            details.append(f"{metadata['duration_ms']} ms")
        if metadata.get("quality"):
            details.append(f"quality: {metadata['quality']}")
        suffix = f" ({', '.join(details)})" if details else ""
        print(f"{icon} Tool completed{suffix}" if success else f"{icon} Tool failed{suffix}")
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
