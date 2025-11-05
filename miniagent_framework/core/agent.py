"""Lean agent runtime inspired by HICA with hybrid planning/decision flow."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Literal, Type, Union, Tuple, Awaitable
from uuid import uuid4
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field, ValidationError, model_validator, ConfigDict
from structlog.contextvars import clear_contextvars

from .core import Message, Thread
from .events import Event, EventType, StreamCallback
from .llm import LLMClient, RetryPolicy
from .logging import logger
from .observability import Observability
from .tools import ToolRegistry, ToolResult


class AgentConfig(BaseModel):
    """Minimal configuration for the MiniAgent runtime."""

    provider: str = "openai"
    model: Optional[str] = "gpt-4.1"
    api_key: Optional[str] = None
    name: Optional[str] = None
    system_prompt: str = (
        "You are a capable assistant that follows the shared plan, decides when to use tools, "
        "and provides clear user-facing answers."
    )
    temperature: float = 0.0
    max_tokens: int = 1024
    max_events_before_summarization: Optional[int] = 20
    planning_enabled: bool = True
    stream_by_default: bool = False
    tool_timeout: Optional[float] = 30.0
    llm_timeout: Optional[float] = 60.0
    planning_required: bool = False
    retry_policy: Optional[RetryPolicy] = None

    @model_validator(mode="after")
    def _validate(self) -> "AgentConfig":
        self.provider = self.provider.lower()
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.tool_timeout is not None and self.tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive when set")
        if self.llm_timeout is not None and self.llm_timeout <= 0:
            raise ValueError("llm_timeout must be positive when set")
        return self


class FinalResponseModel(BaseModel):
    message: Union[str, Dict[str, Any]] = Field(
        ..., description="Final answer for the user (string or structured)"
    )
    summary: Optional[Any] = Field(None, description="Optional short summary (string or structured data)")


class PlanStepModel(BaseModel):
    description: str = Field(..., description="Step description")
    tool: Optional[str] = Field(default=None, description="Recommended tool name or 'none'")
    success: Optional[str] = Field(default=None, description="Success criteria")


class PlanResponseModel(BaseModel):
    steps: List[PlanStepModel]

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        for idx, step in enumerate(self.steps, start=1):
            tool = (step.tool or "").strip()
            suffix = f" (Tool: {tool})" if tool and tool.lower() != "none" else ""
            lines.append(f"{idx}. {step.description.strip()}{suffix}")
        return lines


class ToolCallModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    id: Optional[str] = None
    arguments: Union[Dict[str, Any], str, None] = None
    note: Optional[str] = None


@dataclass
class ActionResult:
    action: Literal["tool", "final", "clarification", "fallback"]
    message: Optional[str] = None
    summary: Optional[Any] = None
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None


@dataclass
class PlanState:
    summary: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    index: int = 0
    parser: Optional[str] = None
    raw: Optional[Any] = None
    skips: Dict[int, str] = field(default_factory=dict)
    message_ts: Optional[str] = None


@dataclass
class TurnState:
    current_user_message_ts: Optional[str] = None
    last_request_id: Optional[str] = None
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    fallback_scheduled: List[str] = field(default_factory=list)
    final_response_message: Optional[str] = None
    final_response_summary: Optional[Any] = None
    plan: PlanState = field(default_factory=PlanState)


@dataclass
class DecisionResult:
    tool_calls: List[Dict[str, Any]]
    final_text: Optional[str]
    metrics: Dict[str, Any]
    ttft_ms: Optional[int] = None


class Agent:
    """Lean agent that selects tools using structured LLM outputs."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[ToolRegistry] = None,
        callbacks: Optional[StreamCallback] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self._apply_env_defaults()
        self.name = self.config.name or "MiniAgent"
        self.tool_registry = tools or ToolRegistry()
        self.llm = LLMClient(
            provider=self.config.provider,
            api_key=self.config.api_key,
            model=self.config.model,
            retry_policy=self.config.retry_policy,
        )
        self.callbacks = callbacks or StreamCallback()
        self.observability = Observability(self.callbacks)
        if self.offline:
            logger.warning(
                "LLM client not configured; agent will return fallback responses"
            )
        self.metrics: Counter[str] = Counter()
        if os.getenv("MINIAGENT_EVENT_LOG", "0") == "1":
            self.callbacks.on_any(self._log_event_telemetry)
        # Store turn states as runtime attributes, not in metadata
        self._turn_states: WeakKeyDictionary[Thread, TurnState] = WeakKeyDictionary()

    @staticmethod
    def _monotonic_ns() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def _elapsed_ms(start_ns: int, *, end_ns: Optional[int] = None) -> int:
        """Convert perf_counter_ns delta to milliseconds with a 1ms floor."""
        stop_ns = end_ns if end_ns is not None else Agent._monotonic_ns()
        return max(1, int(max(0, stop_ns - start_ns) / 1_000_000))

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if not lines:
            return text
        if not lines[0].startswith("```"):
            return text
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "```":
                return "\n".join(lines[1:idx]).strip()
        return "\n".join(lines[1:]).strip()

    @property
    def offline(self) -> bool:
        return self.llm.client is None

    def _get_turn_state(self, thread: Thread) -> TurnState:
        # Check runtime attribute first
        if thread in self._turn_states:
            return self._turn_states[thread]

        # Reconstruct from persisted metadata
        plan_state = PlanState(
            summary=thread.metadata.get("plan"),
            steps=list(thread.metadata.get("plan_steps", []) or []),
            index=thread.metadata.get("plan_index", 0),
            parser=thread.metadata.get("plan_parser"),
            raw=thread.metadata.get("plan_raw"),
            skips=dict(thread.metadata.get("plan_skips", {}) or {}),
            message_ts=thread.metadata.get("plan_message_ts"),
        )
        state = TurnState(
            current_user_message_ts=thread.metadata.get("current_user_message_ts"),
            last_request_id=thread.metadata.get("last_request_id"),
            pending_tool_calls=list(thread.metadata.get("pending_tool_calls", []) or []),
            fallback_scheduled=list(thread.metadata.get("fallback_scheduled", []) or []),
            final_response_message=thread.metadata.get("final_response_message"),
            final_response_summary=thread.metadata.get("final_response_summary"),
            plan=plan_state,
        )
        # Store in runtime cache
        self._turn_states[thread] = state
        return state

    def _save_turn_state(self, thread: Thread, state: TurnState) -> None:
        # Store in runtime cache
        self._turn_states[thread] = state
        
        # Helper to set or remove metadata
        def _set_or_remove(key: str, value: Any) -> None:
            if value is not None and (not isinstance(value, (list, dict)) or value):
                thread.metadata[key] = value
            else:
                thread.metadata.pop(key, None)
        
        # Persist JSON-friendly fields only
        _set_or_remove("current_user_message_ts", state.current_user_message_ts)
        _set_or_remove("last_request_id", state.last_request_id)
        _set_or_remove("pending_tool_calls", list(state.pending_tool_calls) if state.pending_tool_calls else None)
        _set_or_remove("fallback_scheduled", list(state.fallback_scheduled) if state.fallback_scheduled else None)
        _set_or_remove("final_response_message", state.final_response_message)
        _set_or_remove("final_response_summary", state.final_response_summary)
        
        # Plan fields
        plan = state.plan
        _set_or_remove("plan", plan.summary)
        _set_or_remove("plan_steps", list(plan.steps) if plan.steps else None)
        thread.metadata["plan_index"] = plan.index  # Always keep index
        _set_or_remove("plan_parser", plan.parser)
        _set_or_remove("plan_raw", plan.raw)
        _set_or_remove("plan_skips", dict(plan.skips) if plan.skips else None)
        _set_or_remove("plan_message_ts", plan.message_ts)

    def _set_final_response(
        self,
        thread: Thread,
        message: Optional[str],
        summary: Optional[Any],
        *,
        state: Optional[TurnState] = None,
    ) -> TurnState:
        state = state or self._get_turn_state(thread)
        state.final_response_message = message
        state.final_response_summary = summary
        self._save_turn_state(thread, state)
        return state

    def metrics_snapshot(self) -> Dict[str, int]:
        """Return a shallow copy of collected metrics for external reporting."""
        return dict(self.metrics)

    def _log_event_telemetry(self, event: Event) -> None:
        """Optional event sink for debugging/observability."""
        content = event.content
        if isinstance(content, str) and len(content) > 500:
            content = f"{content[:500]}…"
        logger.debug(
            "agent.telemetry.event",
            event_type=event.type.value,
            metadata=event.metadata,
            content=content,
        )

    @staticmethod
    def _new_event_id() -> str:
        """Generate an identifier for the most granular telemetry event."""
        return Observability.new_event_id()

    def _augment_llm_metrics(
        self,
        base: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        *,
        default_model: Optional[str] = None,
        ttft_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.observability.augment_llm_metrics(
            base,
            metrics,
            default_model=default_model,
            ttft_ms=ttft_ms,
        )

    async def _record_llm_error(
        self,
        exc: Exception,
        *,
        phase: Optional[str],
        start_ns: int,
        start_ms: int,
        thread: Optional[Thread],
        event_id: str,
    ) -> int:
        duration_ms = self._elapsed_ms(start_ns)
        payload: Dict[str, Any] = {
            "error": str(exc),
            "latency_ms": duration_ms,
            "start_ms": start_ms,
            "end_ms": start_ms + duration_ms,
        }
        if phase:
            payload["phase"] = phase
        await self.observability.emit(
            EventType.LLM_ERROR,
            payload,
            thread=thread,
            event_id=event_id,
        )
        self.metrics["llm_error_total"] += 1
        return duration_ms

    def _schedule_web_search_fallback(
        self,
        thread: Thread,
        topic: Optional[str],
        *,
        request_id: Optional[str] = None,
    ) -> bool:
        registry = getattr(self.tool_registry, "tools", {}) or {}
        if "web_search" not in registry:
            return False
        state = self._get_turn_state(thread)
        pending = state.pending_tool_calls
        for call in pending:
            if call.get("name") == "web_search":
                return False
        fallback_markers = state.fallback_scheduled
        # More specific marker including query hash
        query = (topic or "").strip() or thread.get_last_user_message() or ""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        marker = f"{state.current_user_message_ts or ''}|web_search|{query_hash}"
        if marker in fallback_markers:
            return False
        query = (topic or "").strip()
        if not query:
            last_user = next((m.content for m in reversed(thread.messages) if m.role == "user"), "")
            query = last_user.strip()
        if not query:
            return False
        fallback_call = {
            "id": uuid4().hex[:8],
            "name": "web_search",
            "arguments": {"query": query},
            "note": "Fallback search after empty knowledge_base result",
            "from_llm": False,
            "source": "fallback_web_search",
            "event_id": None,
        }
        fallback_call["event_id"] = fallback_call["id"]
        self._queue_tool_calls(thread, [fallback_call])
        fallback_markers.append(marker)
        self._save_turn_state(thread, state)
        logger.info(
            "agent.tool.fallback_scheduled",
            session_id=thread.id,
            from_tool="knowledge_base",
            fallback_tool="web_search",
            query=query,
            request_id=request_id,
        )
        return True

    def _prepare_new_turn(
        self, thread: Thread, message: Message, *, request_id: Optional[str] = None
    ) -> None:
        """Reset per-turn state such as planning metadata."""
        state = self._get_turn_state(thread)
        last_planned_ts = state.plan.message_ts
        current_ts = message.timestamp.isoformat()
        state.current_user_message_ts = current_ts
        state.last_request_id = request_id
        # Clear pending tool calls from previous turn
        state.pending_tool_calls.clear()
        if not state.plan.summary or last_planned_ts != current_ts:
            self._reset_plan_state(thread, state=state)
        state.final_response_message = None
        state.final_response_summary = None
        self._save_turn_state(thread, state)
        thread.metadata.pop("final_response_payload", None)

    def _reset_plan_state(self, thread: Thread, *, state: Optional[TurnState] = None) -> None:
        """Clear plan metadata so a fresh plan can be generated."""
        state = state or self._get_turn_state(thread)
        state.plan = PlanState()
        state.fallback_scheduled.clear()
        self._save_turn_state(thread, state)

    async def _await_with_timeout(
        self,
        coro: Awaitable[Any],
        timeout: Optional[float],
        timeout_message: str,
    ) -> Any:
        if timeout is None:
            return await coro
        
        # Create explicit task for better cancellation control
        task = asyncio.create_task(coro)
        try:
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            # Ensure task is cancelled properly
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise TimeoutError(timeout_message) from None

    async def run(
        self,
        user_input: str,
        thread: Optional[Thread] = None,
        context: Optional[str] = None,
        stream: Optional[bool] = None,
        max_iterations: int = 5,
    ) -> str:
        thread = thread or Thread()
        if stream is None:
            stream = self.config.stream_by_default
        request_id = uuid4().hex[:8]
        session_id = thread.id
        base_logger = logger.bind(
            request_id=request_id,
            session_id=session_id,
        )
        try:
            with self.observability.session_scope(session_id=session_id):
                with self.observability.request_scope(request_id):
                    agent_run_event = self._new_event_id()
                    base_logger.bind(event_id=agent_run_event).info(
                        "agent.run.start",
                        user_input=user_input,
                        stream=stream,
                    )
                    message = Message("user", user_input, metadata={"request_id": request_id})
                    thread.add_message(message)
                    self._prepare_new_turn(thread, message, request_id=request_id)
                    await self.observability.emit(
                        EventType.AGENT_START,
                        {"session_id": session_id, "user_input": user_input},
                        thread=thread,
                        event_id=agent_run_event,
                    )

                    async for _ in self.agent_loop(
                        thread,
                        context=context,
                        stream=stream,
                        max_iterations=max_iterations,
                        request_id=request_id,
                    ):
                        pass

                    state = self._get_turn_state(thread)
                    final_message = state.final_response_message
                    if isinstance(final_message, str):
                        return final_message

                    for event in reversed(thread.events):
                        if event.type == EventType.AGENT_RESPONSE and isinstance(event.data, str):
                            self._set_final_response(
                                thread,
                                event.data,
                                None,
                                state=state,
                            )
                            return event.data

                    fallback = "I'm sorry, I couldn't produce an answer this time."
                    self._set_final_response(
                        thread,
                        fallback,
                        None,
                        state=state,
                    )
                    await self.observability.emit(
                        EventType.AGENT_RESPONSE,
                        fallback,
                        thread=thread,
                    )
                    logger.warning(
                        "agent.final_response_missing",
                        session_id=thread.id,
                        event_count=len(thread.events),
                    )
                    return fallback
        except Exception as exc:
            base_logger.error(
                "agent.run.exception",
                error=str(exc),
                exc_type=type(exc).__name__,
            )
            logger.exception("agent.run.exception_trace")
            await self.observability.emit(
                EventType.ERROR,
                {"error": str(exc)},
                thread=thread,
            )
            if isinstance(exc, RuntimeError):
                fallback = f"Configuration issue: {exc}"
            else:
                fallback = "I'm sorry, I ran into an internal error while responding."
            self._set_final_response(thread, fallback, None)
            await self.observability.emit(
                EventType.AGENT_RESPONSE,
                fallback,
                thread=thread,
            )
            return fallback
        finally:
            clear_contextvars()

    async def agent_loop(
        self,
        thread: Thread,
        context: Optional[str] = None,
        stream: Optional[bool] = None,
        max_iterations: int = 5,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Thread, None]:
        if (
            self.config.max_events_before_summarization
            and len(thread.events) > self.config.max_events_before_summarization
        ):
            thread.summarize_events(self.config.max_events_before_summarization)

        if self.offline:
            await self._emit_offline_response(thread)
            yield thread
            return

        await self._ensure_plan(thread, context, request_id=request_id)
        yield thread

        use_stream = bool(stream)
        state = self._get_turn_state(thread)
        state.last_request_id = request_id
        self._save_turn_state(thread, state)

        for iteration in range(max_iterations):
            action = await self._decide_next_action(
                thread, context, iteration, use_stream, request_id=request_id
            )

            if action.action == "tool" and action.tool:
                if action.message:
                    await self.observability.emit(
                        EventType.AGENT_THINKING,
                        {"tool": action.tool, "note": action.message},
                        thread=thread,
                    )
                await self._execute_tool(
                    action.tool,
                    action.arguments or {},
                    thread,
                    call_id=action.tool_call_id,
                    request_id=request_id,
                )
                yield thread
                continue

            if action.action == "clarification" and action.message:
                clarification_msg = action.message
                thread.add_message(Message("assistant", clarification_msg))
                await self.observability.emit(
                    EventType.AGENT_RESPONSE,
                    clarification_msg,
                    thread=thread,
                    extra_metadata={"intent": "clarification"},
                )
                logger.info(
                    "agent.clarification_requested",
                    message=clarification_msg,
                    session_id=thread.id,
                )
                yield thread
                return

            if action.action == "final" and action.message:
                await self._commit_final_response(
                    action.message, action.summary, thread, request_id=request_id
                )
                yield thread
                return

            if action.action == "fallback":
                # If we have a message from the decision phase, use it directly
                if action.message:
                    await self._commit_final_response(
                        action.message, None, thread, request_id=request_id
                    )
                else:
                    await self._finalize_response(
                        thread,
                        context,
                        request_id=request_id,
                    )
                yield thread
                return

            logger.warning(
                "agent.unhandled_action",
                action=action.action,
                session_id=thread.id,
            )
            await self._finalize_response(thread, context, request_id=request_id)
            yield thread
            return

        logger.warning("Max iterations reached without completion", session_id=thread.id)
        await self._finalize_response(thread, context, request_id=request_id)
        yield thread

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _llm_json_call(
        self,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel],
        thread: Optional[Thread] = None,
        request_id: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> BaseModel:
        start_wall_ms = self._now_ms()
        llm_event = event_id or self._new_event_id()
        await self.observability.emit(
            EventType.LLM_CALL_START,
            {"messages": len(messages)},
            thread=thread,
            event_id=llm_event,
        )

        logger.debug(
            "agent.llm.request",
            message_count=len(messages),
            last_user_message=next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                None,
            ),
            session_id=thread.id if thread else None,
            event_id=llm_event,
        )

        start_ns = self._monotonic_ns()
        try:
            raw = await self._await_with_timeout(
                self.llm.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=False,
                    tools=None,
                ),
                self.config.llm_timeout,
                "LLM request timed out",
            )
        except Exception as exc:
            await self._record_llm_error(
                exc,
                phase=None,
                start_ns=start_ns,
                start_ms=start_wall_ms,
                thread=thread,
                event_id=llm_event,
            )
            raise

        if isinstance(raw, dict):
            payload = json.dumps(raw)
        else:
            payload = str(raw)

        payload = self._strip_markdown_code_fence(payload.strip())
        if not payload:
            raise ValueError("Empty response from LLM")

        logger.debug(
            "agent.llm.raw_response",
            payload_preview=payload[:500],
            payload_length=len(payload),
            session_id=thread.id if thread else None,
            event_id=llm_event,
        )

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON", error=str(exc), payload=payload)
            raise

        try:
            validated = response_model.model_validate(data)
        except ValidationError as exc:
            logger.error("Response validation failed", error=str(exc), payload=data)
            raise

        duration_ms = self._elapsed_ms(start_ns)
        metrics_payload = self.llm.consume_last_metrics()
        llm_data: Dict[str, Any] = {
            "latency_ms": duration_ms,
            "start_ms": start_wall_ms,
            "end_ms": start_wall_ms + duration_ms,
        }
        llm_data = self._augment_llm_metrics(
            llm_data,
            metrics_payload,
            default_model=self.config.model or response_model.__name__,
        )
        await self.observability.emit(
            EventType.LLM_CALL_END,
            llm_data,
            thread=thread,
            event_id=llm_event,
        )

        logger.debug(
            "agent.llm.validated_response",
            model=llm_data.get("model", response_model.__name__),
            latency_ms=duration_ms,
            session_id=thread.id if thread else None,
            event_id=llm_event,
        )
        self.metrics["llm_call_total"] += 1
        self.metrics["llm_call_duration_ms_total"] += duration_ms
        return validated

    async def _decide_next_action(
        self,
        thread: Thread,
        context: Optional[str],
        iteration: int,
        stream: bool,
        request_id: Optional[str] = None,
    ) -> ActionResult:
        pending_call = self._pop_next_tool_call(thread)
        if pending_call:
            if not pending_call.get("from_llm", False):
                arguments_obj = pending_call.get("arguments", {})
                if isinstance(arguments_obj, str):
                    arguments_str = arguments_obj
                else:
                    try:
                        arguments_str = json.dumps(arguments_obj, ensure_ascii=False)
                    except (TypeError, ValueError):
                        arguments_str = str(arguments_obj)
                tool_name = pending_call.get("name") or "unknown_tool"
                assistant_message = Message(
                    "assistant",
                    content="",
                    metadata={
                        "internal": True,
                        "type": "tool_call",
                        "include_in_prompt": True,
                        "source": pending_call.get("source", "pending_queue"),
                        "tools": [tool_name],
                    },
                    tool_calls=[
                        {
                            "id": pending_call.get("id"),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments_str,
                            },
                        }
                    ],
                )
                thread.add_message(assistant_message)
            logger.info(
                "agent.intent.selected",
                action="tool",
                tool=pending_call["name"],
                session_id=thread.id,
                source="pending_queue",
            )
            await self.observability.emit(
                EventType.TOOL_SELECTION,
                {
                    "tool": pending_call["name"],
                    "arguments": pending_call["arguments"],
                    "source": "pending_queue",
                },
                thread=thread,
                event_id=pending_call.get("event_id") or pending_call.get("id"),
            )
            return ActionResult(
                action="tool",
                tool=pending_call["name"],
                arguments=pending_call["arguments"],
                tool_call_id=pending_call["id"],
                message=pending_call.get("note"),
            )

        instruction = (
            "Review the conversation and the current plan. "
            "Decide whether to call a tool or respond directly. "
            "Use the available functions when they help you fulfill the request. "
            "Otherwise, respond to the user in natural language."
        )

        messages = self._build_messages(
            instruction, thread, context, iteration, mode="decision"
        )
        decision_event = self._new_event_id()
        await self.observability.emit(
            EventType.LLM_CALL_START,
            {"messages": len(messages), "phase": "decision"},
            thread=thread,
            event_id=decision_event,
        )
        logger.debug(
            "agent.decision.request",
            session_id=thread.id,
            iteration=iteration,
            event_id=decision_event,
        )

        tool_schemas = (
            self.tool_registry.get_schemas() if self.tool_registry.tools else None
        )

        decision_result = await self._gather_decision_result(
            messages,
            tool_schemas,
            thread,
            stream=stream,
            decision_event=decision_event,
            request_id=request_id,
        )

        metrics_payload = self.llm.consume_last_metrics()
        llm_data = self._augment_llm_metrics(
            dict(decision_result.metrics),
            metrics_payload,
            default_model=self.config.model,
            ttft_ms=decision_result.ttft_ms,
        )
        llm_data.setdefault("start_ms", decision_result.metrics.get("start_ms"))
        if llm_data.get("start_ms") is not None:
            llm_data["end_ms"] = llm_data["start_ms"] + decision_result.metrics.get("latency_ms", 0)
        else:
            llm_data.pop("end_ms", None)
        await self.observability.emit(
            EventType.LLM_CALL_END,
            llm_data,
            thread=thread,
            event_id=decision_event,
        )
        self.metrics["llm_call_total"] += 1
        latency_value = decision_result.metrics.get("latency_ms") or 0
        self.metrics["llm_call_duration_ms_total"] += int(latency_value)

        tool_calls_payload = decision_result.tool_calls
        if tool_calls_payload:
            formatted_calls, pending_calls = self._prepare_tool_calls(tool_calls_payload)
            if pending_calls:
                assistant_message = Message(
                    "assistant",
                    content="",
                    metadata={
                        "internal": True,
                        "type": "tool_call",
                        "include_in_prompt": True,
                        "tools": [call["name"] for call in pending_calls],
                    },
                    tool_calls=formatted_calls,
                )
                thread.add_message(assistant_message)
                self._queue_tool_calls(thread, pending_calls)
                next_call = self._pop_next_tool_call(thread)
                if next_call:
                    logger.info(
                        "agent.intent.selected",
                        action="tool",
                        tool=next_call["name"],
                        session_id=thread.id,
                    )
                    await self.observability.emit(
                        EventType.TOOL_SELECTION,
                        {
                            "tool": next_call["name"],
                            "arguments": next_call["arguments"],
                            "tool_call_id": next_call["id"],
                        },
                        thread=thread,
                        event_id=next_call.get("id"),
                    )
                    return ActionResult(
                        action="tool",
                        tool=next_call["name"],
                        arguments=next_call["arguments"],
                        tool_call_id=next_call["id"],
                    )

        action_result = self._parse_final_action(decision_result.final_text)
        
        # If the LLM returned plain text (not JSON), treat it as a final response
        if action_result.action == "final" and action_result.message:
            # Check if this is actually unstructured text by seeing if it contains newlines or markdown
            text = str(action_result.message)
            if text and ("\n" in text or "**" in text or text.startswith("1.") or text.startswith("-")):
                logger.warning(
                    "agent.decision.unstructured_response",
                    session_id=thread.id,
                    iteration=iteration,
                )
                # Return as fallback to trigger finalization with this content
                return ActionResult(
                    action="fallback",
                    message=text,
                    summary=None,
                )
        
        if action_result.action == "fallback":
            logger.warning(
                "agent.decision.unstructured_response",
                session_id=thread.id,
                iteration=iteration,
            )
        else:
            await self.observability.emit(
                EventType.LLM_RESPONSE,
                {
                    "action": action_result.action,
                    "message": action_result.message,
                    "summary": action_result.summary,
                },
                thread=thread,
                event_id=decision_event,
            )
        logger.info(
            "agent.intent.selected",
            action=action_result.action,
            tool=None,
            session_id=thread.id,
            event_id=decision_event,
        )
        return action_result

    async def _stream_decision_response(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        thread: Thread,
        request_id: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> DecisionResult:
        stream_event = event_id or self._new_event_id()
        start_ns = self._monotonic_ns()
        start_ms = self._now_ms()
        ttft_ms: Optional[int] = None
        stream_iter = await self._await_with_timeout(
            self.llm.complete(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                tools=tool_schemas,
            ),
            self.config.llm_timeout,
            "LLM streaming request timed out",
        )

        chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        stream_started = False

        async for chunk in stream_iter:
            if isinstance(chunk, str):
                if not stream_started:
                    stream_started = True
                    ttft_ms = self._elapsed_ms(start_ns)
                    await self.observability.emit(
                        EventType.STREAM_START,
                        None,
                        event_id=stream_event,
                    )
                chunks.append(chunk)
                await self.observability.emit(
                    EventType.STREAM_CHUNK,
                    chunk,
                    event_id=stream_event,
                )
            elif isinstance(chunk, dict):
                extracted = self._extract_tool_calls(chunk)
                if extracted:
                    tool_calls.extend(extracted)
            else:
                logger.debug(
                    "agent.decision.stream.unknown_chunk",
                    chunk_type=type(chunk).__name__,
                    session_id=thread.id,
                    event_id=stream_event,
                )

        if stream_started:
            await self.observability.emit(
                EventType.STREAM_END,
                None,
                event_id=stream_event,
            )

        final_text = "".join(chunks)
        if stream_started and ttft_ms is None:
            ttft_ms = self._elapsed_ms(start_ns)
        if final_text:
            logger.debug(
                "agent.decision.stream.final_text_preview",
                preview=final_text[:200],
                session_id=thread.id,
                event_id=stream_event,
            )

        latency_ms = self._elapsed_ms(start_ns)
        end_ms = start_ms + latency_ms
        metrics: Dict[str, Any] = {
            "phase": "decision",
            "latency_ms": latency_ms,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        if ttft_ms is not None:
            metrics["ttft_ms"] = ttft_ms
        return DecisionResult(
            tool_calls=tool_calls,
            final_text=final_text,
            metrics=metrics,
            ttft_ms=ttft_ms,
        )

    async def _gather_decision_result(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        thread: Thread,
        *,
        stream: bool,
        decision_event: str,
        request_id: Optional[str] = None,
    ) -> DecisionResult:
        start_ns = self._monotonic_ns()
        start_ms = self._now_ms()
        if stream:
            try:
                result = await self._stream_decision_response(
                    messages,
                    tool_schemas,
                    thread,
                    request_id=request_id,
                    event_id=decision_event,
                )
            except Exception as exc:
                await self._record_llm_error(
                    exc,
                    phase="decision",
                    start_ns=start_ns,
                    start_ms=start_ms,
                    thread=thread,
                    event_id=decision_event,
                )
                raise

            latency_ms = result.metrics.get("latency_ms")
            if latency_ms is None:
                latency_ms = self._elapsed_ms(start_ns)
                result.metrics["latency_ms"] = latency_ms
            result.metrics.setdefault("phase", "decision")
            result.metrics.setdefault("start_ms", start_ms)
            result.metrics.setdefault("end_ms", result.metrics["start_ms"] + latency_ms)
            return result

        try:
            raw_response = await self._await_with_timeout(
                self.llm.complete(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=False,
                    tools=tool_schemas,
                ),
                self.config.llm_timeout,
                "LLM decision timed out",
            )
        except Exception as exc:
            await self._record_llm_error(
                exc,
                phase="decision",
                start_ns=start_ns,
                start_ms=start_ms,
                thread=thread,
                event_id=decision_event,
            )
            raise

        logger.debug(
            "agent.decision.raw_response",
            session_id=thread.id,
            response=str(raw_response)[:500],
            event_id=decision_event,
        )
        tool_calls_payload = self._extract_tool_calls(raw_response)
        final_text = self._extract_text_response(raw_response)
        latency_ms = self._elapsed_ms(start_ns)
        metrics: Dict[str, Any] = {
            "phase": "decision",
            "latency_ms": latency_ms,
            "start_ms": start_ms,
            "end_ms": start_ms + latency_ms,
        }
        return DecisionResult(
            tool_calls=tool_calls_payload or [],
            final_text=final_text,
            metrics=metrics,
        )

    def _parse_final_action(self, final_text: Any) -> ActionResult:
        if final_text is None:
            return ActionResult(
                action="fallback",
                message="I'm sorry, I wasn't able to generate a response just now.",
            )

        raw_payload: Optional[Dict[str, Any]] = None
        clarification_requested = False
        serialized_text = ""

        if isinstance(final_text, FinalResponseModel):
            raw_payload = final_text.model_dump(exclude_none=True)
        elif isinstance(final_text, dict):
            raw_payload = final_text
        else:
            serialized_text = str(final_text).strip()
            if not serialized_text:
                return ActionResult(
                    action="fallback",
                    message="I'm sorry, I wasn't able to generate a response just now.",
                )

            serialized_text = self._strip_markdown_code_fence(serialized_text)

            try:
                raw_payload = json.loads(serialized_text)
            except json.JSONDecodeError:
                # Fallback to regex for embedded JSON
                json_match = re.search(
                    r"```(?:json)?\s*(\{.*\})\s*```", str(final_text).strip(), re.DOTALL
                )
                if json_match:
                    try:
                        raw_payload = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        raw_payload = None

        if isinstance(raw_payload, dict):
            clarification_requested = bool(raw_payload.get("clarification"))

        if raw_payload is None:
            # If JSON parsing failed completely, treat the original text as the message
            if serialized_text and isinstance(serialized_text, str):
                return ActionResult(
                    action="final",
                    message=serialized_text,
                    summary=None,
                )
            return ActionResult(
                action="fallback",
                message="",
            )

        try:
            validated = FinalResponseModel.model_validate(raw_payload)
        except ValidationError:
            return ActionResult(action="fallback", message="")

        message_content = validated.message
        if isinstance(message_content, dict):
            message_content = json.dumps(message_content, ensure_ascii=False)

        if clarification_requested:
            return ActionResult(
                action="clarification",
                message=message_content,
                summary=validated.summary,
            )

        return ActionResult(
            action="final",
            message=message_content,
            summary=validated.summary,
        )

    def _extract_tool_calls(self, raw_response: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_response, dict):
            # Direct tool_calls field
            tool_calls = raw_response.get("tool_calls")
            if isinstance(tool_calls, list):
                return tool_calls
            
            # OpenAI-like envelope format
            choices = raw_response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message")
                if isinstance(message, dict):
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list):
                        return tool_calls
            
            # Type-based format
            if raw_response.get("type") == "tool_calls":
                tool_calls = raw_response.get("tool_calls")
                if isinstance(tool_calls, list):
                    return tool_calls
        return []

    def _prepare_tool_calls(
        self, raw_calls: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        formatted_for_messages: List[Dict[str, Any]] = []
        pending_calls: List[Dict[str, Any]] = []
        invalid_count = 0

        for raw_call in raw_calls:
            try:
                call = ToolCallModel.model_validate(raw_call)
            except ValidationError:
                invalid_count += 1
                continue

            name = call.name
            call_id = call.id or f"tool_call_{uuid4().hex[:8]}"
            raw_arguments: Any = call.arguments if call.arguments is not None else {}

            arguments_dict = self._parse_tool_arguments(raw_arguments, name, call_id)
            if isinstance(raw_arguments, str):
                arguments_str = raw_arguments
            else:
                try:
                    arguments_str = json.dumps(raw_arguments, ensure_ascii=False)
                except (TypeError, ValueError):
                    logger.warning(
                        "agent.tool_call.arguments_serialization_failed",
                        tool=name,
                        call_id=call_id,
                        arg_type=type(raw_arguments).__name__,
                    )
                    arguments_str = "{}"

            formatted_for_messages.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_str,
                    },
                }
            )
            pending_calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments_dict,
                    "event_id": call_id,
                    "note": call.note,
                    "from_llm": True,
                    "source": "llm_tool_call",
                }
            )

        # Report invalid count once if any
        if invalid_count > 0:
            logger.debug(
                "agent.tool_call.invalid_calls",
                invalid_count=invalid_count,
                total_calls=len(raw_calls),
            )

        return formatted_for_messages, pending_calls

    def _parse_tool_arguments(
        self, raw_arguments: Any, tool_name: str, call_id: str
    ) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            text = raw_arguments.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                logger.error(
                    "agent.tool.args_parse_failed",
                    tool=tool_name,
                    call_id=call_id,
                    arguments=text,
                    error=str(exc),
                )
                return {}
        logger.warning(
            "agent.tool.args_unexpected_type",
            tool=tool_name,
            call_id=call_id,
            arg_type=type(raw_arguments).__name__,
        )
        return {}

    def _queue_tool_calls(self, thread: Thread, tool_calls: List[Dict[str, Any]]) -> None:
        if not tool_calls:
            return
        state = self._get_turn_state(thread)
        state.pending_tool_calls.extend(tool_calls)
        self._save_turn_state(thread, state)

    def _pop_next_tool_call(self, thread: Thread) -> Optional[Dict[str, Any]]:
        state = self._get_turn_state(thread)
        if not state.pending_tool_calls:
            return None
        next_call = state.pending_tool_calls.pop(0)
        self._save_turn_state(thread, state)
        return next_call

    def _extract_text_response(self, raw_response: Any) -> Optional[str]:
        if raw_response is None:
            return None
        if isinstance(raw_response, str):
            return raw_response
        if isinstance(raw_response, dict):
            # Look for known text fields
            for key in ("message", "content", "text", "response"):
                value = raw_response.get(key)
                if isinstance(value, str):
                    return value
            # For dicts without clear text fields, return None to avoid ambiguity
            return None
        # Non-dict, non-string types
        return None

    async def _finalize_response(
        self,
        thread: Thread,
        context: Optional[str],
        request_id: Optional[str] = None,
        fallback_message: Optional[str] = None,
    ) -> None:
        instruction = (
            "Summarize the conversation and answer the user's request. "
            "Respond as JSON with keys 'message' and optional 'summary'."
        )
        state = self._get_turn_state(thread)
        pending_calls: List[Dict[str, Any]] = list(state.pending_tool_calls)
        state.pending_tool_calls.clear()
        if pending_calls:
            logger.warning(
                "agent.pending_tool_calls.flushed",
                session_id=thread.id,
                pending=len(pending_calls),
            )
            for call in pending_calls:
                tool_name = call.get("name") or "unknown"
                call_id = call.get("id")
                failure_note = {
                    "error": "tool_call_incomplete",
                    "message": "Tool call skipped because the agent finalized the response before completion.",
                }
                thread.add_message(
                    Message(
                        "tool",
                        json.dumps(failure_note, ensure_ascii=False),
                        metadata={
                            "internal": True,
                            "type": "tool_result",
                            "tool": tool_name,
                            "success": False,
                        },
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )
                await self.observability.emit(
                    EventType.TOOL_ERROR,
                    {
                        "tool": tool_name,
                        "error": failure_note["error"],
                        "reason": failure_note["message"],
                    },
                    thread=thread,
                    event_id=call.get("event_id"),
                )

        self._save_turn_state(thread, state)

        plan_index = state.plan.index
        messages = self._build_messages(
            instruction, thread, context, plan_index, mode="final"
        )
        try:
            final_event = self._new_event_id()
            response = await self._llm_json_call(
                messages,
                FinalResponseModel,
                thread,
                request_id=request_id,
                event_id=final_event,
            )
        except Exception as exc:
            logger.error("Final response generation failed", error=str(exc))
            # Check if we have a fallback message from the decision phase
            if fallback_message and isinstance(fallback_message, str):
                # Use the actual content from the decision phase
                fallback = fallback_message
            else:
                fallback = "I'm sorry, I couldn't generate a response right now."
            
            if isinstance(fallback, dict):
                fallback = json.dumps(fallback, ensure_ascii=False)
            thread.add_message(Message("assistant", fallback))
            self._set_final_response(thread, fallback, None, state=state)
            await self.observability.emit(
                EventType.AGENT_RESPONSE,
                fallback,
                thread=thread,
                event_id=final_event,
            )
            return

        await self._commit_final_response(
            response.message, response.summary, thread, request_id=request_id, event_id=final_event
        )

    async def _execute_tool(
        self,
        intent: str,
        arguments: Dict[str, Any],
        thread: Thread,
        *,
        call_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        tool_event = call_id or self._new_event_id()
        start_wall_ms = self._now_ms()
        start_ns = self._monotonic_ns()
        start_payload = {"tool": intent, "arguments": arguments}
        await self.observability.emit(
            EventType.TOOL_EXECUTION_START,
            {**start_payload, "start_ms": start_wall_ms},
            thread=thread,
            event_id=tool_event,
        )

        await self.observability.emit(
            EventType.TOOL_EXECUTION,
            {"tool": intent, "parameters": arguments},
            thread=thread,
            event_id=tool_event,
        )

        logger.info(
            "agent.tool.execute",
            tool=intent,
            arguments=arguments,
            session_id=thread.id,
            event_id=tool_event,
        )

        duration_ms: Optional[int] = None
        try:
            result: ToolResult = await self._await_with_timeout(
                self.tool_registry.execute_tool(intent, arguments),
                self.config.tool_timeout,
                f"Tool '{intent}' execution timed out",
            )
        except Exception as exc:
            duration_ms = self._elapsed_ms(start_ns)
            await self.observability.emit(
                EventType.TOOL_ERROR,
                {
                    "tool": intent,
                    "error": str(exc),
                    "latency_ms": duration_ms,
                    "start_ms": start_wall_ms,
                    "end_ms": start_wall_ms + duration_ms,
                },
                thread=thread,
                event_id=tool_event,
            )
            result = ToolResult(success=False, error=str(exc))
            self.metrics["tool_error_total"] += 1
            self.metrics[f"tool_{intent}_error_total"] += 1
        else:
            duration_ms = self._elapsed_ms(start_ns)

        payload = result.to_dict()
        metadata_block = payload.setdefault("metadata", {})
        metadata_block["duration_ms"] = duration_ms
        metadata_block["start_ms"] = start_wall_ms
        metadata_block["end_ms"] = start_wall_ms + duration_ms
        metadata_block["event_id"] = tool_event
        has_signal = any(
            value
            for value in (
                payload.get("llm"),
                payload.get("display"),
                payload.get("data"),
            )
            if value not in (None, "", [], {})
        )
        # Assess tool result quality
        quality_label, quality_reason = self._assess_tool_quality(intent, result, has_signal)
        if quality_label:
            metadata_block["quality"] = quality_label
            if quality_reason:
                metadata_block["quality_reason"] = quality_reason
            logger.warning(
                "agent.tool.result.low_quality",
                tool=intent,
                call_id=call_id,
                event_id=tool_event,
                quality=quality_label,
                reason=quality_reason,
            )

        if (
            intent == "knowledge_base"
            and result.success
            and isinstance(result.data, list)
            and not result.data
        ):
            if self._schedule_web_search_fallback(
                thread,
                arguments.get("topic") if isinstance(arguments, dict) else None,
                request_id=request_id,
            ):
                metadata_block["fallback_scheduled"] = "web_search"
        await self.observability.emit(
            EventType.TOOL_RESULT,
            payload,
            thread=thread,
            event_id=tool_event,
        )
        logger.info(
            "agent.tool.result",
            tool=intent,
            success=result.success,
            duration_ms=duration_ms,
            session_id=thread.id,
            event_id=tool_event,
        )

        await self.observability.emit(
            EventType.TOOL_EXECUTION_END,
            {"tool": intent, "success": result.success, "duration_ms": duration_ms},
            thread=thread,
            event_id=tool_event,
        )
        self.metrics["tool_call_total"] += 1
        self.metrics["tool_call_duration_ms_total"] += duration_ms or 0
        self.metrics[f"tool_{intent}_call_total"] += 1
        if result.success:
            self.metrics["tool_success_total"] += 1
        else:
            self.metrics["tool_failure_total"] += 1
        summary = payload.get("llm") or payload.get("display") or payload.get("data")
        if isinstance(summary, (dict, list)):
            summary = json.dumps(summary)
        await self.observability.emit(
            EventType.AGENT_THINKING,
            {"tool": intent, "result": summary},
            thread=thread,
            event_id=tool_event,
        )
        logger.debug(
            "agent.tool.summary",
            tool=intent,
            summary_preview=summary[:200] if isinstance(summary, str) else summary,
            session_id=thread.id,
            event_id=tool_event,
        )
        if result.success:
            self._record_plan_progress(thread, reason=f"tool:{intent}")

        tool_content = (
            payload.get("llm") or payload.get("data") or payload.get("display") or payload
        )
        if not isinstance(tool_content, str):
            try:
                tool_content = json.dumps(tool_content, ensure_ascii=False)
            except (TypeError, ValueError):
                tool_content = str(tool_content)

        if call_id:
            tool_message_metadata = {
                "internal": True,
                "type": "tool_result",
                "tool": intent,
                "success": result.success,
            }
            thread.add_message(
                Message(
                    "tool",
                    tool_content,
                    metadata=tool_message_metadata,
                    tool_call_id=call_id,
                    name=intent,
                )
            )

    async def _commit_final_response(
        self,
        message: Union[str, Dict[str, Any]],
        summary: Optional[Any],
        thread: Thread,
        *,
        request_id: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {"message": message}
        if summary is not None:
            payload["summary"] = summary
        if isinstance(message, dict):
            visible_message = json.dumps(message, ensure_ascii=False)
        else:
            visible_message = str(message)

        state = self._get_turn_state(thread)
        state.pending_tool_calls.clear()
        state = self._set_final_response(
            thread,
            visible_message,
            summary,
            state=state,
        )
        # Store payload only in message metadata, not thread metadata

        message_metadata: Dict[str, Any] = {"response_payload": payload}
        if request_id:
            message_metadata["request_id"] = request_id

        thread.add_message(
            Message("assistant", visible_message, metadata=message_metadata)
        )
        await self.observability.emit(
            EventType.AGENT_RESPONSE,
            visible_message,
            thread=thread,
            event_id=event_id,
        )
        logger.info(
            "agent.final_response",
            message=visible_message,
            summary=summary,
            session_id=thread.id,
            event_id=event_id or self.observability.new_event_id(),
        )
        if summary is not None:
            await self.observability.emit(
                EventType.INFO,
                {"summary": summary},
                thread=thread,
                event_id=event_id,
            )
        plan_steps = state.plan.steps
        plan_index = state.plan.index
        if plan_steps and plan_index < len(plan_steps):
            for step in plan_steps[plan_index:]:
                position = step.get("position", plan_index + 1)
                self._note_plan_skip(thread, position, "final_response_emitted")
        self._complete_remaining_plan_steps(thread)

    def _build_messages(
        self,
        prompt: str,
        thread: Optional[Thread],
        context: Optional[str],
        iteration: int = 0,
        mode: str = "decision",
    ) -> List[Dict[str, Any]]:
        # Build complete system message content
        system_parts = [self.config.system_prompt]
        
        # Add tool metadata
        metadata = self._tool_metadata()
        if metadata:
            system_parts.append(f"Available tools:\n{metadata}")
        
        # Add context
        if context:
            system_parts.append(f"Context:\n{context}")
        
        # Add plan information if available
        if thread:
            state = self._get_turn_state(thread)
            plan_summary = state.plan.summary
            plan_steps = state.plan.steps
            plan_index = state.plan.index
            if plan_summary:
                if iteration == 0:
                    system_parts.append(
                        "Planned workflow (execute sequentially):\n"
                        f"{plan_summary}\n"
                        "Follow the steps in order. Think through each step internally, "
                        "call tools when required, and verify success criteria before moving on."
                    )
                elif plan_steps:
                    if plan_index < len(plan_steps):
                        current_step = plan_steps[plan_index]
                        expected_tool = (current_step.get("tool") or "").strip()
                        step_instructions = (
                            f"Current plan step {current_step.get('position', plan_index + 1)}: {current_step['description']}. "
                        )
                        if expected_tool and expected_tool.lower() not in {"none", ""}:
                            step_instructions += (
                                f"Use tool '{expected_tool}' if needed to satisfy this step. "
                            )
                        step_instructions += (
                            "Integrate prior tool results, reason carefully, and confirm the success criteria."
                        )
                        system_parts.append(step_instructions)
                    else:
                        system_parts.append(
                            "All planned steps have been attempted. Consolidate findings and prepare the final response."
                        )
        
        # Add mode-specific instructions
        if mode == "decision":
            system_parts.append(
                "Decision protocol:\n"
                "- Decide whether to continue the reasoning with a tool call or respond directly.\n"
                "- Call a tool when its output is required to answer confidently.\n"
                "- When no tool is needed, reply by emitting STRICT JSON with keys 'message' and optional 'summary'.\n"
                "- Include \"clarification\": true when you must ask the user for more information; place the question in 'message'.\n"
                "- Always keep the user-facing response concise and helpful."
            )
        elif mode == "final":
            system_parts.append(
                "Compose the final user-facing answer summarizing tool evidence. Respond as JSON with keys 'message', optional 'summary', and include 'clarification': true if you still need more information."
            )
        
        # Add the specific prompt/instruction
        system_parts.append(prompt)
        
        # Create single system message
        system_content = "\n\n".join(system_parts)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]
        
        # Add conversation history
        if thread:
            for message in thread.messages:
                meta = message.metadata or {}
                if meta.get("exclude_from_prompt"):
                    continue
                role = message.role
                content = message.content or ""
                message_payload: Dict[str, Any] = {"role": role, "content": content}
                
                if message.tool_calls:
                    message_payload["tool_calls"] = message.tool_calls
                if message.tool_call_id:
                    message_payload["tool_call_id"] = message.tool_call_id
                if message.name:
                    message_payload["name"] = message.name
                
                messages.append(message_payload)
        
        return messages

    def _tool_metadata(self) -> str:
        lines = []
        for tool in self.tool_registry.tools.values():
            schema = tool.to_function_schema()
            description = schema.get("description", "")
            lines.append(f"<tool> {schema.get('name')} : {description}")
        return "\n".join(lines)

    def _should_generate_plan(
        self,
        thread: Thread,
        user_message: str,
        context: Optional[str],
    ) -> bool:
        if not user_message:
            return False

        # Skip planning if no tools are registered
        if not self.tool_registry.tools:
            return False

        message_text = user_message.strip()
        if not message_text:
            return False

        # Multi-line requests often imply multi-step tasks
        if "\n" in message_text:
            return True

        word_count = len(message_text.split())
        if word_count >= 20:
            return True

        if message_text.count("?") >= 2:
            return True

        lowered = message_text.lower()
        plan_triggers = {
            "plan",
            "steps",
            "workflow",
            "outline",
            "strategy",
            "roadmap",
            "multi-step",
            "break down",
            "analysis",
        }
        if any(trigger in lowered for trigger in plan_triggers):
            return True

        if context:
            context_words = len(context.split())
            if context_words >= 30:
                return True

        # Default: skip plan for short, direct questions
        return False

    async def _ensure_plan(
        self, thread: Thread, context: Optional[str], request_id: Optional[str] = None
    ) -> None:
        if not self.config.planning_enabled:
            return

        state = self._get_turn_state(thread)
        current_ts = state.current_user_message_ts
        existing_plan_ts = state.plan.message_ts
        if state.plan.summary and existing_plan_ts == current_ts:
            return

        self._reset_plan_state(thread, state=state)

        last_user_message = thread.get_last_user_message() or ""
        if not self._should_generate_plan(thread, last_user_message, context):
            logger.debug(
                "agent.plan.skipped",
                session_id=thread.id,
                reason="heuristic",
            )
            return

        plan_event = self._new_event_id()
        await self.observability.emit(
            EventType.PLAN_GENERATING,
            {"status": "starting", "session_id": thread.id},
            thread=thread,
            event_id=plan_event,
        )

        available_tools: List[str] = []
        if hasattr(self.tool_registry, "tools") and isinstance(self.tool_registry.tools, dict):
            available_tools = sorted(self.tool_registry.tools.keys())

        tool_text = ", ".join(available_tools) if available_tools else "registered tools"

        planning_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a planning assistant for another agent. "
                    "Respond ONLY in JSON with the shape {\"steps\": [{\"description\": str, \"tool\": str|\"none\", \"success\": str}]}. "
                    "Break the user's request into short, sequential steps. Mention the tool to use for each step where applicable from: "
                    f"{tool_text}."
                ),
            }
        ]

        planning_messages.append({"role": "user", "content": last_user_message})
        if context:
            planning_messages.append({"role": "system", "content": f"Additional context: {context}"})

        logger.info(
            "agent.plan.request",
            session_id=thread.id,
            tools=available_tools,
            event_id=plan_event,
        )

        plan_start_ns = self._monotonic_ns()

        try:
            plan_response = await self._llm_json_call(
                planning_messages, PlanResponseModel, thread, request_id=request_id, event_id=plan_event
            )
            normalized_steps, parser_used = self._normalize_plan_steps(plan_response)
        except Exception as exc:
            duration_ms = self._elapsed_ms(plan_start_ns)
            logger.warning(
                "agent.plan.failed",
                error=str(exc),
                session_id=thread.id,
                duration_ms=duration_ms,
            )
            await self.observability.emit(
                EventType.PLAN_GENERATING,
                {"status": "failed", "error": str(exc), "duration_ms": duration_ms},
                thread=thread,
                event_id=plan_event,
            )
            if self.config.planning_required:
                raise RuntimeError("Planning required but failed") from exc
            return

        duration_ms = self._elapsed_ms(plan_start_ns)

        if not normalized_steps:
            if self.config.planning_required:
                raise RuntimeError("Planning required but produced no steps")
            return

        if isinstance(plan_response, PlanResponseModel):
            summary_lines = plan_response.summary_lines()
        else:
            summary_lines = []
            for step in normalized_steps:
                tool_label = step["tool"]
                if tool_label and tool_label.lower() not in {"none", ""}:
                    summary_lines.append(f"{step['position']}. {step['description']} (Tool: {tool_label})")
                else:
                    summary_lines.append(f"{step['position']}. {step['description']}")
        plan_summary = "\n".join(summary_lines)

        plan_state = state.plan
        plan_state.summary = plan_summary
        plan_state.steps = normalized_steps or []
        plan_state.index = 0
        plan_state.parser = parser_used
        if isinstance(plan_response, PlanResponseModel):
            plan_state.raw = plan_response.model_dump(exclude_none=True)
        else:
            plan_state.raw = plan_response
        plan_state.message_ts = current_ts
        plan_state.skips.clear()
        self._save_turn_state(thread, state)

        thread.add_message(
            Message(
                "assistant",
                f"[Plan]\n{plan_summary}",
                metadata={"internal": True, "type": "plan", "exclude_from_prompt": True},
            )
        )
        for step in normalized_steps:
            await self.observability.emit(
                EventType.PLAN_STEP,
                {"step": step, "plan": plan_summary},
                thread=thread,
                event_id=plan_event,
            )
        await self.observability.emit(
            EventType.PLAN_GENERATING,
            {"status": "completed", "steps": len(normalized_steps), "duration_ms": duration_ms},
            thread=thread,
            event_id=plan_event,
        )
        logger.info(
            "agent.plan.created",
            session_id=thread.id,
            steps=len(normalized_steps),
            parser=parser_used,
            duration_ms=duration_ms,
            event_id=plan_event,
        )
        self.metrics["plan_generated_total"] += 1
        self.metrics["plan_generation_duration_ms_total"] += duration_ms

    def _normalize_plan_steps(
        self, plan_data: PlanResponseModel | str
    ) -> tuple[Optional[List[Dict[str, Any]]], str]:
        steps: List[Dict[str, Any]] = []
        parser_used = "structured"

        if isinstance(plan_data, PlanResponseModel):
            plan_model = plan_data
        else:
            parser_used = "text"
            plan_text = plan_data.strip()
            numbered_pattern = re.compile(r"^\s*(\d+)[\).:-]?\s*(.*)")
            for line in plan_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                match = numbered_pattern.match(line)
                if not match:
                    continue
                description = match.group(2).strip()
                if not description:
                    continue
                steps.append(
                    {
                        "position": len(steps) + 1,
                        "description": description,
                        "tool": "",
                        "success": "",
                    }
                )
            return (steps or None), parser_used

        for idx, step in enumerate(plan_model.steps, start=1):
            description = step.description.strip()
            if not description:
                continue
            steps.append(
                {
                    "position": idx,
                    "description": description,
                    "tool": (step.tool or "").strip(),
                    "success": (step.success or "").strip(),
                }
            )
        return (steps or None), parser_used

    def _note_plan_skip(self, thread: Thread, position: int, reason: str) -> None:
        """Record why a particular plan step was skipped."""
        state = self._get_turn_state(thread)
        state.plan.skips[position] = reason
        self._save_turn_state(thread, state)

    def _record_plan_progress(
        self,
        thread: Thread,
        steps_completed: int = 1,
        reason: Optional[str] = None,
    ) -> None:
        if not self.config.planning_enabled:
            return
        state = self._get_turn_state(thread)
        plan_steps = state.plan.steps
        if not plan_steps:
            return
        current_index = state.plan.index
        new_index = min(current_index + steps_completed, len(plan_steps))
        if new_index <= current_index:
            return
        state.plan.index = new_index
        completed_step = plan_steps[new_index - 1]
        progress_payload = {
            "plan_progress": new_index,
            "total_steps": len(plan_steps),
            "completed_step": completed_step.get("description"),
        }
        if reason:
            progress_payload["reason"] = reason
        progress_event = self._new_event_id()
        event = self.observability.event(
            EventType.AGENT_THINKING,
            progress_payload,
            event_id=progress_event,
        )
        thread.add_event(event)
        logger.info(
            "agent.plan.progress",
            session_id=thread.id,
            completed=new_index,
            total=len(plan_steps),
            reason=reason,
            event_id=progress_event,
        )
        self._save_turn_state(thread, state)

    def _complete_remaining_plan_steps(self, thread: Thread) -> None:
        if not self.config.planning_enabled:
            return
        state = self._get_turn_state(thread)
        plan_steps = state.plan.steps
        if not plan_steps:
            return
        plan_index = state.plan.index
        if plan_index < len(plan_steps):
            skip_map = state.plan.skips
            skipped_steps: List[Dict[str, Any]] = []
            for step in plan_steps[plan_index:]:
                position = step.get("position", len(skipped_steps) + plan_index + 1)
                skipped_steps.append(
                    {
                        "step": position,
                        "description": step.get("description"),
                        "tool": step.get("tool"),
                        "reason": skip_map.get(position, "not_attempted"),
                    }
                )
            planned = len(plan_steps)
            coverage = round(plan_index / planned, 3) if planned else 1.0
            payload = {
                "planned_steps": planned,
                "completed_steps": plan_index,
                "plan_coverage": coverage,
                "skipped": skipped_steps,
            }
            incomplete_event_id = self._new_event_id()
            incomplete_event = self.observability.event(
                EventType.AGENT_THINKING,
                {"plan_incomplete": payload},
                event_id=incomplete_event_id,
            )
            thread.add_event(incomplete_event)
            logger.warning(
                "agent.plan.incomplete",
                session_id=thread.id,
                **payload,
                event_id=incomplete_event_id,
            )
            state.plan.index = len(plan_steps)
            self._save_turn_state(thread, state)
            return
        state.plan.index = len(plan_steps)
        completion_event_id = self._new_event_id()
        completion_record = self.observability.event(
            EventType.AGENT_THINKING,
            {"plan_complete": True, "total_steps": len(plan_steps)},
            event_id=completion_event_id,
        )
        thread.add_event(completion_record)
        logger.info(
            "agent.plan.completed",
            session_id=thread.id,
            total=len(plan_steps),
            event_id=completion_event_id,
        )
        self._save_turn_state(thread, state)

    async def _emit_offline_response(self, thread: Thread) -> None:
        fallback = (
            "Hi! I'm offline because no LLM provider is configured. "
            "Set an API key (e.g., OPENAI_API_KEY) to enable full responses."
        )
        logger.warning("agent.offline_response", session_id=thread.id)
        thread.add_message(Message("assistant", fallback))
        self._set_final_response(thread, fallback, None)
        await self.observability.emit(
            EventType.AGENT_RESPONSE,
            fallback,
            thread=thread,
        )

    def _assess_tool_quality(
        self,
        intent: str,
        result: ToolResult,
        has_signal: bool
    ) -> tuple[Optional[str], Optional[str]]:
        """Assess the quality of a tool result."""
        if result.success and not has_signal:
            return "low", "empty_result"
        elif intent == "action_plan" and result.success:
            if not isinstance(result.data, (list, dict)):
                return "low", "expected_structured_plan"
        elif intent == "knowledge_base" and result.success:
            if isinstance(result.data, list) and not result.data:
                return "low", "knowledge_base_empty"
        return None, None

    def _apply_env_defaults(self) -> None:
        """Populate API key/model from environment if not provided."""
        provider = self.config.provider.lower()
        # Get fields that were explicitly set in Pydantic v2
        fields_set = self.config.model_fields_set if hasattr(self.config, "model_fields_set") else set()

        api_key_sources = [
            f"{provider.upper()}_API_KEY",  # e.g. OPENAI_API_KEY
            "MINIAGENT_API_KEY",
        ]
        if not self.config.api_key:
            for var in api_key_sources:
                value = os.getenv(var)
                if value:
                    self.config.api_key = value
                    logger.info("agent.config.api_key_loaded", source=var)
                    break

        model_sources = [
            "MINIAGENT_MODEL",
            f"{provider.upper()}_MODEL",
        ]
        if "model" not in fields_set:
            for var in model_sources:
                value = os.getenv(var)
                if value:
                    self.config.model = value
                    logger.info("agent.config.model_loaded", source=var, model=value)
                    break



class AgentFactory:
    """Helper for constructing agents from a base configuration."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def create(
        self,
        *,
        overrides: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolRegistry] = None,
    ) -> Agent:
        config = self.config
        if overrides:
            config = self.config.model_copy(update=overrides)
        return Agent(config=config, tools=tools)
