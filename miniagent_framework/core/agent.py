"""Lean agent runtime inspired by HICA with hybrid planning/decision flow."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Literal, Type, Union, Tuple, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, model_validator
from structlog.contextvars import bind_contextvars, clear_contextvars, get_contextvars, unbind_contextvars

from .core import Message, Thread
from .events import Event, EventType, StreamCallback
from .llm import LLMClient, RetryPolicy
from .logging import logger
from .tools import ToolRegistry, ToolResult


@contextmanager
def correlation_scope(correlation_id: Optional[str]):
    """Context manager to bind/unbind correlation id for structured logging."""
    if not correlation_id:
        yield
        return
    context = get_contextvars()
    previous = context.get("correlation_id")
    bind_contextvars(correlation_id=correlation_id)
    try:
        yield
    finally:
        if previous is None:
            try:
                unbind_contextvars("correlation_id")
            except LookupError:
                pass
        else:
            bind_contextvars(correlation_id=previous)


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
        object.__setattr__(self, "provider", self.provider.lower())
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


@dataclass
class ActionResult:
    action: Literal["tool", "final", "clarification", "fallback"]
    message: Optional[str] = None
    summary: Optional[Any] = None
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None


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
        self.offline = self.llm.client is None
        if self.offline:
            logger.warning(
                "LLM client not configured; agent will return fallback responses"
            )
        self.metrics: Counter[str] = Counter()
        if os.getenv("MINIAGENT_EVENT_LOG", "0") == "1":
            self.callbacks.on_any(self._log_event_telemetry)

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

    def _event_metadata(
        self, request_id: Optional[str] = None, correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if request_id:
            metadata["request_id"] = request_id
        if correlation_id:
            metadata["correlation_id"] = correlation_id
        return metadata

    def _prepare_new_turn(
        self, thread: Thread, message: Message, *, request_id: Optional[str] = None
    ) -> None:
        """Reset per-turn state such as planning metadata."""
        last_planned_ts = thread.metadata.get("plan_message_ts")
        current_ts = message.timestamp.isoformat()
        thread.metadata["current_user_message_ts"] = current_ts
        thread.metadata["last_request_id"] = request_id
        if not thread.metadata.get("plan") or last_planned_ts != current_ts:
            self._reset_plan_state(thread)

    def _reset_plan_state(self, thread: Thread) -> None:
        """Clear plan metadata so a fresh plan can be generated."""
        for key in ("plan", "plan_steps", "plan_index", "plan_parser", "plan_raw"):
            thread.metadata.pop(key, None)
        thread.metadata["plan_message_ts"] = None

    async def _await_with_timeout(
        self,
        coro: Awaitable[Any],
        timeout: Optional[float],
        timeout_message: str,
    ) -> Any:
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(timeout_message) from exc

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
        try:
            bind_contextvars(request_id=request_id, thread_id=thread.id)
            run_logger = logger.bind(request_id=request_id, thread_id=thread.id)
            run_logger.info(
                "agent.run.start",
                user_input=user_input,
                stream=stream,
            )
            message = Message("user", user_input, metadata={"request_id": request_id})
            thread.add_message(message)
            self._prepare_new_turn(thread, message, request_id=request_id)
            start_event = Event(
                EventType.AGENT_START,
                {"thread_id": thread.id, "user_input": user_input},
                metadata={"request_id": request_id},
            )
            thread.add_event(start_event)
            await self.callbacks.emit(start_event)

            async for _ in self.agent_loop(
                thread,
                context=context,
                stream=stream,
                max_iterations=max_iterations,
                request_id=request_id,
            ):
                pass
        except Exception as exc:
            run_logger = logger.bind(request_id=request_id, thread_id=thread.id)
            run_logger.error(
                "agent.run.exception",
                error=str(exc),
                exc_type=type(exc).__name__,
            )
            logger.exception("agent.run.exception_trace")
            error_event = Event(
                EventType.ERROR,
                {"error": str(exc)},
                metadata={"request_id": request_id},
            )
            thread.add_event(error_event)
            await self.callbacks.emit(error_event)
            if isinstance(exc, RuntimeError):
                fallback = f"Configuration issue: {exc}"
            else:
                fallback = "I'm sorry, I ran into an internal error while responding."
            response_event = Event(
                EventType.AGENT_RESPONSE,
                fallback,
                metadata={"request_id": request_id},
            )
            thread.add_event(response_event)
            await self.callbacks.emit(response_event)
            return fallback
        finally:
            clear_contextvars()

        for event in reversed(thread.events):
            if event.type == EventType.AGENT_RESPONSE and isinstance(event.data, str):
                return event.data

        fallback = "I'm sorry, I couldn't produce an answer this time."
        response_event = Event(
            EventType.AGENT_RESPONSE,
            fallback,
            metadata={"request_id": request_id},
        )
        thread.add_event(response_event)
        await self.callbacks.emit(response_event)
        logger.warning(
            "agent.final_response_missing",
            thread_id=thread.id,
            event_count=len(thread.events),
        )
        return fallback

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
        thread.metadata["last_request_id"] = request_id

        for iteration in range(max_iterations):
            action = await self._decide_next_action(
                thread, context, iteration, use_stream, request_id=request_id
            )

            if action.action == "tool" and action.tool:
                if action.message:
                    thread.add_event(
                        Event(
                            EventType.AGENT_THINKING,
                            {"tool": action.tool, "note": action.message},
                            metadata={"request_id": request_id},
                        )
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
                clarification_event = Event(
                    EventType.AGENT_RESPONSE,
                    clarification_msg,
                    metadata={"request_id": request_id, "intent": "clarification"},
                )
                thread.add_event(clarification_event)
                await self.callbacks.emit(clarification_event)
                logger.info(
                    "agent.clarification_requested",
                    message=clarification_msg,
                    thread_id=thread.id,
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
                await self._finalize_response(
                    thread,
                    context,
                    request_id=request_id,
                    fallback_message=action.message,
                )
                yield thread
                return

            logger.warning(
                "agent.unhandled_action",
                action=action.action,
                thread_id=thread.id,
            )
            await self._finalize_response(thread, context, request_id=request_id)
            yield thread
            return

        logger.warning("Max iterations reached without completion", thread_id=thread.id)
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
        correlation_id: Optional[str] = None,
    ) -> BaseModel:
        correlation = correlation_id or uuid4().hex[:8]
        with correlation_scope(correlation):
            llm_start = Event(
                EventType.LLM_CALL_START,
                {"messages": len(messages)},
                metadata=self._event_metadata(request_id, correlation),
            )
            if thread:
                thread.add_event(llm_start)
            await self.callbacks.emit(llm_start)

            logger.debug(
                "agent.llm.request",
                message_count=len(messages),
                last_user_message=next(
                    (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                    None,
                ),
                thread_id=thread.id if thread else None,
            )

            start_time = time.perf_counter()
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
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                error_event = Event(
                    EventType.LLM_ERROR,
                    {"error": str(exc), "duration_ms": duration_ms},
                    metadata=self._event_metadata(request_id, correlation),
                )
                if thread:
                    thread.add_event(error_event)
                await self.callbacks.emit(error_event)
                self.metrics["llm_error_total"] += 1
                raise

            if isinstance(raw, dict):
                payload = json.dumps(raw)
            else:
                payload = str(raw)

            payload = payload.strip()
            if not payload:
                raise ValueError("Empty response from LLM")

            logger.debug(
                "agent.llm.raw_response",
                payload_preview=payload[:500],
                payload_length=len(payload),
                thread_id=thread.id if thread else None,
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

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            llm_end = Event(
                EventType.LLM_CALL_END,
                {"model": response_model.__name__, "duration_ms": duration_ms},
                metadata=self._event_metadata(request_id, correlation),
            )
            if thread:
                thread.add_event(llm_end)
            await self.callbacks.emit(llm_end)

            logger.debug(
                "agent.llm.validated_response",
                model=response_model.__name__,
                duration_ms=duration_ms,
                thread_id=thread.id if thread else None,
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
            logger.info(
                "agent.intent.selected",
                action="tool",
                tool=pending_call["name"],
                thread_id=thread.id,
                source="pending_queue",
            )
            tool_selection_event = Event(
                EventType.TOOL_SELECTION,
                {
                    "tool": pending_call["name"],
                    "arguments": pending_call["arguments"],
                    "source": "pending_queue",
                },
                metadata=self._event_metadata(request_id, pending_call.get("id")),
            )
            thread.add_event(tool_selection_event)
            await self.callbacks.emit(tool_selection_event)
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
        decision_correlation = uuid4().hex[:8]
        tool_calls_payload: Optional[List[Dict[str, Any]]] = None
        final_text: Optional[str] = None
        with correlation_scope(decision_correlation):
            llm_start = Event(
                EventType.LLM_CALL_START,
                {"messages": len(messages), "phase": "decision"},
                metadata=self._event_metadata(request_id, decision_correlation),
            )
            thread.add_event(llm_start)
            await self.callbacks.emit(llm_start)
            logger.debug(
                "agent.decision.request",
                thread_id=thread.id,
                iteration=iteration,
            )

            tool_schemas = (
                self.tool_registry.get_schemas() if self.tool_registry.tools else None
            )

            decision_start = time.perf_counter()
            duration_ms = None
            try:
                if stream:
                    tool_calls_payload, final_text = await self._stream_decision_response(
                        messages,
                        tool_schemas,
                        thread,
                        request_id=request_id,
                        correlation_id=decision_correlation,
                    )
                else:
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
                    logger.debug(
                        "agent.decision.raw_response",
                        thread_id=thread.id,
                        response=str(raw_response)[:500],
                    )
                    tool_calls_payload = self._extract_tool_calls(raw_response)
                    final_text = self._extract_text_response(raw_response)
            except Exception as exc:
                duration_ms = int((time.perf_counter() - decision_start) * 1000)
                error_event = Event(
                    EventType.LLM_ERROR,
                    {"error": str(exc), "phase": "decision", "duration_ms": duration_ms},
                    metadata=self._event_metadata(request_id, decision_correlation),
                )
                thread.add_event(error_event)
                await self.callbacks.emit(error_event)
                self.metrics["llm_error_total"] += 1
                raise
            finally:
                if duration_ms is None:
                    duration_ms = int((time.perf_counter() - decision_start) * 1000)
                llm_end = Event(
                    EventType.LLM_CALL_END,
                    {"phase": "decision", "duration_ms": duration_ms},
                    metadata=self._event_metadata(request_id, decision_correlation),
                )
                thread.add_event(llm_end)
                await self.callbacks.emit(llm_end)
                self.metrics["llm_call_total"] += 1
                self.metrics["llm_call_duration_ms_total"] += duration_ms

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
                        thread_id=thread.id,
                    )
                    selection_event = Event(
                        EventType.TOOL_SELECTION,
                        {
                            "tool": next_call["name"],
                            "arguments": next_call["arguments"],
                            "tool_call_id": next_call["id"],
                        },
                        metadata=self._event_metadata(request_id, next_call.get("id")),
                    )
                    thread.add_event(selection_event)
                    await self.callbacks.emit(selection_event)
                    return ActionResult(
                        action="tool",
                        tool=next_call["name"],
                        arguments=next_call["arguments"],
                        tool_call_id=next_call["id"],
                    )
            logger.warning(
                "agent.tool_call.parsing_failed",
                thread_id=thread.id,
                payload=tool_calls_payload,
            )

        action_result = self._parse_final_action(final_text)
        if action_result.action == "fallback":
            logger.warning(
                "agent.decision.unstructured_response",
                thread_id=thread.id,
                iteration=iteration,
            )
        else:
            response_event = Event(
                EventType.LLM_RESPONSE,
                {
                    "action": action_result.action,
                    "message": action_result.message,
                    "summary": action_result.summary,
                },
                metadata=self._event_metadata(request_id, decision_correlation),
            )
            thread.add_event(response_event)
            await self.callbacks.emit(response_event)
        logger.info(
            "agent.intent.selected",
            action=action_result.action,
            tool=None,
            thread_id=thread.id,
        )
        return action_result

    async def _stream_decision_response(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        thread: Thread,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
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
                    await self.callbacks.emit(
                        Event(
                            EventType.STREAM_START,
                            None,
                            metadata=self._event_metadata(request_id, correlation_id),
                        )
                    )
                chunks.append(chunk)
                await self.callbacks.emit(
                    Event(
                        EventType.STREAM_CHUNK,
                        chunk,
                        metadata=self._event_metadata(request_id, correlation_id),
                    )
                )
            elif isinstance(chunk, dict):
                extracted = self._extract_tool_calls(chunk)
                if extracted:
                    tool_calls.extend(extracted)
            else:
                logger.debug(
                    "agent.decision.stream.unknown_chunk",
                    chunk_type=type(chunk).__name__,
                    thread_id=thread.id,
                )

        if stream_started:
            await self.callbacks.emit(
                Event(
                    EventType.STREAM_END,
                    None,
                    metadata=self._event_metadata(request_id, correlation_id),
                )
            )

        final_text = "".join(chunks)
        if final_text:
            logger.debug(
                "agent.decision.stream.final_text_preview",
                preview=final_text[:200],
                thread_id=thread.id,
            )

        return tool_calls, final_text

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
            try:
                raw_payload = json.loads(serialized_text)
            except json.JSONDecodeError:
                json_match = re.search(
                    r"```(?:json)?\s*(\{.*\})\s*```", serialized_text, re.DOTALL
                )
                if json_match:
                    try:
                        raw_payload = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        raw_payload = None

        if isinstance(raw_payload, dict):
            clarification_requested = bool(raw_payload.get("clarification"))

        if raw_payload is None:
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
            tool_calls = raw_response.get("tool_calls")
            if isinstance(tool_calls, list):
                return tool_calls
            if (
                raw_response.get("type") == "tool_calls"
                and isinstance(tool_calls, list)
            ):
                return tool_calls
        return []

    def _prepare_tool_calls(
        self, raw_calls: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        formatted_for_messages: List[Dict[str, Any]] = []
        pending_calls: List[Dict[str, Any]] = []

        for raw_call in raw_calls:
            name = raw_call.get("name")
            if not name:
                logger.warning("agent.tool_call.missing_name", payload=raw_call)
                continue
            call_id = raw_call.get("id") or f"tool_call_{uuid4().hex[:8]}"
            raw_arguments = raw_call.get("arguments") or "{}"

            arguments_dict = self._parse_tool_arguments(raw_arguments, name, call_id)
            arguments_str = (
                raw_arguments
                if isinstance(raw_arguments, str)
                else json.dumps(raw_arguments, ensure_ascii=False)
            )

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
                    "correlation_id": call_id,
                    "note": raw_call.get("note"),
                }
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
        queue = thread.metadata.setdefault("pending_tool_calls", [])
        queue.extend(tool_calls)

    def _pop_next_tool_call(self, thread: Thread) -> Optional[Dict[str, Any]]:
        queue: List[Dict[str, Any]] = thread.metadata.get("pending_tool_calls", [])
        if not queue:
            return None
        next_call = queue.pop(0)
        if not queue:
            thread.metadata.pop("pending_tool_calls", None)
        else:
            thread.metadata["pending_tool_calls"] = queue
        return next_call

    def _extract_text_response(self, raw_response: Any) -> Optional[str]:
        if raw_response is None:
            return None
        if isinstance(raw_response, str):
            return raw_response
        if isinstance(raw_response, dict):
            for key in ("message", "content", "text", "response"):
                value = raw_response.get(key)
                if isinstance(value, str):
                    return value
        try:
            return json.dumps(raw_response, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(raw_response)

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
        plan_index = thread.metadata.get("plan_index", 0)
        messages = self._build_messages(
            instruction, thread, context, plan_index, mode="final"
        )
        try:
            response = await self._llm_json_call(
                messages,
                FinalResponseModel,
                thread,
                request_id=request_id,
            )
        except Exception as exc:
            logger.error("Final response generation failed", error=str(exc))
            fallback = (
                fallback_message
                or "I'm sorry, I couldn't generate a response right now."
            )
            if isinstance(fallback, dict):
                fallback = json.dumps(fallback, ensure_ascii=False)
            fallback_event = Event(
                EventType.AGENT_RESPONSE,
                fallback,
                metadata=self._event_metadata(request_id),
            )
            thread.add_message(Message("assistant", fallback))
            thread.add_event(fallback_event)
            await self.callbacks.emit(fallback_event)
            return

        await self._commit_final_response(
            response.message, response.summary, thread, request_id=request_id
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
        correlation_id = call_id or f"tool_call_{uuid4().hex[:8]}"
        with correlation_scope(correlation_id):
            start_payload = {"tool": intent, "arguments": arguments}
            tool_start_event = Event(
                EventType.TOOL_EXECUTION_START,
                start_payload,
                metadata=self._event_metadata(request_id, correlation_id),
            )
            thread.add_event(tool_start_event)
            await self.callbacks.emit(tool_start_event)

            tool_execution_event = Event(
                EventType.TOOL_EXECUTION,
                {"tool": intent, "parameters": arguments},
                metadata=self._event_metadata(request_id, correlation_id),
            )
            thread.add_event(tool_execution_event)
            await self.callbacks.emit(tool_execution_event)

            logger.info(
                "agent.tool.execute",
                tool=intent,
                arguments=json.dumps(arguments) if arguments else "{}",
            )

            start_time = time.perf_counter()
            duration_ms = None
            try:
                result: ToolResult = await self._await_with_timeout(
                    self.tool_registry.execute_tool(intent, arguments),
                    self.config.tool_timeout,
                    f"Tool '{intent}' execution timed out",
                )
            except TimeoutError as exc:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                error_event = Event(
                    EventType.TOOL_ERROR,
                    {"tool": intent, "error": str(exc), "duration_ms": duration_ms},
                    metadata=self._event_metadata(request_id, correlation_id),
                )
                thread.add_event(error_event)
                await self.callbacks.emit(error_event)
                result = ToolResult(
                    success=False,
                    error=str(exc),
                )
                self.metrics[f"tool_error_total"] += 1
                self.metrics[f"tool_{intent}_error_total"] += 1
            except Exception as exc:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                error_event = Event(
                    EventType.TOOL_ERROR,
                    {"tool": intent, "error": str(exc), "duration_ms": duration_ms},
                    metadata=self._event_metadata(request_id, correlation_id),
                )
                thread.add_event(error_event)
                await self.callbacks.emit(error_event)
                result = ToolResult(success=False, error=str(exc))
                self.metrics[f"tool_error_total"] += 1
                self.metrics[f"tool_{intent}_error_total"] += 1
            else:
                duration_ms = int((time.perf_counter() - start_time) * 1000)

            payload = result.to_dict()
            payload.setdefault("metadata", {})["duration_ms"] = duration_ms
            tool_result_event = Event(
                EventType.TOOL_RESULT,
                payload,
                metadata=self._event_metadata(request_id, correlation_id),
            )
            thread.add_event(tool_result_event)
            await self.callbacks.emit(tool_result_event)
            logger.info(
                "agent.tool.result",
                tool=intent,
                success=result.success,
                duration_ms=duration_ms,
            )

            end_event = Event(
                EventType.TOOL_EXECUTION_END,
                {"tool": intent, "success": result.success, "duration_ms": duration_ms},
                metadata=self._event_metadata(request_id, correlation_id),
            )
            thread.add_event(end_event)
            await self.callbacks.emit(end_event)
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
        thread.add_event(
            Event(
                EventType.AGENT_THINKING,
                {"tool": intent, "result": summary},
                metadata=self._event_metadata(request_id, correlation_id),
            )
        )
        logger.debug(
            "agent.tool.summary",
            tool=intent,
            summary_preview=summary[:200] if isinstance(summary, str) else summary,
            thread_id=thread.id,
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
    ) -> None:
        payload: Dict[str, Any] = {}
        if isinstance(message, dict):
            payload["message"] = message
        else:
            payload["message"] = message
        if summary is not None:
            payload["summary"] = summary
        final_message = json.dumps(payload, ensure_ascii=False)
        thread.metadata.pop("pending_tool_calls", None)
        thread.add_message(
            Message("assistant", final_message, metadata={"request_id": request_id} if request_id else {})
        )
        response_event = Event(
            EventType.AGENT_RESPONSE,
            final_message,
            metadata=self._event_metadata(request_id),
        )
        thread.add_event(response_event)
        await self.callbacks.emit(response_event)
        logger.info(
            "agent.final_response",
            message=final_message,
            summary=summary,
            thread_id=thread.id,
        )
        if summary is not None:
            info_event = Event(
                EventType.INFO,
                {"summary": summary},
                metadata=self._event_metadata(request_id),
            )
            thread.add_event(info_event)
        self._complete_remaining_plan_steps(thread)

    def _build_messages(
        self,
        prompt: str,
        thread: Optional[Thread],
        context: Optional[str],
        iteration: int = 0,
        mode: str = "decision",
    ) -> List[Dict[str, Any]]:
        metadata = self._tool_metadata()
        system_content = self.config.system_prompt
        if metadata:
            system_content += f"\n\nAvailable tools:\n{metadata}"
        if context:
            system_content += f"\n\nContext:\n{context}"

        if thread:
            plan_summary: Optional[str] = thread.metadata.get("plan")
            plan_steps: List[Dict[str, Any]] = thread.metadata.get("plan_steps", [])
            plan_index = thread.metadata.get("plan_index", 0)
            if plan_summary:
                if iteration == 0:
                    system_content += (
                        "\n\nPlanned workflow (execute sequentially):\n"
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
                        system_content += f"\n\n{step_instructions}"
                    else:
                        system_content += (
                            "\n\nAll planned steps have been attempted. Consolidate findings and prepare the final response."
                        )

        if mode == "decision":
            system_content += (
                "\n\nDecision protocol:\n"
                "- Decide whether to continue the reasoning with a tool call or respond directly.\n"
                "- Call a tool when its output is required to answer confidently.\n"
                "- When no tool is needed, reply by emitting STRICT JSON with keys 'message' and optional 'summary'.\n"
                "- Include \"clarification\": true when you must ask the user for more information; place the question in 'message'.\n"
                "- Always keep the user-facing response concise and helpful."
            )
        elif mode == "final":
            system_content += (
                "\n\nCompose the final user-facing answer summarizing tool evidence. Respond as JSON with keys 'message', optional 'summary', and include 'clarification': true if you still need more information."
            )
        elif mode == "clarification":
            system_content += (
                "\n\nAsk a concise question to obtain the missing information before proceeding."
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

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

        messages.append({"role": "system", "content": prompt})
        return messages

    def _tool_metadata(self) -> str:
        lines = []
        for tool in self.tool_registry.tools.values():
            schema = tool.to_function_schema()
            description = schema.get("description", "")
            lines.append(f"<tool> {schema.get('name')} : {description}")
        return "\n".join(lines)

    async def _generate_clarification_response(
        self,
        reason: Optional[str],
        thread: Thread,
        context: Optional[str],
        iteration: int,
    ) -> str:
        prompt = (
            "You need to ask the user for clarification before proceeding. "
            "Craft a concise, polite question that explains what information you need next."
        )
        if reason:
            prompt += f" Reason for clarification: {reason}."
        messages = self._build_messages(
            prompt, thread, context, iteration, mode="clarification"
        )
        try:
            response = await self._llm_json_call(messages, FinalResponseModel, thread)
            message = response.message
            if isinstance(message, dict):
                message = json.dumps(message, ensure_ascii=False)
            return message
        except Exception:
            raw = await self.llm.complete(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=256,
                stream=False,
                tools=None,
            )
            return raw if isinstance(raw, str) else json.dumps(raw)

    def _should_generate_plan(
        self,
        thread: Thread,
        user_message: str,
        context: Optional[str],
    ) -> bool:
        if not user_message:
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

        current_ts = thread.metadata.get("current_user_message_ts")
        existing_plan_ts = thread.metadata.get("plan_message_ts")
        if thread.metadata.get("plan") and existing_plan_ts == current_ts:
            return

        self._reset_plan_state(thread)

        last_user_message = thread.get_last_user_message() or ""
        if not self._should_generate_plan(thread, last_user_message, context):
            logger.debug(
                "agent.plan.skipped",
                thread_id=thread.id,
                reason="heuristic",
            )
            return

        plan_correlation_id = uuid4().hex[:8]
        with correlation_scope(plan_correlation_id):
            start_event = Event(
                EventType.PLAN_GENERATING,
                {"status": "starting", "thread_id": thread.id},
                metadata=self._event_metadata(request_id, plan_correlation_id),
            )
            thread.add_event(start_event)
            await self.callbacks.emit(start_event)

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
                thread_id=thread.id,
                tools=available_tools,
            )

            plan_start = time.perf_counter()

            try:
                plan_response = await self._llm_json_call(
                    planning_messages, PlanResponseModel, thread, request_id=request_id, correlation_id=plan_correlation_id
                )
                normalized_steps, parser_used = self._normalize_plan_steps(plan_response)
            except Exception as exc:
                duration_ms = int((time.perf_counter() - plan_start) * 1000)
                logger.warning(
                    "agent.plan.failed",
                    error=str(exc),
                    thread_id=thread.id,
                    duration_ms=duration_ms,
                )
                failure_event = Event(
                    EventType.PLAN_GENERATING,
                    {"status": "failed", "error": str(exc), "duration_ms": duration_ms},
                    metadata=self._event_metadata(request_id, plan_correlation_id),
                )
                thread.add_event(failure_event)
                await self.callbacks.emit(failure_event)
                if self.config.planning_required:
                    raise RuntimeError("Planning required but failed") from exc
                return

            duration_ms = int((time.perf_counter() - plan_start) * 1000)

            if not normalized_steps:
                if self.config.planning_required:
                    raise RuntimeError("Planning required but produced no steps")
                return

            summary_lines = []
            for step in normalized_steps:
                tool_label = step["tool"]
                if tool_label and tool_label.lower() not in {"none", ""}:
                    summary_lines.append(f"{step['position']}. {step['description']} (Tool: {tool_label})")
                else:
                    summary_lines.append(f"{step['position']}. {step['description']}")
            plan_summary = "\n".join(summary_lines)

            thread.metadata["plan"] = plan_summary
            thread.metadata["plan_steps"] = normalized_steps
            thread.metadata["plan_index"] = 0
            thread.metadata["plan_parser"] = parser_used
            if isinstance(plan_response, PlanResponseModel):
                thread.metadata["plan_raw"] = plan_response.model_dump(exclude_none=True)
            else:
                thread.metadata["plan_raw"] = plan_response
            thread.metadata["plan_message_ts"] = current_ts

            thread.add_message(
                Message(
                    "assistant",
                    f"[Plan]\n{plan_summary}",
                    metadata={"internal": True, "type": "plan", "exclude_from_prompt": True},
                )
            )
            for step in normalized_steps:
                plan_step_event = Event(
                    EventType.PLAN_STEP,
                    {"step": step, "plan": plan_summary},
                    metadata=self._event_metadata(request_id, plan_correlation_id),
                )
                thread.add_event(plan_step_event)
                await self.callbacks.emit(plan_step_event)
            plan_ready_event = Event(
                EventType.PLAN_GENERATING,
                {"status": "completed", "steps": len(normalized_steps), "duration_ms": duration_ms},
                metadata=self._event_metadata(request_id, plan_correlation_id),
            )
            thread.add_event(plan_ready_event)
            await self.callbacks.emit(plan_ready_event)
            logger.info(
                "agent.plan.created",
                thread_id=thread.id,
                steps=len(normalized_steps),
                parser=parser_used,
                duration_ms=duration_ms,
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

    def _record_plan_progress(
        self,
        thread: Thread,
        steps_completed: int = 1,
        reason: Optional[str] = None,
    ) -> None:
        if not self.config.planning_enabled:
            return
        plan_steps: List[Dict[str, Any]] = thread.metadata.get("plan_steps", [])
        if not plan_steps:
            return
        current_index = thread.metadata.get("plan_index", 0)
        new_index = min(current_index + steps_completed, len(plan_steps))
        if new_index <= current_index:
            return
        thread.metadata["plan_index"] = new_index
        completed_step = plan_steps[new_index - 1]
        progress_payload = {
            "plan_progress": new_index,
            "total_steps": len(plan_steps),
            "completed_step": completed_step.get("description"),
        }
        if reason:
            progress_payload["reason"] = reason
        request_id = thread.metadata.get("last_request_id")
        thread.add_event(
            Event(
                EventType.AGENT_THINKING,
                progress_payload,
                metadata=self._event_metadata(request_id),
            )
        )
        logger.info(
            "agent.plan.progress",
            thread_id=thread.id,
            completed=new_index,
            total=len(plan_steps),
            reason=reason,
        )

    def _complete_remaining_plan_steps(self, thread: Thread) -> None:
        if not self.config.planning_enabled:
            return
        plan_steps: List[Dict[str, Any]] = thread.metadata.get("plan_steps", [])
        if not plan_steps:
            return
        thread.metadata["plan_index"] = len(plan_steps)
        thread.add_event(
            Event(
                EventType.AGENT_THINKING,
                {"plan_complete": True, "total_steps": len(plan_steps)},
                metadata=self._event_metadata(thread.metadata.get("last_request_id")),
            )
        )
        logger.info("agent.plan.completed", thread_id=thread.id, total=len(plan_steps))

    async def _emit_offline_response(self, thread: Thread) -> None:
        fallback = (
            "Hi! I'm offline because no LLM provider is configured. "
            "Set an API key (e.g., OPENAI_API_KEY) to enable full responses."
        )
        logger.warning("agent.offline_response", thread_id=thread.id)
        thread.add_message(Message("assistant", fallback))
        thread.add_event(Event(EventType.AGENT_RESPONSE, fallback))
        await self.callbacks.emit(Event(EventType.AGENT_RESPONSE, fallback))

    def _apply_env_defaults(self) -> None:
        """Populate API key/model from environment if not provided."""
        provider = self.config.provider.lower()
        fields_set = getattr(
            self.config,
            "model_fields_set",
            getattr(self.config, "__fields_set__", set()),
        )

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
