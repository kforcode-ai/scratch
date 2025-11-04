"""Observability helpers for session/request/operation scoped telemetry."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
from uuid import uuid4

from structlog.contextvars import bind_contextvars, get_contextvars, unbind_contextvars

from .events import Event, EventType, StreamCallback


@dataclass
class ObservabilityState:
    """Session/request identifiers kept in contextvars for observability."""

    session_id: str
    request_id: str
    trace_id: str
    thread_id: Optional[str] = None


@dataclass
class OperationContext:
    """Operation-level metadata for hierarchical observability."""

    operation_id: str
    operation_type: str
    parent_operation_id: Optional[str] = None


_observability_state: ContextVar[Optional[ObservabilityState]] = ContextVar(
    "miniagent_observability_state", default=None
)
_operation_stack: ContextVar[Tuple[OperationContext, ...]] = ContextVar(
    "miniagent_operation_stack", default=()
)


class Observability:
    """Helper for consistent event metadata, operation scoping, and emissions."""

    def __init__(self, callbacks: StreamCallback) -> None:
        self.callbacks = callbacks

    @staticmethod
    def _restore_context(previous: Dict[str, Any], keys: Tuple[str, ...]) -> None:
        """Restore structlog contextvars that were overridden in the scope."""
        for key in keys:
            if key in previous:
                bind_contextvars(**{key: previous[key]})
            else:
                try:
                    unbind_contextvars(key)
                except LookupError:
                    pass

    @contextmanager
    def session_scope(
        self,
        *,
        session_id: str,
        request_id: str,
        trace_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Iterator[ObservabilityState]:
        """Bind a unified session/request context for observability."""
        state = ObservabilityState(
            session_id=session_id,
            request_id=request_id,
            trace_id=trace_id or request_id,
            thread_id=thread_id or session_id,
        )
        state_token: Token[Optional[ObservabilityState]] = _observability_state.set(state)
        stack_token: Token[Tuple[OperationContext, ...]] = _operation_stack.set(tuple())
        previous = get_contextvars()

        bindings: Dict[str, Any] = {
            "session_id": state.session_id,
            "conversation_id": state.session_id,
            "thread_id": state.thread_id,
            "request_id": state.request_id,
            "trace_id": state.trace_id,
        }
        for key, value in list(bindings.items()):
            if value is None:
                bindings.pop(key)
        binding_keys = tuple(bindings.keys())
        if bindings:
            bind_contextvars(**bindings)

        try:
            yield state
        finally:
            _operation_stack.reset(stack_token)
            _observability_state.reset(state_token)
            self._restore_context(previous, binding_keys)

    @contextmanager
    def operation_scope(
        self,
        operation_type: str,
        *,
        operation_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[OperationContext]:
        """Bind operation-level identifiers, returning the created context."""
        state = _observability_state.get()
        if state is None:
            raise RuntimeError("operation_scope requires an active session_scope")

        stack = _operation_stack.get(tuple())
        parent = stack[-1].operation_id if stack else None
        op_context = OperationContext(
            operation_id=operation_id or uuid4().hex[:8],
            operation_type=operation_type,
            parent_operation_id=parent,
        )
        stack_token: Token[Tuple[OperationContext, ...]] = _operation_stack.set(stack + (op_context,))
        previous = get_contextvars()

        bindings: Dict[str, Any] = {
            "session_id": state.session_id,
            "conversation_id": state.session_id,
            "thread_id": state.thread_id,
            "request_id": state.request_id,
            "trace_id": state.trace_id,
            "operation_id": op_context.operation_id,
            "operation_type": op_context.operation_type,
            "parent_operation_id": op_context.parent_operation_id,
        }
        if attributes:
            bindings.update({k: v for k, v in attributes.items() if v is not None})
        cleaned_bindings = {k: v for k, v in bindings.items() if v is not None}
        binding_keys = tuple(cleaned_bindings.keys())
        bind_contextvars(**cleaned_bindings)

        try:
            yield op_context
        finally:
            _operation_stack.reset(stack_token)
            self._restore_context(previous, binding_keys)

    def metadata(
        self,
        request_id: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = get_contextvars()
        metadata: Dict[str, Any] = {}

        session_id = context.get("session_id") or context.get("thread_id")
        if session_id:
            metadata["session_id"] = session_id
        conversation_id = context.get("conversation_id")
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        thread_id = context.get("thread_id")
        if thread_id:
            metadata["thread_id"] = thread_id
        trace_id = context.get("trace_id")
        if trace_id:
            metadata["trace_id"] = trace_id
        active_request = context.get("request_id")
        if active_request:
            metadata["request_id"] = active_request
        active_operation = context.get("operation_id")
        if active_operation:
            metadata["operation_id"] = active_operation
        parent_operation_id = context.get("parent_operation_id")
        if parent_operation_id:
            metadata["parent_operation_id"] = parent_operation_id
        operation_type = context.get("operation_type")
        if operation_type:
            metadata["operation_type"] = operation_type
        if request_id:
            metadata["request_id"] = request_id
        if operation_id:
            metadata["operation_id"] = operation_id
        return metadata

    def event(
        self,
        event_type: EventType,
        data: Any,
        *,
        request_id: Optional[str] = None,
        operation_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        metadata = self.metadata(request_id=request_id, operation_id=operation_id)
        if extra_metadata:
            metadata.update(extra_metadata)
        return Event(event_type, data, metadata)

    async def emit(
        self,
        event_type: EventType,
        data: Any,
        *,
        thread: Optional["Thread"] = None,  # Forward reference for type checking
        request_id: Optional[str] = None,
        operation_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        from .core import Thread  # Local import to avoid circular dependency

        event = self.event(
            event_type,
            data,
            request_id=request_id,
            operation_id=operation_id,
            extra_metadata=extra_metadata,
        )
        if thread:
            thread.add_event(event)
        await self.callbacks.emit(event)
        return event

    @staticmethod
    def augment_llm_metrics(
        base: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        *,
        default_model: Optional[str] = None,
        ttft_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Merge provider metrics with base operation data."""
        result = dict(base)
        details = dict(metrics or {})

        model_name = details.pop("model", None)
        if model_name:
            result["model"] = model_name
        elif default_model:
            result.setdefault("model", default_model)

        prompt_tokens = details.pop("prompt_tokens", None)
        completion_tokens = details.pop("completion_tokens", None)
        total_tokens = details.pop("total_tokens", None)
        cached_tokens = details.pop("cached_tokens", None)

        if prompt_tokens is not None:
            result["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            result["completion_tokens"] = int(completion_tokens)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        if total_tokens is not None:
            result["total_tokens"] = int(total_tokens)
        if cached_tokens is not None:
            cached_tokens = int(cached_tokens)
            result["cached_tokens"] = cached_tokens
            result["cache_hit"] = cached_tokens > 0

        if (
            result.get("latency_ms", 0) > 0
            and total_tokens is not None
        ):
            latency_seconds = result["latency_ms"] / 1000.0
            if latency_seconds > 0:
                result["tokens_per_second"] = round(total_tokens / latency_seconds, 3)

        if ttft_ms is not None:
            result["ttft_ms"] = ttft_ms

        retries = details.pop("retries", None)
        if retries is not None:
            result["retries"] = retries

        provider_name = details.pop("provider", None)
        if provider_name:
            result["provider"] = provider_name

        finish_reason = details.pop("finish_reason", None)
        if finish_reason:
            result["finish_reason"] = finish_reason

        stream_flag = details.pop("stream", None)
        if stream_flag is not None:
            result["stream"] = stream_flag

        cost_usd = details.pop("cost_usd", None)
        if cost_usd is not None:
            result["cost_usd"] = cost_usd

        if details:
            result.update(details)

        return result


__all__ = [
    "Observability",
    "ObservabilityState",
    "OperationContext",
]
