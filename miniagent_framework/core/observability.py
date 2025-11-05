"""Simplified observability helpers for session/request scoped telemetry."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from .events import Event, EventType, StreamCallback


@dataclass
class ObservabilityContext:
    """Minimal context shared across a session and its active request."""

    session_id: str
    request_id: Optional[str] = None


_observability_context: ContextVar[Optional[ObservabilityContext]] = ContextVar(
    "miniagent_observability_context",
    default=None,
)


class Observability:
    """Helper for binding session/request identifiers and emitting events."""

    def __init__(self, callbacks: StreamCallback) -> None:
        self.callbacks = callbacks

    @staticmethod
    def new_event_id() -> str:
        """Return a short identifier for the most atomic telemetry event."""
        return uuid4().hex[:8]

    @contextmanager
    def session_scope(self, *, session_id: str) -> Iterator[ObservabilityContext]:
        """Bind a session for the duration of the scope."""
        token: Token[Optional[ObservabilityContext]] = _observability_context.set(
            ObservabilityContext(session_id=session_id)
        )
        try:
            yield _observability_context.get()
        finally:
            _observability_context.reset(token)

    @contextmanager
    def request_scope(self, request_id: str) -> Iterator[ObservabilityContext]:
        """Bind a request nested under the active session."""
        current = _observability_context.get()
        if current is None:
            raise RuntimeError("request_scope requires an active session_scope")
        token: Token[Optional[ObservabilityContext]] = _observability_context.set(
            ObservabilityContext(session_id=current.session_id, request_id=request_id)
        )
        try:
            yield _observability_context.get()
        finally:
            _observability_context.reset(token)

    def metadata(self, *, event_id: Optional[str] = None) -> Dict[str, Any]:
        """Build metadata for an event, generating an event identifier if needed."""
        context = _observability_context.get()
        metadata: Dict[str, Any] = {}
        if context:
            metadata["session_id"] = context.session_id
            if context.request_id:
                metadata["request_id"] = context.request_id
        metadata["event_id"] = event_id or self.new_event_id()
        return metadata

    def event(
        self,
        event_type: EventType,
        data: Any,
        *,
        event_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        metadata = self.metadata(event_id=event_id)
        if extra_metadata:
            metadata.update(extra_metadata)
        return Event(event_type, data, metadata)

    async def emit(
        self,
        event_type: EventType,
        data: Any,
        *,
        thread: Optional["Thread"] = None,
        event_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        from .core import Thread  # Local import to avoid circular dependency

        event = self.event(
            event_type,
            data,
            event_id=event_id,
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

        if result.get("latency_ms", 0) > 0 and total_tokens is not None:
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
    "ObservabilityContext",
]
