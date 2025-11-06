"""
Agent – a lean, single-loop alternative to the full MiniAgent runtime.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from .core import Message, Thread
from .events import Event, EventType, StreamCallback
from .logging import logger
from .llm import LLMClient
from .observability import Observability
from .tools import ToolRegistry, ToolResult


@dataclass
class AgentConfig:
    """Minimal configuration needed to run Agent."""

    provider: str = "openai"
    model: Optional[str] = "gpt-4.1"
    api_key: Optional[str] = None
    system_prompt: str = (
        "You are a helpful assistant. Use registered tools when they are useful. "
        "Answer the user directly when you have enough information."
    )
    temperature: float = 0.0
    max_tokens: int = 1024
    tool_timeout: Optional[float] = 30.0
    llm_timeout: Optional[float] = 60.0
    stream_by_default: bool = False
    max_steps: int = 5

    def resolve_api_key(self) -> Optional[str]:
        """Resolve an API key from the config or environment variables."""
        if self.api_key:
            return self.api_key
        provider_key = os.getenv(f"{self.provider.upper()}_API_KEY")
        fallback_key = os.getenv("OPENAI_API_KEY")
        return provider_key or fallback_key


class ToolCall(BaseModel):
    """Lightweight representation of a requested tool call."""

    name: str
    id: Optional[str] = None
    arguments: Any = Field(default_factory=dict)


class Agent:
    """
    Lightweight agent that relies on a single loop:
    ask the LLM → execute tool calls (if any) → repeat until final answer.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[ToolRegistry] = None,
        llm_client: Optional[LLMClient] = None,
        callbacks: Optional[StreamCallback] = None,
    ) -> None:
        self.config = config or AgentConfig()
        resolved_api_key = self.config.resolve_api_key()
        # Keep the resolved key on the config for downstream consumers
        self.config.api_key = resolved_api_key

        self.tools = tools or ToolRegistry()
        self.llm = llm_client or LLMClient(
            provider=self.config.provider,
            api_key=resolved_api_key,
            model=self.config.model,
        )
        self.callbacks = callbacks or StreamCallback()
        self.observability = Observability(self.callbacks)
        self.callbacks.on_any(self._log_event_debug)
        self._stream_chunk_listeners: List[Callable[[str], Any]] = []
        self._stream_end_listeners: List[Callable[[], Any]] = []

    @property
    def offline(self) -> bool:
        """Return True when the LLM client is not usable."""
        return getattr(self.llm, "client", None) is None

    def on_stream_chunk(self, handler: Callable[[str], Any]) -> None:
        """Register a handler to receive streaming text chunks."""
        self._stream_chunk_listeners.append(handler)

    def on_stream_end(self, handler: Callable[[], Any]) -> None:
        """Register a handler invoked when the current stream finishes."""
        self._stream_end_listeners.append(handler)

    async def run(
        self,
        user_input: str,
        thread: Optional[Thread] = None,
        context: Optional[str] = None,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Execute a conversational turn. Returns the assistant's final reply.
        """
        thread = thread or Thread()
        thread.add_message(Message(role="user", content=user_input))

        await self.observability.emit(
            EventType.AGENT_START,
            {"max_steps": self.config.max_steps, "user_message": user_input},
            thread=thread,
        )

        if self.offline:
            fallback = "LLM client is not configured. Please provide an API key."
            thread.add_message(Message(role="assistant", content=fallback))
            await self.observability.emit(
                EventType.AGENT_ERROR,
                {"reason": "offline", "message": fallback},
                thread=thread,
            )
            await self.observability.emit(
                EventType.AGENT_COMPLETE,
                {"reason": "offline"},
                thread=thread,
            )
            return fallback

        steps = 0
        while steps < self.config.max_steps:
            steps += 1
            await self.observability.emit(
                EventType.AGENT_THINKING,
                {"step": steps, "message_count": len(thread.messages)},
                thread=thread,
            )
            try:
                reply, tool_calls = await self._llm_decide(
                    thread,
                    context=context,
                    stream=bool(stream if stream is not None else self.config.stream_by_default),
                )
            except asyncio.TimeoutError:
                timeout_msg = "The language model did not respond in time."
                logger.warning("slim_agent.llm_timeout")
                thread.add_message(Message(role="assistant", content=timeout_msg))
                return timeout_msg
            except Exception as err:
                logger.exception("slim_agent.llm_error", error=str(err))
                error_msg = f"An error occurred while contacting the model: {err}"
                thread.add_message(Message(role="assistant", content=error_msg))
                await self.observability.emit(
                    EventType.AGENT_ERROR,
                    {"reason": "llm_error", "message": error_msg},
                    thread=thread,
                )
                return error_msg

            if tool_calls:
                for call in tool_calls:
                    await self._execute_tool(call, thread)
                # Tool results become part of the thread; loop again for the next decision.
                continue

            final_text = reply or "Sorry, I could not produce an answer."
            thread.add_message(Message(role="assistant", content=final_text))
            await self.observability.emit(
                EventType.AGENT_RESPONSE,
                {"text": final_text},
                thread=thread,
            )
            await self.observability.emit(
                EventType.AGENT_COMPLETE,
                {"reason": "responded"},
                thread=thread,
            )
            return final_text

        limit_msg = "I could not finish within the allowed number of steps."
        thread.add_message(Message(role="assistant", content=limit_msg))
        logger.info("slim_agent.step_limit_reached", steps=steps)
        await self.observability.emit(
            EventType.AGENT_ERROR,
            {"reason": "step_limit", "message": limit_msg},
            thread=thread,
        )
        await self.observability.emit(
            EventType.AGENT_COMPLETE,
            {"reason": "step_limit"},
            thread=thread,
        )
        return limit_msg

    async def _llm_decide(
        self,
        thread: Thread,
        *,
        context: Optional[str],
        stream: bool,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """
        Ask the LLM for the next action. Returns (message, tool_calls).
        """
        system_parts = [self.config.system_prompt]
        if context:
            system_parts.append(f"Context:\n{context}")
        system_message = {"role": "system", "content": "\n\n".join(system_parts)}

        messages = [system_message]
        messages.extend(message.to_dict() for message in thread.messages)

        tool_schemas = self.tools.get_schemas() if self.tools.tools else None

        llm_event_id = Observability.new_event_id()
        await self.observability.emit(
            EventType.LLM_CALL_START,
            {
                "stream": stream,
                "messages": len(messages),
                "tools_available": len(tool_schemas or []),
            },
            thread=thread,
            event_id=llm_event_id,
        )

        start_ns = time.perf_counter_ns()

        try:
            if stream:
                stream_iter = await asyncio.wait_for(
                    self.llm.complete(
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=tool_schemas,
                        stream=True,
                    ),
                    timeout=self.config.llm_timeout if self.config.llm_timeout else None,
                )
                reply, tool_calls = await asyncio.wait_for(
                    self._consume_stream(stream_iter, thread, llm_event_id),
                    timeout=self.config.llm_timeout if self.config.llm_timeout else None,
                )
            else:
                raw_response = await asyncio.wait_for(
                    self.llm.complete(
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=tool_schemas,
                        stream=False,
                    ),
                    timeout=self.config.llm_timeout if self.config.llm_timeout else None,
                )
                reply, tool_calls = self._parse_completion(raw_response)
        except Exception as exc:
            duration_ms = self._elapsed_ms(start_ns)
            await self.observability.emit(
                EventType.LLM_ERROR,
                {"error": str(exc), "latency_ms": duration_ms},
                thread=thread,
                event_id=llm_event_id,
            )
            raise

        duration_ms = self._elapsed_ms(start_ns)
        await self.observability.emit(
            EventType.LLM_CALL_END,
            {"latency_ms": duration_ms, "tool_calls": len(tool_calls or []), "has_text": bool(reply)},
            thread=thread,
            event_id=llm_event_id,
        )

        if tool_calls:
            await self._emit_tool_selection(tool_calls, thread, event_id=llm_event_id)
            return None, tool_calls

        response_text = reply or ""
        await self.observability.emit(
            EventType.LLM_RESPONSE,
            {"text": response_text},
            thread=thread,
            event_id=llm_event_id,
        )
        return response_text, []

    async def _execute_tool(self, call: ToolCall, thread: Thread) -> None:
        """
        Execute the requested tool and append both the call and result to the thread.
        """
        arguments = call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                logger.warning("slim_agent.arguments_not_json", name=call.name)
                arguments = {}
        elif arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            logger.warning("slim_agent.arguments_not_dict", name=call.name, type=type(arguments).__name__)
            arguments = {}

        await self.observability.emit(
            EventType.TOOL_EXECUTION_START,
            {"tool": call.name, "arguments": arguments},
            thread=thread,
        )

        tool_call_payload = {
            "id": call.id or "",
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(arguments),
            },
        }
        thread.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[tool_call_payload],
                metadata={"internal": True, "type": "tool_call"},
            )
        )

        try:
            result: ToolResult = await asyncio.wait_for(
                self.tools.execute_tool(call.name, arguments),
                timeout=self.config.tool_timeout if self.config.tool_timeout else None,
            )
        except asyncio.TimeoutError:
            logger.warning("slim_agent.tool_timeout", tool=call.name)
            payload = {"error": f"Tool '{call.name}' timed out."}
            thread.add_message(
                Message(
                    role="tool",
                    content=json.dumps(payload),
                    name=call.name,
                    tool_call_id=call.id,
                    metadata={"internal": True, "type": "tool_result", "success": False},
                )
            )
            await self.observability.emit(
                EventType.TOOL_ERROR,
                {"tool": call.name, "error": payload["error"]},
                thread=thread,
            )
            return
        except Exception as err:
            logger.exception("slim_agent.tool_error", tool=call.name, error=str(err))
            payload = {"error": str(err)}
            thread.add_message(
                Message(
                    role="tool",
                    content=json.dumps(payload),
                    name=call.name,
                    tool_call_id=call.id,
                    metadata={"internal": True, "type": "tool_result", "success": False},
                )
            )
            await self.observability.emit(
                EventType.TOOL_ERROR,
                {"tool": call.name, "error": payload["error"]},
                thread=thread,
            )
            return

        result_payload = result.to_dict()
        content = (
            result_payload.get("llm")
            or result_payload.get("display")
            or result_payload.get("data")
            or result_payload
        )
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        thread.add_message(
            Message(
                role="tool",
                content=content,
                name=call.name,
                tool_call_id=call.id,
                metadata={"internal": True, "type": "tool_result", "success": result.success},
            )
        )
        await self.observability.emit(
            EventType.TOOL_RESULT,
            {"tool": call.name, "success": result.success, "result": result.to_dict()},
            thread=thread,
        )

    async def _notify_stream_chunk(self, text: str) -> None:
        """Forward streaming text chunks to registered listeners."""
        if not text:
            return
        for listener in list(self._stream_chunk_listeners):
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(text)
                else:
                    listener(text)
            except Exception:
                logger.exception("slim_agent.stream_listener_error")

    async def _notify_stream_end(self) -> None:
        """Signal listeners that streaming has completed for this turn."""
        for listener in list(self._stream_end_listeners):
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener()
                else:
                    listener()
            except Exception:
                logger.exception("slim_agent.stream_end_listener_error")

    async def _consume_stream(
        self,
        stream_iter,
        thread: Thread,
        event_id: str,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        async for chunk in stream_iter:
            if isinstance(chunk, dict) and chunk.get("tool_calls"):
                for raw_call in chunk.get("tool_calls", []):
                    try:
                        call = ToolCall.model_validate(raw_call)
                        tool_calls.append(call)
                    except ValidationError:
                        logger.warning("slim_agent.invalid_tool_call", raw=raw_call)
                continue

            text_piece = str(chunk)
            if not text_piece:
                continue
            text_parts.append(text_piece)
            await self._notify_stream_chunk(text_piece)

        await self._notify_stream_end()

        if tool_calls:
            return None, tool_calls
        return "".join(text_parts), []

    def _parse_completion(
        self,
        raw_response: Any,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        if isinstance(raw_response, str):
            return raw_response, []

        if isinstance(raw_response, dict):
            tool_calls_raw = raw_response.get("tool_calls") or []
            tool_calls: List[ToolCall] = []
            for item in tool_calls_raw:
                try:
                    call = ToolCall.model_validate(item)
                    tool_calls.append(call)
                except ValidationError:
                    logger.warning("slim_agent.invalid_tool_call", raw=item)
                    continue

            if tool_calls:
                return None, tool_calls

            content = raw_response.get("content")
            if isinstance(content, list):
                text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                return "".join(text_parts), []
            if content is not None:
                return str(content), []

            message = raw_response.get("message")
            if message is not None:
                return str(message), []

        return str(raw_response), []

    async def _emit_tool_selection(
        self,
        tool_calls: List[ToolCall],
        thread: Thread,
        *,
        event_id: str,
    ) -> None:
        for index, call in enumerate(tool_calls, start=1):
            await self.observability.emit(
                EventType.TOOL_SELECTION,
                {"tool": call.name, "arguments": call.arguments, "index": index},
                thread=thread,
                event_id=event_id,
            )

    @staticmethod
    def _elapsed_ms(start_ns: int) -> int:
        return max(1, int(max(0, time.perf_counter_ns() - start_ns) / 1_000_000))

    @staticmethod
    def _log_event_debug(event: Event) -> None:
        payload = event.content
        if isinstance(payload, dict):
            preview = payload.copy()
            text = preview.get("text")
            if isinstance(text, str) and len(text) > 200:
                preview["text"] = f"{text[:200]}…"
        else:
            preview = payload
        logger.info(
            "slim_agent.event",
            event_type=event.type.value,
            metadata=event.metadata,
            payload=preview,
        )
