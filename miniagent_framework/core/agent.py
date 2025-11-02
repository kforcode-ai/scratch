"""Lean agent runtime inspired by HICA with hybrid planning/decision flow."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Literal, Type, Union, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, model_validator

from .core import Message, Thread
from .events import Event, EventType, StreamCallback
from .llm import LLMClient, RetryPolicy
from .logging import logger
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
    retry_policy: Optional[RetryPolicy] = None

    @model_validator(mode="after")
    def _validate(self) -> "AgentConfig":
        object.__setattr__(self, "provider", self.provider.lower())
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
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
        logger.info(
            "agent.run.start",
            user_input=user_input,
            thread_id=thread.id,
            stream=stream,
        )
        message = Message("user", user_input)
        thread.add_message(message)

        try:
            async for _ in self.agent_loop(
                thread,
                context=context,
                stream=stream,
                max_iterations=max_iterations,
            ):
                pass
        except Exception as exc:
            logger.error(
                "agent.run.exception",
                error=str(exc),
                exc_type=type(exc).__name__,
                thread_id=thread.id,
            )
            logger.exception("agent.run.exception_trace")
            thread.add_event(Event(EventType.ERROR, {"error": str(exc)}))
            if isinstance(exc, RuntimeError):
                fallback = f"Configuration issue: {exc}"
            else:
                fallback = "I'm sorry, I ran into an internal error while responding."
            thread.add_event(Event(EventType.AGENT_RESPONSE, fallback))
            await self.callbacks.emit(Event(EventType.AGENT_RESPONSE, fallback))
            return fallback

        for event in reversed(thread.events):
            if event.type == EventType.AGENT_RESPONSE and isinstance(event.data, str):
                return event.data

        fallback = "I'm sorry, I couldn't produce an answer this time."
        thread.add_event(Event(EventType.AGENT_RESPONSE, fallback))
        await self.callbacks.emit(Event(EventType.AGENT_RESPONSE, fallback))
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

        await self._ensure_plan(thread, context)
        yield thread

        use_stream = bool(stream)

        for iteration in range(max_iterations):
            action = await self._decide_next_action(thread, context, iteration, use_stream)

            if action.action == "tool" and action.tool:
                if action.message:
                    thread.add_event(
                        Event(
                            EventType.AGENT_THINKING,
                            {"tool": action.tool, "note": action.message},
                        )
                    )
                await self._execute_tool(
                    action.tool,
                    action.arguments or {},
                    thread,
                    call_id=action.tool_call_id,
                )
                yield thread
                continue

            if action.action == "clarification" and action.message:
                clarification_msg = action.message
                thread.add_message(Message("assistant", clarification_msg))
                thread.add_event(Event(EventType.AGENT_RESPONSE, clarification_msg))
                await self.callbacks.emit(Event(EventType.AGENT_RESPONSE, clarification_msg))
                logger.info(
                    "agent.clarification_requested",
                    message=clarification_msg,
                    thread_id=thread.id,
                )
                yield thread
                return

            if action.action == "final" and action.message:
                await self._commit_final_response(action.message, action.summary, thread)
                yield thread
                return

            if action.action == "fallback":
                await self._commit_final_response(action.message or "", action.summary, thread)
                yield thread
                return

            logger.warning(
                "agent.unhandled_action",
                action=action.action,
                thread_id=thread.id,
            )
            await self._finalize_response(thread, context)
            yield thread
            return

        logger.warning("Max iterations reached without completion", thread_id=thread.id)
        await self._finalize_response(thread, context)
        yield thread

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _llm_json_call(
        self,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel],
        thread: Optional[Thread] = None,
    ) -> BaseModel:
        await self.callbacks.emit(Event(EventType.LLM_CALL, {"messages": len(messages)}))
        logger.debug(
            "agent.llm.request",
            message_count=len(messages),
            last_user_message=next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                None,
            ),
            thread_id=thread.id if thread else None,
        )

        raw = await self.llm.complete(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
            tools=None,
        )

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

        logger.debug(
            "agent.llm.validated_response",
            model=response_model.__name__,
            thread_id=thread.id if thread else None,
        )
        return validated

    async def _decide_next_action(
        self,
        thread: Thread,
        context: Optional[str],
        iteration: int,
        stream: bool,
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
        await self.callbacks.emit(Event(EventType.LLM_CALL, {"messages": len(messages)}))
        logger.debug(
            "agent.decision.request",
            thread_id=thread.id,
            iteration=iteration,
        )

        tool_schemas = (
            self.tool_registry.get_schemas() if self.tool_registry.tools else None
        )

        if stream:
            tool_calls_payload, final_text = await self._stream_decision_response(
                messages, tool_schemas, thread
            )
        else:
            raw_response = await self.llm.complete(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=False,
                tools=tool_schemas,
            )
            logger.debug(
                "agent.decision.raw_response",
                thread_id=thread.id,
                response=str(raw_response)[:500],
            )
            tool_calls_payload = self._extract_tool_calls(raw_response)
            final_text = self._extract_text_response(raw_response)

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
                    await self.callbacks.emit(
                        Event(
                            EventType.LLM_RESPONSE,
                            {
                                "action": "tool",
                                "tool": next_call["name"],
                                "arguments": next_call["arguments"],
                                "tool_call_id": next_call["id"],
                            },
                        )
                    )
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

        if final_text is None or not str(final_text).strip():
            logger.warning(
                "agent.decision.empty_response",
                thread_id=thread.id,
                iteration=iteration,
            )
            return ActionResult(
                action="fallback",
                message="I'm sorry, I wasn't able to generate a response just now.",
            )

        thread.add_event(
            Event(
                EventType.LLM_RESPONSE,
                {"action": "final", "message": final_text},
            )
        )
        await self.callbacks.emit(
            Event(EventType.LLM_RESPONSE, {"action": "final", "message": final_text})
        )
        logger.info(
            "agent.intent.selected",
            action="final",
            tool=None,
            thread_id=thread.id,
        )
        return ActionResult(action="final", message=final_text)

    async def _stream_decision_response(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        thread: Thread,
    ) -> Tuple[List[Dict[str, Any]], str]:
        stream_iter = await self.llm.complete(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            tools=tool_schemas,
        )

        chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        stream_started = False

        async for chunk in stream_iter:
            if isinstance(chunk, str):
                if not stream_started:
                    stream_started = True
                    await self.callbacks.emit(Event(EventType.STREAM_START, None))
                chunks.append(chunk)
                await self.callbacks.emit(Event(EventType.STREAM_CHUNK, chunk))
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
            await self.callbacks.emit(Event(EventType.STREAM_END, None))

        final_text = "".join(chunks)
        if final_text:
            logger.debug(
                "agent.decision.stream.final_text_preview",
                preview=final_text[:200],
                thread_id=thread.id,
            )

        return tool_calls, final_text

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

    async def _finalize_response(self, thread: Thread, context: Optional[str]) -> None:
        instruction = (
            "Summarize the conversation and answer the user's request. "
            "Respond as JSON with keys 'message' and optional 'summary'."
        )
        plan_index = thread.metadata.get("plan_index", 0)
        messages = self._build_messages(
            instruction, thread, context, plan_index, mode="final"
        )
        try:
            response = await self._llm_json_call(messages, FinalResponseModel, thread)
        except Exception as exc:
            logger.error("Final response generation failed", error=str(exc))
            fallback = "I'm sorry, I couldn't generate a response right now."
            thread.add_message(Message("assistant", fallback))
            thread.add_event(Event(EventType.AGENT_RESPONSE, fallback))
            await self.callbacks.emit(Event(EventType.AGENT_RESPONSE, fallback))
            return

        await self._commit_final_response(response.message, response.summary, thread)

    async def _execute_tool(
        self,
        intent: str,
        arguments: Dict[str, Any],
        thread: Thread,
        *,
        call_id: Optional[str] = None,
    ) -> None:
        thread.add_event(
            Event(EventType.TOOL_CALL, {"tool": intent, "arguments": arguments})
        )
        await self.callbacks.emit(
            Event(EventType.TOOL_EXECUTION, {"tool": intent, "parameters": arguments})
        )
        logger.info(
            "agent.tool.execute",
            tool=intent,
            arguments=json.dumps(arguments) if arguments else "{}",
        )

        result: ToolResult = await self.tool_registry.execute_tool(intent, arguments)

        payload = result.to_dict()
        thread.add_event(Event(EventType.TOOL_RESULT, payload))
        await self.callbacks.emit(Event(EventType.TOOL_RESULT, payload))
        logger.info(
            "agent.tool.result",
            tool=intent,
            success=result.success,
        )

        summary = payload.get("llm") or payload.get("display") or payload.get("data")
        if isinstance(summary, (dict, list)):
            summary = json.dumps(summary)
        thread.add_event(
            Event(
                EventType.AGENT_THINKING,
                {"tool": intent, "result": summary},
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
        self, message: Union[str, Dict[str, Any]], summary: Optional[Any], thread: Thread
    ) -> None:
        final_message = message
        if not isinstance(final_message, str):
            final_message = json.dumps(final_message, ensure_ascii=False)

        thread.metadata.pop("pending_tool_calls", None)
        thread.add_message(Message("assistant", final_message))
        thread.add_event(Event(EventType.AGENT_RESPONSE, final_message))
        await self.callbacks.emit(Event(EventType.AGENT_RESPONSE, final_message))
        logger.info(
            "agent.final_response",
            message=final_message,
            summary=summary,
            thread_id=thread.id,
        )
        if summary is not None:
            thread.add_event(Event(EventType.INFO, {"summary": summary}))
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
                "- When no tool is needed, reply to the user in natural language.\n"
                "- Always keep the user-facing response concise and helpful."
            )
        elif mode == "final":
            system_content += (
                "\n\nCompose the final user-facing answer summarizing tool evidence."
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

    async def _ensure_plan(self, thread: Thread, context: Optional[str]) -> None:
        if not self.config.planning_enabled or thread.metadata.get("plan"):
            return

        last_user_message = thread.get_last_user_message() or ""
        if not self._should_generate_plan(thread, last_user_message, context):
            logger.debug(
                "agent.plan.skipped",
                thread_id=thread.id,
                reason="heuristic",
            )
            return

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

        try:
            plan_response = await self._llm_json_call(planning_messages, PlanResponseModel, thread)
            normalized_steps, parser_used = self._normalize_plan_steps(plan_response)
        except Exception as exc:
            logger.warning(
                "agent.plan.failed",
                error=str(exc),
                thread_id=thread.id,
            )
            return

        if not normalized_steps:
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

        thread.add_message(
            Message(
                "assistant",
                f"[Plan]\n{plan_summary}",
                metadata={"internal": True, "type": "plan", "exclude_from_prompt": True},
            )
        )
        thread.add_event(
            Event(
                EventType.AGENT_THINKING,
                {"plan": plan_summary, "steps": normalized_steps},
            )
        )
        logger.info(
            "agent.plan.created",
            thread_id=thread.id,
            steps=len(normalized_steps),
            parser=parser_used,
        )

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
        thread.add_event(Event(EventType.AGENT_THINKING, progress_payload))
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
            Event(EventType.AGENT_THINKING, {"plan_complete": True, "total_steps": len(plan_steps)})
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
