"""
Core agent implementation with conversation threads and streaming
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import asyncio
from datetime import datetime

from .events import Event, EventType, StreamCallback
from .llm import LLMClient, RetryPolicy, LLMProvider
from .tools import ToolRegistry, ToolResult


@dataclass
class Message:
    """A message in the conversation"""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[Dict]] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool result messages

    def to_dict(self) -> Dict:
        result = {
            "role": self.role,
            "content": self.content
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


class Thread:
    """
    Conversation thread with event history
    """
    
    def __init__(self, thread_id: Optional[str] = None):
        self.id = thread_id or self._generate_id()
        self.messages: List[Message] = []
        self.events: List[Event] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
    
    def _generate_id(self) -> str:
        """Generate unique thread ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def add_message(self, message: Message):
        """Add a message to the thread"""
        self.messages.append(message)
        return self
    
    def add_event(self, event: Event):
        """Add an event to the history"""
        self.events.append(event)
        return self
    
    def get_messages_for_llm(self) -> List[Dict]:
        """Get messages formatted for LLM in OpenAI format"""
        formatted_messages = []
        for msg in self.messages:
            msg_dict = msg.to_dict()
            # Skip internal tool execution markers for APIs
            if msg.role == "assistant" and msg.content.startswith("[Tool:"):
                continue
            else:
                formatted_messages.append(msg_dict)
        return formatted_messages
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None


@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str = "Assistant"
    system_prompt: str = "You are a helpful AI assistant."
    provider: str = "openai"  # "openai", "gemini", "anthropic"
    model: Optional[str] = None  # Auto-selects default for provider if None
    api_key: Optional[str] = None  # Provider API key (uses env vars if None)
    temperature: float = 0.1
    max_tokens: int = 2000
    retry_policy: Optional[RetryPolicy] = None
    stream_by_default: bool = True


class Agent:
    """
    Main agent class with tool use and streaming

    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[ToolRegistry] = None,
        callbacks: Optional[StreamCallback] = None
    ):
        self.config = config or AgentConfig()
        self.tools = tools or ToolRegistry()
        self.callbacks = callbacks or StreamCallback()
        self.llm = LLMClient(
            provider=self.config.provider,
            api_key=self.config.api_key,
            model=self.config.model,
            retry_policy=self.config.retry_policy
        )
    
    async def run(
        self,
        user_input: str,
        thread: Optional[Thread] = None,
        context: Optional[str] = None,
        stream: Optional[bool] = None,
        max_iterations: int = 10
    ) -> str:
        """
        Main agent execution loop with proper iteration and state management
        Inspired by the agent runtime pattern
        """
        # Create or use thread
        if thread is None:
            thread = Thread()

        # Add user message
        thread.add_message(Message("user", user_input))
        thread.add_event(Event(EventType.USER_INPUT, user_input))

        # Emit user input event
        await self.callbacks.emit(Event(EventType.USER_INPUT, user_input))

        # Check if llm client available
        if self.llm.client is None:
            raise RuntimeError("No LLM client is configured. Please provide a valid LLM client.")
        return await self._run(thread, context, stream)

    async def _run(
        self,
        thread: Thread,
        context: Optional[str],
        stream: Optional[bool],
        max_agent_iterations: int = 5
    ) -> str:
        """Run with LLM API using proper agent loop"""

        for iteration in range(max_agent_iterations):
            try:
                # Build messages for LLM
                messages = thread.get_messages_for_llm()

                # Debug: show messages being sent to LLM (uncomment for debugging)
                # if iteration > 0:
                #     print(f"DEBUG: Iteration {iteration} - sending {len(messages)} messages to LLM:")
                #     for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
                #         print(f"  {len(messages)-3+i}: {msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}...")

                # Add system prompt for first iteration, or tool result instructions for subsequent iterations
                if iteration == 0:
                    messages.insert(0, {
                        "role": "system",
                        "content": self.config.system_prompt
                    })

                    # Add context
                    if context:
                        messages.append({"role": "system", "content": f"Context: {context}"})
                else:
                    # For subsequent iterations, add instructions about using tool results
                    messages.insert(0, {
                        "role": "system",
                        "content": "You have received tool results. Use the actual data provided in the tool messages to give accurate, direct answers. Do not say you don't have access to data. Extract and summarize the relevant information from the tool results."
                    })

                # Emit LLM call event
                await self.callbacks.emit(Event(EventType.LLM_CALL, {"messages": len(messages), "iteration": iteration}))

                # Get available tools (only in first iteration to avoid loops)
                tools = []
                if iteration == 0 and self.tools:
                    tools = self.tools.get_schemas()

                if stream and iteration == 0:  # Only stream the first response
                    # Stream initial response
                    accumulated = ""
                    tool_calls_detected = []

                    await self.callbacks.emit(Event(EventType.STREAM_START, None))

                    stream_gen = await self.llm.complete(
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=tools,
                        stream=True
                    )

                    async for chunk in stream_gen:
                        if isinstance(chunk, dict) and "tool_calls" in chunk:
                            # Handle tool calls in streaming
                            tool_calls_detected = chunk["tool_calls"]
                            await self.callbacks.emit(Event(EventType.AGENT_THINKING, {"tool_calls": tool_calls_detected}))
                        elif isinstance(chunk, str):
                            accumulated += chunk
                            await self.callbacks.emit(Event(EventType.STREAM_CHUNK, chunk))

                    await self.callbacks.emit(Event(EventType.STREAM_END, accumulated))

                    if tool_calls_detected:
                        # Format and add tool calls to thread
                        formatted_tool_calls = []
                        for tc in tool_calls_detected:
                            formatted_tc = {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"]
                                }
                            }
                            formatted_tool_calls.append(formatted_tc)

                        thread.add_message(Message("assistant", accumulated, tool_calls=formatted_tool_calls))

                        # Execute tools
                        await self._execute_tools(tool_calls_detected, thread)
                        continue  # Continue to next iteration for final response
                    else:
                        # Direct response
                        thread.add_message(Message("assistant", accumulated))
                        thread.add_event(Event(EventType.AGENT_RESPONSE, accumulated))
                        return accumulated

                else:
                    # Non-streaming call (for tool result processing)
                    response = await self.llm.complete(
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=tools,
                        stream=False
                    )

                    # Handle tool calls if present
                    if isinstance(response, dict) and "tool_calls" in response:
                        # Format tool_calls for OpenAI API
                        formatted_tool_calls = []
                        for tc in response["tool_calls"]:
                            formatted_tc = {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"]
                                }
                            }
                            formatted_tool_calls.append(formatted_tc)

                        # Add the assistant message with tool calls to thread
                        thread.add_message(Message("assistant", response.get("content", ""), tool_calls=formatted_tool_calls))
                        await self.callbacks.emit(Event(EventType.AGENT_THINKING, {"tool_calls": formatted_tool_calls}))

                        # Execute tools
                        tool_calls = response["tool_calls"]
                        await self._execute_tools(tool_calls, thread)
                        continue  # Continue to next iteration for final response

                    else:
                        # Final response (no more tool calls)
                        content = response if isinstance(response, str) else response.get("content", "")
                        thread.add_message(Message("assistant", content))
                        thread.add_event(Event(EventType.AGENT_RESPONSE, content))
                        return content

            except Exception as e:
                error_msg = f"Error in agent loop iteration {iteration}: {str(e)}"
                await self.callbacks.emit(Event(EventType.ERROR, error_msg))
                return f"Sorry, I encountered an error: {str(e)}"

        # Max iterations reached
        return "I apologize, but I couldn't complete the task within the allowed iterations. Please try rephrasing your request."
    
    async def _decide_action(self, thread: Thread, context: Optional[str]) -> Dict:
        """
        Decide what action to take
        """
        # Build prompt for decision
        messages = thread.get_messages_for_llm()
        
        # Add system prompt
        system_msg = {
            "role": "system",
            "content": f"""{self.config.system_prompt}

You can:
1. THINK - Internal reasoning (return: {{"action": "think", "content": "your thoughts"}})
2. USE_TOOLS - Use available tools (return: {{"action": "use_tools", "tools": [...]}})
3. RESPOND - Direct response (return: {{"action": "respond"}})

Available tools:
{json.dumps([t.to_function_schema() for t in self.tools.tools.values()], indent=2)}

Respond with JSON indicating your decision."""
        }
        
        decision_messages = [system_msg] + messages
        
        # Add context if provided
        if context:
            decision_messages.append({"role": "system", "content": f"Context: {context}"})
        
        # Get decision
        response = await self.llm.complete(decision_messages, stream=False)
        
        # Parse decision
        try:
            if isinstance(response, str):
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                else:
                    # Default to respond
                    decision = {"action": "respond"}
            elif isinstance(response, dict):
                decision = response
            else:
                # Default to respond for unexpected types
                decision = {"action": "respond"}
        except Exception as e:
            # Log the error for debugging
            print(f"Decision parsing error: {e}, response: {response}")
            decision = {"action": "respond"}
        
        # Ensure decision is always a dictionary
        if not isinstance(decision, dict):
            decision = {"action": "respond"}

        thread.add_event(Event(EventType.AGENT_THINKING, decision))
        return decision
    
    async def _think_and_continue(self, thought: str, thread: Thread, context: Optional[str]) -> str:
        """Handle thinking step"""
        # Emit thinking event
        await self.callbacks.emit(Event(EventType.AGENT_THINKING, thought))
        
        # Add thinking to thread
        thread.add_message(Message("assistant", f"[Thinking: {thought}]"))
        
        # Continue with next decision
        return await self.run(
            user_input=thread.get_last_user_message() or "",
            thread=thread,
            context=context
        )
    
    async def _execute_tools(self, tool_calls: List[Dict], thread: Thread) -> None:
        """Execute tools and add results to thread in OpenAI format"""

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
            arguments = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments", {})

            # Parse parameters if string
            if isinstance(arguments, str):
                try:
                    parameters = json.loads(arguments)
                except:
                    parameters = {}
            else:
                parameters = arguments

            # Emit tool execution event
            await self.callbacks.emit(Event(
                EventType.TOOL_EXECUTION,
                {"tool": tool_name, "parameters": parameters, "tool_call_id": tool_call_id}
            ))

            # Execute tool
            result = await self.tools.execute(tool_name, parameters)

            # Emit tool result event
            await self.callbacks.emit(Event(
                EventType.TOOL_RESULT,
                result.to_dict()
            ))

            # Add tool result message in OpenAI format
            tool_content = result.llm_content or str(result.data)
            thread.add_message(Message(
                role="tool",
                content=tool_content,
                tool_call_id=tool_call_id
            ))
            thread.add_event(Event(
                EventType.TOOL_RESULT,
                result.to_dict()
            ))
    
    async def _generate_response(
        self,
        thread: Thread,
        context: Optional[str],
        stream: Optional[bool] = None
    ) -> str:
        """Generate the final response"""
        
        stream = stream if stream is not None else self.config.stream_by_default
        
        # Build messages
        messages = thread.get_messages_for_llm()
        
        # Add system prompt
        messages.insert(0, {
            "role": "system",
            "content": self.config.system_prompt
        })
        
        # Add context
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        
        # Emit LLM call event
        await self.callbacks.emit(Event(EventType.LLM_CALL, {"messages": len(messages)}))
        
        if stream:
            # Stream response
            accumulated = ""
            await self.callbacks.emit(Event(EventType.STREAM_START, None))
            
            # Get the async generator
            stream_gen = await self.llm.complete(messages, stream=True)
            
            # Iterate over the generator
            async for chunk in stream_gen:
                accumulated += chunk
                await self.callbacks.emit(Event(EventType.STREAM_CHUNK, chunk))
            
            await self.callbacks.emit(Event(EventType.STREAM_END, accumulated))
            
            # Add to thread
            thread.add_message(Message("assistant", accumulated))
            thread.add_event(Event(EventType.AGENT_RESPONSE, accumulated))
            
            return accumulated
        else:
            # Single response
            response = await self.llm.complete(messages, stream=False)
            
            # Handle tool calls in response
            if isinstance(response, dict) and response.get("type") == "tool_calls":
                return await self._use_tools_and_respond(
                    response["tool_calls"],
                    thread,
                    context
                )
            
            # Emit response event
            await self.callbacks.emit(Event(EventType.LLM_RESPONSE, response))
            
            # Add to thread
            thread.add_message(Message("assistant", response))
            thread.add_event(Event(EventType.AGENT_RESPONSE, response))
            
            return response
