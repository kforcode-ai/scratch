"""
Event system for tracking agent actions and streaming updates
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Callable
from datetime import datetime


class EventType(Enum):
    """Types of events in the agent lifecycle"""
    # Core events
    USER_INPUT = "user_input"
    AGENT_THINKING = "agent_thinking"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULT = "tool_result"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    AGENT_RESPONSE = "agent_response"
    ERROR = "error"
    RETRY = "retry"
    
    # Streaming events
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"


@dataclass
class Event:
    """An event in the agent's execution"""
    type: EventType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary"""
        return {
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class StreamCallback:
    """Callback handler for streaming events"""
    
    def __init__(self):
        self.handlers: Dict[EventType, Callable] = {}
    
    def on(self, event_type: EventType, handler: Callable):
        """Register a handler for an event type"""
        self.handlers[event_type] = handler
        return self
    
    async def emit(self, event: Event):
        """Emit an event to registered handlers"""
        if event.type in self.handlers:
            handler = self.handlers[event.type]
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
    
    def on_thinking(self, handler: Callable):
        """Shortcut for thinking events"""
        return self.on(EventType.AGENT_THINKING, handler)
    
    def on_tool_use(self, handler: Callable):
        """Shortcut for tool events"""
        return self.on(EventType.TOOL_EXECUTION, handler)
    
    def on_streaming(self, handler: Callable):
        """Shortcut for streaming events"""
        return self.on(EventType.STREAM_CHUNK, handler)


import asyncio
