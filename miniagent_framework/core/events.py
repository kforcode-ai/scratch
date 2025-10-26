"""
Event system for tracking agent actions, streaming updates, and session management
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Callable, List
from datetime import datetime
import asyncio


class EventType(Enum):
    """Types of events in the agent lifecycle"""
    # Core events
    USER_INPUT = "user_input"
    AGENT_THINKING = "agent_thinking"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULT = "tool_result"
    AGENT_RESPONSE = "agent_response"
    RETRY = "retry"
    
    # LLM Events
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    LLM_THINKING = "llm_thinking"
    
    # Tool Events (additional detail)
    TOOL_CALL = "tool_call"
    TOOL_ERROR = "tool_error"
    
    # Streaming events
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    
    # Agent Events
    AGENT_START = "agent_start"
    AGENT_ERROR = "agent_error"
    AGENT_COMPLETE = "agent_complete"
    
    # Session Events (for Redis sessions)
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"
    SESSION_ENDED = "session_ended"
    SESSION_ERROR = "session_error"
    
    # Cache Events (for caching layer)
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_SET = "cache_set"
    
    # System Events
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Event:
    """An event in the agent's execution"""
    type: EventType
    content: Any = None  # Support both 'content' and 'data' naming
    data: Any = None     # For backward compatibility
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Support both content and data fields for backward compatibility
        if self.content is None and self.data is not None:
            self.content = self.data
        elif self.data is None and self.content is not None:
            self.data = self.content
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary"""
        return {
            "type": self.type.value,
            "content": self.content,
            "data": self.data,  # Include both for compatibility
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class StreamCallback:
    """
    Callback handler for streaming events
    Supports both single and multiple handlers per event type
    """
    
    def __init__(self, allow_multiple_handlers: bool = True):
        self.allow_multiple_handlers = allow_multiple_handlers
        self.handlers: Dict[EventType, List[Callable]] = {}
        self._default_handler: Optional[Callable] = None
    
    def on(self, event_type: EventType, handler: Callable):
        """Register a handler for an event type"""
        if self.allow_multiple_handlers:
            # Support multiple handlers per event
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
        else:
            # Single handler per event (backward compatibility)
            self.handlers[event_type] = [handler]
        return self
    
    def on_any(self, handler: Callable):
        """Register a handler for all events"""
        self._default_handler = handler
        return self
    
    def off(self, event_type: EventType, handler: Callable = None):
        """Unregister a handler"""
        if event_type in self.handlers:
            if handler and self.allow_multiple_handlers:
                # Remove specific handler
                if handler in self.handlers[event_type]:
                    self.handlers[event_type].remove(handler)
                    if not self.handlers[event_type]:
                        del self.handlers[event_type]
            else:
                # Remove all handlers for this event
                del self.handlers[event_type]
    
    async def emit(self, event: Event):
        """Emit an event to registered handlers"""
        # Call specific handlers
        if event.type in self.handlers:
            handlers = self.handlers[event.type]
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    # Log error but don't stop other handlers
                    import logging
                    logging.error(f"Error in event handler for {event.type}: {e}")
        
        # Call default handler
        if self._default_handler:
            try:
                if asyncio.iscoroutinefunction(self._default_handler):
                    await self._default_handler(event)
                else:
                    self._default_handler(event)
            except Exception as e:
                import logging
                logging.error(f"Error in default event handler: {e}")
    
    # Convenience methods for common events
    def on_thinking(self, handler: Callable):
        """Shortcut for thinking events"""
        return self.on(EventType.AGENT_THINKING, handler)
    
    def on_tool_use(self, handler: Callable):
        """Shortcut for tool events"""
        return self.on(EventType.TOOL_EXECUTION, handler)
    
    def on_streaming(self, handler: Callable):
        """Shortcut for streaming events"""
        return self.on(EventType.STREAM_CHUNK, handler)
    
    def on_error(self, handler: Callable):
        """Shortcut for error events"""
        return self.on(EventType.ERROR, handler)
    
    def on_session_event(self, handler: Callable):
        """Register handler for all session events"""
        self.on(EventType.SESSION_CREATED, handler)
        self.on(EventType.SESSION_RESUMED, handler)
        self.on(EventType.SESSION_ENDED, handler)
        self.on(EventType.SESSION_ERROR, handler)
        return self
    
    def clear(self):
        """Clear all handlers"""
        self.handlers.clear()
        self._default_handler = None