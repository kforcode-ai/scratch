"""
Conversation primitives for MiniAgent Framework.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime
import hashlib
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

from .events import Event, EventType
from .logging import logger

@dataclass
class Message:
    """A message in the conversation"""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[Dict]] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool result messages
    name: Optional[str] = None  # Tool name for tool responses

    def to_dict(self) -> Dict:
        result = {
            "role": self.role,
            "content": self.content
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result


class Thread:
    """
    Conversation thread with enhanced event history and checkpointing
    """
    
    def __init__(self, thread_id: Optional[str] = None):
        self.id = thread_id or self._generate_id()
        self.messages: List[Message] = []
        self.events: List[Event] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self._checkpoint_every = 10  # Auto-checkpoint every N events
        self._checkpoints: List[Dict] = []  # Store checkpoints
    
    def _generate_id(self) -> str:
        """Generate unique thread ID"""
        return str(uuid.uuid4())[:8]
    
    def add_message(self, message: Message):
        """Add a message to the thread with event tracking"""
        self.messages.append(message)

        # Add corresponding event for audit trail
        event_type = EventType.USER_INPUT if message.role == "user" else EventType.LLM_RESPONSE
        self.add_event(Event(type=event_type, data=message.to_dict()))

        return self
    
    def add_event(self, event: Event):
        """Add an event to the history with auto-checkpointing"""
        self.events.append(event)
        
        # Auto-checkpoint for fast recovery
        if len(self.events) % self._checkpoint_every == 0:
            self.create_checkpoint()
        
        return self
    
    def create_checkpoint(self) -> Dict:
        """Create a state snapshot for fast recovery"""
        checkpoint = {
            "thread_id": self.id,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "event_count": len(self.events),
            "last_events": [e.to_dict() for e in self.events[-5:]],
            "state_hash": self._compute_state_hash()
        }
        self._checkpoints.append(checkpoint)
        return checkpoint
    
    def _compute_state_hash(self) -> str:
        """Compute hash of current state for verification"""
        state_str = json.dumps([e.to_dict() for e in self.events], default=str)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def to_redis_dict(self) -> Dict:
        """Serialize thread for Redis storage"""
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "checkpoints": self._checkpoints
        }
    
    @classmethod
    def from_redis_dict(cls, data: Dict) -> "Thread":
        """Deserialize thread from Redis"""
        thread = cls(thread_id=data["id"])
        
        # Restore messages
        for msg_data in data.get("messages", []):
            thread.messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                metadata=msg_data.get("metadata", {}),
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
                name=msg_data.get("name"),
            ))
        
        # Restore events
        for event_data in data.get("events", []):
            # Find the EventType by value, not by name
            event_type_value = event_data["type"]
            event_type = None
            for et in EventType:
                if et.value == event_type_value:
                    event_type = et
                    break
            if event_type:
                thread.events.append(Event(
                    type=event_type,
                    data=event_data["data"]
                ))
        
        thread.metadata = data.get("metadata", {})
        thread._checkpoints = data.get("checkpoints", [])
        return thread

    # ------------------------------------------------------------------
    # Context utilities inspired by HICA
    # ------------------------------------------------------------------

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value in the thread metadata."""
        self.metadata[key] = value
        logger.debug("Context updated", extra={"key": key})

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a context value from metadata."""
        return self.metadata.get(key, default)

    def summarize_events(self, max_events: int = 10) -> None:
        """Truncate stored events keeping only the most recent entries."""
        if len(self.events) > max_events:
            removed = len(self.events) - max_events
            self.events = self.events[-max_events:]
            cursor = uuid.uuid4().hex[:8]
            self.metadata["events_cursor"] = cursor
            logger.info(
                "thread.events.truncated",
                thread_id=self.id,
                cursor=cursor,
                approx_events_hidden=removed,
                next_url=f"/threads/{self.id}/events?cursor={cursor}",
            )

    def awaiting_human_response(self) -> bool:
        """Return True if the latest event is a clarification request."""
        if not self.events:
            return False
        last = self.events[-1]
        data = last.data or {}
        return (
            last.type == EventType.AGENT_THINKING and
            isinstance(data, dict) and
            data.get("intent") == "clarification"
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def serialize_for_llm(self, fmt: str = "json") -> str:
        """Serialize thread events for inspection or prompt injection."""
        context_summary = (
            f"Thread Context: {json.dumps(self.metadata)}\n\n" if self.metadata else ""
        )

        def _filter_events() -> List[Event]:
            filtered: List[Event] = []
            for event in self.events:
                if event.type not in {EventType.LLM_CALL}:
                    filtered.append(event)
            return filtered

        events = _filter_events()

        if fmt == "xml":
            serialized = "\n".join(self._serialize_event_xml(e) for e in events)
        else:
            serialized = json.dumps([e.to_dict() for e in events], indent=2)
        return f"{context_summary}{serialized}"

    def _serialize_event_xml(self, event: Event) -> str:
        root = ET.Element(event.type.value)
        payload = event.data or {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                child = ET.SubElement(root, str(key))
                child.text = str(value)
        else:
            root.text = str(payload)
        return ET.tostring(root, encoding="unicode", method="xml")

    def to_json(self) -> str:
        """Serialize thread to a JSON string."""
        return json.dumps(self.to_redis_dict(), default=str, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "Thread":
        """Deserialize thread from JSON string."""
        data = json.loads(payload)
        return cls.from_redis_dict(data)

    def validate(self) -> bool:
        """Basic validation ensuring events follow the expected structure."""
        if not self.events:
            logger.warning("Thread has no events")
            return False
        for event in self.events:
            if not isinstance(event, Event):
                logger.error("Invalid event object", extra={"event": event})
                return False
            if event.data is None and event.content is None:
                logger.error("Event missing payload", extra={"type": event.type})
                return False
        return True

    def save_to_file(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load_from_file(cls, path: str) -> "Thread":
        return cls.from_json(Path(path).read_text())
    
    def get_messages_for_llm(self) -> List[Dict]:
        """Get messages formatted for LLM in OpenAI format"""
        formatted_messages = []
        for msg in self.messages:
            msg_dict = msg.to_dict()
            if msg.metadata.get("internal"):
                continue
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
