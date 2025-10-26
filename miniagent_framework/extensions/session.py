"""
Session-aware agent with Redis-backed persistence
"""
import uuid
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ..core.core import Agent, Thread, Message, AgentConfig
from .redis_client import RedisSessionManager, RedisConfig, get_redis_manager
from ..core.events import Event, EventType, StreamCallback
from ..core.tools import ToolRegistry

logger = logging.getLogger(__name__)


class SessionThread(Thread):
    """
    Enhanced Thread with Redis persistence
    """
    
    def __init__(self, session_id: str, redis_manager: RedisSessionManager):
        super().__init__()
        self.session_id = session_id
        self.redis_manager = redis_manager
        self._dirty = False  # Track if thread needs saving
    
    def add_message(self, message: Message):
        """Add message and mark thread as dirty"""
        super().add_message(message)
        self._dirty = True
    
    def add_event(self, event: Event):
        """Add event and mark thread as dirty"""
        super().add_event(event)
        self._dirty = True
    
    async def save(self) -> bool:
        """Save thread to Redis"""
        if not self._dirty:
            return True
        
        thread_data = {
            "messages": [msg.to_dict() for msg in self.messages],
            "events": [
                {
                    "type": event.type.value,
                    "data": event.data,
                    "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat()
                }
                for event in self.events
            ]
        }
        
        success = await self.redis_manager.save_thread(self.session_id, thread_data)
        if success:
            self._dirty = False
        return success
    
    @classmethod
    async def load(cls, session_id: str, redis_manager: RedisSessionManager) -> Optional['SessionThread']:
        """Load thread from Redis"""
        thread_data = await redis_manager.load_thread(session_id)
        
        if not thread_data:
            return None
        
        thread = cls(session_id, redis_manager)
        
        # Restore messages
        for msg_dict in thread_data.get("messages", []):
            message = Message(
                role=msg_dict["role"],
                content=msg_dict["content"],
                tool_calls=msg_dict.get("tool_calls"),
                tool_call_id=msg_dict.get("tool_call_id")
            )
            thread.messages.append(message)
        
        # Restore events
        for event_dict in thread_data.get("events", []):
            try:
                event = Event(
                    type=EventType(event_dict["type"]),
                    data=event_dict.get("data")
                )
                thread.events.append(event)
            except ValueError:
                logger.warning(f"Unknown event type: {event_dict.get('type')}")
        
        thread._dirty = False
        return thread


class SessionAgent(Agent):
    """
    Session-aware agent with Redis-backed conversation persistence
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[ToolRegistry] = None,
        callbacks: Optional[StreamCallback] = None,
        redis_config: Optional[RedisConfig] = None
    ):
        super().__init__(config, tools, callbacks)
        self.redis_config = redis_config or RedisConfig()
        self.redis_manager: Optional[RedisSessionManager] = None
        self.sessions: Dict[str, SessionThread] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize Redis connection and start cleanup task"""
        self.redis_manager = await get_redis_manager(self.redis_config)
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("SessionAgent initialized with Redis backend")
    
    async def shutdown(self):
        """Cleanup resources"""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save all active sessions
        for session_id, thread in self.sessions.items():
            await thread.save()
        
        logger.info("SessionAgent shutdown complete")
    
    async def create_session(
        self, 
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "agent_name": self.config.name,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Create session in Redis
        await self.redis_manager.create_session(session_id, session_data)
        
        # Create thread
        thread = SessionThread(session_id, self.redis_manager)
        self.sessions[session_id] = thread
        
        # Emit session created event
        await self.callbacks.emit(Event(
            EventType.SESSION_CREATED,
            {"session_id": session_id, "user_id": user_id}
        ))
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    async def resume_session(self, session_id: str) -> bool:
        """
        Resume an existing session
        
        Returns:
            bool: True if session was resumed successfully
        """
        # Check if already loaded
        if session_id in self.sessions:
            return True
        
        # Load from Redis
        session_data = await self.redis_manager.get_session(session_id)
        if not session_data:
            logger.warning(f"Session {session_id} not found")
            return False
        
        # Load thread
        thread = await SessionThread.load(session_id, self.redis_manager)
        if not thread:
            # Create new thread if not found
            thread = SessionThread(session_id, self.redis_manager)
        
        self.sessions[session_id] = thread
        
        # Emit session resumed event
        await self.callbacks.emit(Event(
            EventType.SESSION_RESUMED,
            {"session_id": session_id}
        ))
        
        logger.info(f"Resumed session {session_id}")
        return True
    
    async def end_session(self, session_id: str) -> bool:
        """
        End and save a session
        
        Returns:
            bool: True if session was ended successfully
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not active")
            return False
        
        thread = self.sessions[session_id]
        
        # Save thread
        await thread.save()
        
        # Update session metadata
        session_data = await self.redis_manager.get_session(session_id)
        if session_data:
            session_data["ended_at"] = datetime.now().isoformat()
            session_data["total_messages"] = len(thread.messages)
            await self.redis_manager.update_session(session_id, session_data)
        
        # Remove from active sessions
        del self.sessions[session_id]
        
        # Emit session ended event
        await self.callbacks.emit(Event(
            EventType.SESSION_ENDED,
            {"session_id": session_id}
        ))
        
        logger.info(f"Ended session {session_id}")
        return True
    
    async def run_session(
        self,
        session_id: str,
        user_input: str,
        context: Optional[str] = None,
        stream: Optional[bool] = None
    ) -> str:
        """
        Run agent with session context
        
        Args:
            session_id: Session identifier
            user_input: User's input message
            context: Optional additional context
            stream: Whether to stream the response
            
        Returns:
            Agent's response
        """
        # Ensure session is loaded
        if session_id not in self.sessions:
            resumed = await self.resume_session(session_id)
            if not resumed:
                # Create new session if not found
                session_id = await self.create_session()
        
        thread = self.sessions[session_id]
        
        # Run agent with thread
        response = await self.run(
            user_input=user_input,
            thread=thread,
            context=context,
            stream=stream
        )
        
        # Auto-save thread after each interaction
        await thread.save()
        
        # Update session last activity
        await self.redis_manager.extend_session_ttl(session_id, 600)  # Extend by 10 minutes
        
        return response
    
    async def get_session_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            resumed = await self.resume_session(session_id)
            if not resumed:
                return None
        
        thread = self.sessions[session_id]
        return [msg.to_dict() for msg in thread.messages]
    
    async def list_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all active sessions, optionally filtered by user"""
        sessions = await self.redis_manager.list_sessions()
        
        active_sessions = []
        for session_id in sessions:
            session_data = await self.redis_manager.get_session(session_id)
            if session_data:
                if user_id and session_data.get("user_id") != user_id:
                    continue
                active_sessions.append({
                    "session_id": session_id,
                    "user_id": session_data.get("user_id"),
                    "created_at": session_data.get("created_at"),
                    "last_activity": session_data.get("last_activity"),
                    "message_count": session_data.get("total_messages", 0)
                })
        
        return active_sessions
    
    async def clear_session_history(self, session_id: str) -> bool:
        """Clear conversation history while keeping session active"""
        if session_id not in self.sessions:
            resumed = await self.resume_session(session_id)
            if not resumed:
                return False
        
        thread = self.sessions[session_id]
        thread.messages.clear()
        thread.events.clear()
        thread._dirty = True
        
        await thread.save()
        
        logger.info(f"Cleared history for session {session_id}")
        return True
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions from memory"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Get all sessions from Redis
                redis_sessions = set(await self.redis_manager.list_sessions())
                
                # Find sessions in memory that are no longer in Redis
                expired = []
                for session_id in self.sessions:
                    if session_id not in redis_sessions:
                        expired.append(session_id)
                
                # Remove expired sessions from memory
                for session_id in expired:
                    del self.sessions[session_id]
                    logger.info(f"Removed expired session {session_id} from memory")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")


class SessionManager:
    """
    High-level session management interface
    """
    
    def __init__(self, agent: SessionAgent):
        self.agent = agent
    
    async def create_or_resume_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Create new session or resume existing one"""
        if session_id:
            success = await self.agent.resume_session(session_id)
            if success:
                return session_id
        
        return await self.agent.create_session(user_id=user_id)
    
    async def chat(
        self,
        session_id: str,
        message: str,
        stream: bool = True
    ) -> str:
        """Send message and get response"""
        return await self.agent.run_session(
            session_id=session_id,
            user_input=message,
            stream=stream
        )
    
    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return await self.agent.get_session_history(session_id) or []
    
    async def clear_history(self, session_id: str) -> bool:
        """Clear conversation history"""
        return await self.agent.clear_session_history(session_id)
    
    async def end_session(self, session_id: str) -> bool:
        """End session"""
        return await self.agent.end_session(session_id)
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active sessions"""
        return await self.agent.list_active_sessions(user_id=user_id)
