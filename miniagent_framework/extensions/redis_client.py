"""
Redis client for session management and caching
"""
import json
import asyncio
import redis.asyncio as redis
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)

@dataclass
class RedisConfig:
    """Configuration for Redis connection"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = False  # Keep False to handle both text and binary
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Session-specific settings
    session_ttl: int = 3600  # 1 hour default
    max_session_size: int = 1024 * 1024  # 1MB max per session
    
    # Namespace prefixes
    session_prefix: str = "session:"
    thread_prefix: str = "thread:"
    cache_prefix: str = "cache:"
    lock_prefix: str = "lock:"


class RedisSessionManager:
    """
    Manages sessions and conversation threads in Redis
    Provides distributed session management for the agent framework
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._lock_timeout = 10  # seconds
        
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=self.config.decode_responses,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            # Test connection
            await self._client.ping()
            logger.info(f"Redis connected successfully to {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            if not self._client:
                return False
            await self._client.ping()
            return True
        except:
            return False
    
    # ============== Session Management ==============
    
    async def create_session(self, session_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Create a new session
        
        Args:
            session_id: Unique session identifier
            data: Session data to store
            ttl: Time to live in seconds (uses config default if not specified)
        """
        key = f"{self.config.session_prefix}{session_id}"
        ttl = ttl or self.config.session_ttl
        
        try:
            # Serialize data
            serialized = json.dumps(data)
            
            # Check size limit
            if len(serialized.encode()) > self.config.max_session_size:
                logger.warning(f"Session {session_id} exceeds size limit")
                return False
            
            # Store with TTL
            await self._client.setex(key, ttl, serialized)
            
            # Store metadata
            await self._store_session_metadata(session_id, data)
            
            logger.debug(f"Created session {session_id} with TTL {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data"""
        key = f"{self.config.session_prefix}{session_id}"
        
        try:
            data = await self._client.get(key)
            if data:
                # Refresh TTL on access
                await self._client.expire(key, self.config.session_ttl)
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update existing session"""
        key = f"{self.config.session_prefix}{session_id}"
        
        try:
            # Check if session exists
            if not await self._client.exists(key):
                logger.warning(f"Session {session_id} does not exist")
                return False
            
            # Update data
            serialized = json.dumps(data)
            
            # Check size limit
            if len(serialized.encode()) > self.config.max_session_size:
                logger.warning(f"Session {session_id} update exceeds size limit")
                return False
            
            # Get current TTL and preserve it
            ttl = await self._client.ttl(key)
            if ttl > 0:
                await self._client.setex(key, ttl, serialized)
            else:
                await self._client.set(key, serialized)
            
            # Update metadata
            await self._store_session_metadata(session_id, data)
            
            logger.debug(f"Updated session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        key = f"{self.config.session_prefix}{session_id}"
        
        try:
            # Delete session data
            result = await self._client.delete(key)
            
            # Delete associated thread
            thread_key = f"{self.config.thread_prefix}{session_id}"
            await self._client.delete(thread_key)
            
            # Delete metadata
            await self._delete_session_metadata(session_id)
            
            logger.debug(f"Deleted session {session_id}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def extend_session_ttl(self, session_id: str, additional_seconds: int) -> bool:
        """Extend session TTL"""
        key = f"{self.config.session_prefix}{session_id}"
        
        try:
            current_ttl = await self._client.ttl(key)
            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                await self._client.expire(key, new_ttl)
                logger.debug(f"Extended session {session_id} TTL by {additional_seconds}s")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to extend session {session_id} TTL: {e}")
            return False
    
    async def list_sessions(self, pattern: str = "*") -> List[str]:
        """List all active sessions matching pattern"""
        try:
            cursor = 0
            sessions = []
            match_pattern = f"{self.config.session_prefix}{pattern}"
            
            while True:
                cursor, keys = await self._client.scan(
                    cursor, 
                    match=match_pattern,
                    count=100
                )
                
                # Extract session IDs from keys
                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    session_id = key.replace(self.config.session_prefix, "")
                    sessions.append(session_id)
                
                if cursor == 0:
                    break
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    # ============== Thread Management ==============
    
    async def save_thread(self, session_id: str, thread_data: Any) -> bool:
        """
        Save conversation thread data
        Uses pickle for complex Python objects
        """
        key = f"{self.config.thread_prefix}{session_id}"
        
        try:
            # Serialize thread data with pickle for complex objects
            serialized = pickle.dumps(thread_data)
            
            # Store with same TTL as session
            session_ttl = await self._client.ttl(f"{self.config.session_prefix}{session_id}")
            if session_ttl > 0:
                await self._client.setex(key, session_ttl, serialized)
            else:
                await self._client.setex(key, self.config.session_ttl, serialized)
            
            logger.debug(f"Saved thread for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save thread for session {session_id}: {e}")
            return False
    
    async def load_thread(self, session_id: str) -> Optional[Any]:
        """Load conversation thread data"""
        key = f"{self.config.thread_prefix}{session_id}"
        
        try:
            data = await self._client.get(key)
            if data:
                # Refresh TTL on access
                await self._client.expire(key, self.config.session_ttl)
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to load thread for session {session_id}: {e}")
            return None
    
    async def append_to_thread(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Append a message to existing thread"""
        try:
            # Load existing thread
            thread_data = await self.load_thread(session_id)
            if thread_data is None:
                thread_data = {"messages": [], "events": []}
            
            # Append message
            if "messages" not in thread_data:
                thread_data["messages"] = []
            thread_data["messages"].append(message)
            
            # Save updated thread
            return await self.save_thread(session_id, thread_data)
            
        except Exception as e:
            logger.error(f"Failed to append to thread for session {session_id}: {e}")
            return False
    
    # ============== Caching ==============
    
    async def cache_set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set cache value"""
        cache_key = f"{self.config.cache_prefix}{key}"
        
        try:
            serialized = json.dumps(value) if not isinstance(value, (str, bytes)) else value
            await self._client.setex(cache_key, ttl, serialized)
            logger.debug(f"Cached {key} with TTL {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_key = f"{self.config.cache_prefix}{key}"
        
        try:
            data = await self._client.get(cache_key)
            if data:
                try:
                    return json.loads(data)
                except:
                    return data
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cache {key}: {e}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cached value"""
        cache_key = f"{self.config.cache_prefix}{key}"
        
        try:
            result = await self._client.delete(cache_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache {key}: {e}")
            return False
    
    # ============== Distributed Locking ==============
    
    async def acquire_lock(self, resource: str, timeout: Optional[int] = None) -> bool:
        """
        Acquire a distributed lock
        
        Args:
            resource: Resource identifier to lock
            timeout: Lock timeout in seconds
        """
        lock_key = f"{self.config.lock_prefix}{resource}"
        timeout = timeout or self._lock_timeout
        
        try:
            # Try to acquire lock with NX (only if not exists) and EX (expiry)
            result = await self._client.set(
                lock_key, 
                "1", 
                nx=True, 
                ex=timeout
            )
            if result:
                logger.debug(f"Acquired lock for {resource}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to acquire lock for {resource}: {e}")
            return False
    
    async def release_lock(self, resource: str) -> bool:
        """Release a distributed lock"""
        lock_key = f"{self.config.lock_prefix}{resource}"
        
        try:
            result = await self._client.delete(lock_key)
            if result:
                logger.debug(f"Released lock for {resource}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to release lock for {resource}: {e}")
            return False
    
    async def is_locked(self, resource: str) -> bool:
        """Check if resource is locked"""
        lock_key = f"{self.config.lock_prefix}{resource}"
        
        try:
            return bool(await self._client.exists(lock_key))
            
        except Exception as e:
            logger.error(f"Failed to check lock for {resource}: {e}")
            return False
    
    # ============== Private Helper Methods ==============
    
    async def _store_session_metadata(self, session_id: str, data: Dict[str, Any]):
        """Store session metadata for monitoring"""
        metadata_key = f"metadata:{session_id}"
        metadata = {
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "user_id": data.get("user_id", "anonymous"),
            "agent_name": data.get("agent_name", "default")
        }
        await self._client.hset(metadata_key, mapping=metadata)
        await self._client.expire(metadata_key, self.config.session_ttl)
    
    async def _delete_session_metadata(self, session_id: str):
        """Delete session metadata"""
        metadata_key = f"metadata:{session_id}"
        await self._client.delete(metadata_key)
    
    # ============== Pub/Sub for Real-time Updates ==============
    
    async def subscribe(self, channels: List[str]):
        """Subscribe to Redis channels for real-time updates"""
        if not self._pubsub:
            self._pubsub = self._client.pubsub()
        
        await self._pubsub.subscribe(*channels)
        logger.info(f"Subscribed to channels: {channels}")
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """Publish message to a channel"""
        try:
            serialized = json.dumps(message)
            num_subscribers = await self._client.publish(channel, serialized)
            logger.debug(f"Published to {channel}, reached {num_subscribers} subscribers")
            return num_subscribers
            
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            return 0
    
    async def get_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get message from subscribed channels"""
        if not self._pubsub:
            return None
        
        try:
            message = await self._pubsub.get_message(timeout=timeout)
            if message and message['type'] == 'message':
                return json.loads(message['data'])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get message: {e}")
            return None


# Singleton instance for easy access
_redis_manager: Optional[RedisSessionManager] = None

async def get_redis_manager(config: Optional[RedisConfig] = None) -> RedisSessionManager:
    """Get or create Redis manager singleton"""
    global _redis_manager
    
    if _redis_manager is None:
        _redis_manager = RedisSessionManager(config)
        await _redis_manager.connect()
    
    return _redis_manager

async def cleanup_redis():
    """Cleanup Redis connections"""
    global _redis_manager
    
    if _redis_manager:
        await _redis_manager.disconnect()
        _redis_manager = None
