# Redis Session Management for MiniAgent

This enhancement adds production-grade session management to the MiniAgent framework using Redis as the backend store.

## 🚀 Features

### Core Capabilities
- **Persistent Sessions**: Conversations survive application restarts
- **Distributed State**: Multiple agent instances can share sessions
- **Auto-expiry**: Sessions automatically expire after configurable TTL
- **Session History**: Full conversation history with message replay
- **User Management**: Sessions can be associated with user IDs
- **Real-time Updates**: Pub/Sub support for multi-instance coordination

### Technical Features
- **Thread Persistence**: Complete conversation threads saved to Redis
- **Atomic Operations**: Thread-safe session updates
- **Distributed Locking**: Prevents race conditions in multi-instance setups
- **Caching Layer**: Built-in caching for tool results and LLM responses
- **Health Monitoring**: Automatic health checks and reconnection
- **Memory Management**: Automatic cleanup of expired sessions

## 📦 Installation

### 1. Install Redis Dependencies
```bash
pip install redis hiredis
```

### 2. Start Redis Server

#### Option A: Using Docker (Recommended)
```bash
# Start Redis with docker-compose
docker-compose up -d

# Verify Redis is running
docker-compose ps

# View Redis data (optional)
# Open http://localhost:8081 for Redis Commander UI
```

#### Option B: Local Installation
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Verify connection
redis-cli ping
# Should return: PONG
```

## 🎮 Usage

### Basic Usage

```python
from miniagent import AgentConfig
from miniagent.session import SessionAgent, SessionManager
from miniagent.redis_client import RedisConfig

# Configure Redis
redis_config = RedisConfig(
    host="localhost",
    port=6379,
    session_ttl=3600,  # 1 hour
    max_session_size=1024*1024  # 1MB
)

# Create session-aware agent
agent = SessionAgent(
    config=AgentConfig(name="MyBot"),
    redis_config=redis_config
)

# Initialize Redis connection
await agent.initialize()

# Create session manager
manager = SessionManager(agent)

# Start a new session
session_id = await manager.create_or_resume_session(user_id="user123")

# Chat with persistent context
response = await manager.chat(session_id, "Hello!")
response = await manager.chat(session_id, "What did I just say?")  # Remembers context

# Session persists across restarts!
```

### Interactive Demo

Run the included demo to see Redis sessions in action:

```bash
# Make sure Redis is running
docker-compose up -d

# Run the demo
python demo_redis_session.py
```

Features demonstrated:
- Create new sessions
- Resume existing sessions
- List all active sessions
- View conversation history
- Clear session history
- Sessions persist across application restarts

## 🏗️ Architecture

### Session Lifecycle

```
User Request → Session Manager → Redis Check → Load/Create Thread → Agent Processing → Save to Redis
```

### Data Structure in Redis

```
session:<id>          # Session metadata (JSON)
thread:<id>           # Conversation thread (Pickle)
cache:<key>           # Cached responses (JSON)
lock:<resource>       # Distributed locks
metadata:<id>         # Session analytics
```

### Key Components

1. **RedisSessionManager**: Low-level Redis operations
   - Connection pooling
   - Serialization/deserialization
   - TTL management
   - Distributed locking

2. **SessionThread**: Enhanced Thread with persistence
   - Auto-save on changes
   - Lazy loading from Redis
   - Message and event storage

3. **SessionAgent**: Session-aware agent
   - Manages multiple concurrent sessions
   - Automatic session cleanup
   - Background persistence

4. **SessionManager**: High-level interface
   - Simple chat API
   - Session lifecycle management
   - History management

## 🔧 Configuration

### Redis Configuration

```python
RedisConfig(
    host="localhost",           # Redis server host
    port=6379,                 # Redis server port
    db=0,                      # Database number
    password=None,             # Optional password
    max_connections=50,        # Connection pool size
    session_ttl=3600,          # Session TTL in seconds
    max_session_size=1048576,  # Max session size in bytes
    health_check_interval=30   # Health check interval
)
```

### Environment Variables

```bash
# .env file
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password  # Optional
```

## 📊 Monitoring

### Session Metrics

```python
# List all active sessions
sessions = await manager.list_sessions()

# Get session details
for session in sessions:
    print(f"Session: {session['session_id']}")
    print(f"User: {session['user_id']}")
    print(f"Messages: {session['message_count']}")
    print(f"Created: {session['created_at']}")
```

### Redis Monitoring

```bash
# Check Redis memory usage
redis-cli INFO memory

# Monitor commands in real-time
redis-cli MONITOR

# View all keys
redis-cli KEYS "session:*"

# Get session TTL
redis-cli TTL "session:<id>"
```

### Redis Commander UI

Access the web UI at http://localhost:8081 to:
- Browse all sessions
- View conversation threads
- Monitor performance
- Manage TTLs

## 🚨 Production Considerations

### High Availability

1. **Redis Cluster**: Use Redis Cluster for horizontal scaling
2. **Sentinel**: Use Redis Sentinel for automatic failover
3. **Persistence**: Enable AOF (Append Only File) for durability
4. **Backups**: Regular snapshots of Redis data

### Performance Tuning

```python
# Optimize for high throughput
redis_config = RedisConfig(
    max_connections=100,        # Increase connection pool
    socket_timeout=2,          # Faster timeout
    retry_on_timeout=True,     # Auto-retry on timeout
    session_ttl=1800,          # Shorter TTL for memory
)

# Use pipelining for batch operations
async with redis_manager._client.pipeline() as pipe:
    pipe.set("key1", "value1")
    pipe.set("key2", "value2")
    await pipe.execute()
```

### Security

1. **Authentication**: Always use password in production
2. **Encryption**: Use TLS/SSL for Redis connections
3. **Network**: Bind Redis to private network interfaces
4. **ACLs**: Use Redis ACLs to limit command access

### Memory Management

```bash
# Redis configuration for production
maxmemory 2gb
maxmemory-policy allkeys-lru  # Evict least recently used keys
```

## 🧪 Testing

```python
# Test Redis connection
async def test_redis():
    from miniagent.redis_client import get_redis_manager
    
    manager = await get_redis_manager()
    
    # Health check
    assert await manager.health_check()
    
    # Test session operations
    session_id = "test_session"
    data = {"user": "test", "timestamp": "2024-01-01"}
    
    # Create
    assert await manager.create_session(session_id, data)
    
    # Read
    retrieved = await manager.get_session(session_id)
    assert retrieved["user"] == "test"
    
    # Update
    data["status"] = "active"
    assert await manager.update_session(session_id, data)
    
    # Delete
    assert await manager.delete_session(session_id)
    
    print("✅ All Redis tests passed!")

# Run test
asyncio.run(test_redis())
```

## 🎯 Use Cases

1. **Customer Support Bot**: Maintain context across multiple support sessions
2. **Educational Assistant**: Remember student progress and learning history
3. **Personal Assistant**: Keep user preferences and conversation history
4. **Multi-turn Workflows**: Complex tasks spanning multiple interactions
5. **Team Collaboration**: Shared sessions for collaborative AI interactions

## 🔍 Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if Redis is running
   redis-cli ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

2. **Session Not Found**
   ```python
   # Session may have expired, check TTL
   ttl = await redis_client.ttl(f"session:{session_id}")
   print(f"TTL remaining: {ttl} seconds")
   ```

3. **Memory Issues**
   ```bash
   # Check Redis memory
   redis-cli --bigkeys
   redis-cli MEMORY DOCTOR
   ```

4. **Performance Issues**
   ```bash
   # Monitor slow queries
   redis-cli SLOWLOG GET 10
   
   # Check connection pool
   redis-cli CLIENT LIST
   ```

## 📚 Advanced Features

### Custom Serialization

```python
# Use MessagePack for better performance
import msgpack

class CustomRedisManager(RedisSessionManager):
    def serialize(self, data):
        return msgpack.packb(data)
    
    def deserialize(self, data):
        return msgpack.unpackb(data)
```

### Event Streaming

```python
# Subscribe to session events
await redis_manager.subscribe(["session_updates"])

# Listen for updates
async for message in redis_manager.get_messages():
    print(f"Session update: {message}")
```

### Distributed Rate Limiting

```python
async def rate_limit(user_id: str, max_requests: int = 10):
    key = f"rate_limit:{user_id}"
    
    count = await redis_manager._client.incr(key)
    if count == 1:
        await redis_manager._client.expire(key, 60)  # 1 minute window
    
    return count <= max_requests
```

## 🚀 Next Steps

With Redis session management implemented, consider:

1. **Add Vector Search**: Use Redis Vector Similarity for semantic memory
2. **Implement Caching**: Cache expensive LLM/tool responses
3. **Add Analytics**: Track usage patterns and performance metrics
4. **Multi-tenancy**: Separate Redis databases per tenant
5. **Geo-distribution**: Redis geo-replication for global apps

## 📝 License

MIT License - See LICENSE file for details

