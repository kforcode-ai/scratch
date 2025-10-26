#!/usr/bin/env python3
"""
Demo: Redis-backed session management for MiniAgent
Shows persistent conversations across restarts
"""
import asyncio
import os
import sys
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent_framework.core import AgentConfig
from miniagent_framework.core.llm import RetryPolicy, RetryStrategy
from miniagent_framework.extensions.session import SessionAgent, SessionManager
from miniagent_framework.extensions.redis_client import RedisConfig
from miniagent_framework.core.tools import ToolRegistry, Tool, ToolResult
from miniagent_framework.core.events import StreamCallback, Event, EventType

# Load environment variables
load_dotenv()


# ============== Custom Tools ==============

class CalculateTool(Tool):
    """Mathematical calculation tool"""
    name = "calculate"
    description = "Perform mathematical calculations"
    
    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    
    async def execute(self, expression: str) -> ToolResult:
        try:
            # Safe evaluation of math expressions
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return ToolResult(
                success=True,
                data={"expression": expression, "result": result},
                display_content=f"🧮 Calculation: {expression} = {result}",
                llm_content=f"The result of {expression} is {result}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                display_content=f"❌ Calculation error: {str(e)}"
            )


class GetDateTimeTool(Tool):
    """Get current date and time"""
    name = "get_datetime"
    description = "Get the current date and time"
    
    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        now = datetime.now()
        formatted = now.strftime("%I:%M %p on %A, %B %d, %Y")
        
        return ToolResult(
            success=True,
            data={"datetime": now.isoformat()},
            display_content=f"📅 Current time: {formatted}",
            llm_content=f"The current date and time is {formatted}"
        )


# ============== Demo Functions ==============

def create_callbacks() -> StreamCallback:
    """Create callbacks for event handling"""
    callbacks = StreamCallback()
    
    # Session events
    def on_session_created(event: Event):
        print(f"📝 New session created: {event.data.get('session_id')[:8]}...")
    
    def on_session_resumed(event: Event):
        print(f"📂 Session resumed: {event.data.get('session_id')[:8]}...")
    
    def on_session_ended(event: Event):
        print(f"👋 Session ended: {event.data.get('session_id')[:8]}...")
    
    # Tool events
    def on_tool_call(event: Event):
        tool_name = event.data.get("tool", "unknown")
        args = event.data.get("arguments", {})
        print(f"🔧 Using tool: {tool_name} {args}")
    
    def on_tool_result(event: Event):
        print(f"   ✓ Tool completed")
    
    # Streaming events
    def on_stream_chunk(event: Event):
        print(event.data, end="", flush=True)
    
    def on_stream_end(event: Event):
        print()  # New line after streaming
    
    # Cache events
    def on_cache_hit(event: Event):
        print(f"⚡ Cache hit: {event.data.get('key', 'unknown')}")
    
    def on_cache_miss(event: Event):
        print(f"💭 Cache miss: {event.data.get('key', 'unknown')}")
    
    # Register handlers
    callbacks.on(EventType.SESSION_CREATED, on_session_created)
    callbacks.on(EventType.SESSION_RESUMED, on_session_resumed)
    callbacks.on(EventType.SESSION_ENDED, on_session_ended)
    callbacks.on(EventType.TOOL_CALL, on_tool_call)
    callbacks.on(EventType.TOOL_RESULT, on_tool_result)
    callbacks.on(EventType.STREAM_CHUNK, on_stream_chunk)
    callbacks.on(EventType.STREAM_END, on_stream_end)
    callbacks.on(EventType.CACHE_HIT, on_cache_hit)
    callbacks.on(EventType.CACHE_MISS, on_cache_miss)
    
    return callbacks


async def interactive_demo():
    """Interactive demo with Redis session management"""
    
    print("=" * 60)
    print("🚀 MiniAgent with Redis Session Management")
    print("=" * 60)
    
    # Check Redis availability
    redis_config = RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD"),
        session_ttl=3600  # 1 hour sessions
    )
    
    # Try to connect to Redis
    try:
        from miniagent.redis_client import RedisSessionManager
        test_redis = RedisSessionManager(redis_config)
        await test_redis.connect()
        if not await test_redis.health_check():
            print("⚠️  Redis not available. Sessions will not persist.")
            print("   To enable persistence, start Redis: redis-server")
            print()
            await test_redis.disconnect()
            redis_config = None
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("   Sessions will not persist across restarts.")
        print()
        redis_config = None
    
    # Setup tools
    registry = ToolRegistry()
    registry.register(CalculateTool())
    registry.register(GetDateTimeTool())
    
    # Configure agent
    config = AgentConfig(
        name="RedisBot",
        system_prompt="""You are a helpful AI assistant with session memory.
You can remember our conversation history even across restarts.

Available tools:
- calculate: Perform mathematical calculations
- get_datetime: Get current date and time

Always use the appropriate tools to provide accurate responses.""",
        model="gpt-4o-mini",
        temperature=0.7,
        retry_policy=RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            max_retries=3,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        ),
        stream_by_default=True
    )
    
    # Create session agent
    callbacks = create_callbacks()
    agent = SessionAgent(config, registry, callbacks, redis_config)
    
    if redis_config:
        await agent.initialize()
    
    # Create session manager
    manager = SessionManager(agent)
    
    # Session management
    session_id: Optional[str] = None
    user_id = "demo_user"
    
    print("Commands:")
    print("  • 'new' - Start a new session")
    print("  • 'list' - List all active sessions")
    print("  • 'resume <id>' - Resume a specific session")
    print("  • 'history' - Show conversation history")
    print("  • 'clear' - Clear current session history")
    print("  • 'quit' or 'exit' - End session and exit")
    print("-" * 60)
    
    # Check for existing sessions
    if redis_config:
        sessions = await manager.list_sessions(user_id)
        if sessions:
            print(f"\n📚 Found {len(sessions)} existing session(s):")
            for sess in sessions[:3]:  # Show max 3
                created = sess.get('created_at', 'unknown')
                msg_count = sess.get('message_count', 0)
                print(f"   • {sess['session_id'][:8]}... ({msg_count} messages, created: {created[:19]})")
            
            print("\nType 'resume <id>' to continue a session or press Enter for new session")
            choice = input("Choice: ").strip()
            
            if choice.startswith("resume ") and len(choice) > 7:
                resume_id = choice[7:].strip()
                # Find full session ID from partial
                for sess in sessions:
                    if sess['session_id'].startswith(resume_id):
                        session_id = sess['session_id']
                        break
    
    # Create or resume session
    if not session_id:
        session_id = await manager.create_or_resume_session(user_id=user_id)
        print(f"\n✨ Started new session: {session_id[:8]}...")
    else:
        await manager.create_or_resume_session(session_id=session_id, user_id=user_id)
        print(f"\n✨ Resumed session: {session_id[:8]}...")
        
        # Show recent history
        history = await manager.get_history(session_id)
        if history:
            print("\n📜 Recent conversation:")
            for msg in history[-4:]:  # Show last 2 exchanges
                role = "You" if msg['role'] == 'user' else "Bot"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                if msg['role'] in ['user', 'assistant']:
                    print(f"   {role}: {content}")
    
    print("\n" + "-" * 60)
    print("💬 Start chatting! (type 'help' for commands)")
    print("-" * 60)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                await manager.end_session(session_id)
                print("\n👋 Session saved. Goodbye!")
                break
            
            elif user_input.lower() == 'new':
                # Save current session and start new
                await manager.end_session(session_id)
                session_id = await manager.create_or_resume_session(user_id=user_id)
                print(f"✨ Started new session: {session_id[:8]}...")
                continue
            
            elif user_input.lower() == 'list':
                sessions = await manager.list_sessions(user_id)
                if sessions:
                    print(f"\n📚 Active sessions ({len(sessions)}):")
                    for sess in sessions:
                        created = sess.get('created_at', 'unknown')[:19]
                        msg_count = sess.get('message_count', 0)
                        current = " (current)" if sess['session_id'] == session_id else ""
                        print(f"   • {sess['session_id'][:8]}... - {msg_count} msgs, created: {created}{current}")
                else:
                    print("No active sessions found.")
                continue
            
            elif user_input.lower().startswith('resume '):
                resume_id = user_input[7:].strip()
                sessions = await manager.list_sessions(user_id)
                
                # Find full session ID
                found = None
                for sess in sessions:
                    if sess['session_id'].startswith(resume_id):
                        found = sess['session_id']
                        break
                
                if found:
                    await manager.end_session(session_id)
                    session_id = found
                    await manager.create_or_resume_session(session_id=session_id)
                    print(f"✨ Resumed session: {session_id[:8]}...")
                else:
                    print(f"Session '{resume_id}' not found.")
                continue
            
            elif user_input.lower() == 'history':
                history = await manager.get_history(session_id)
                if history:
                    print("\n📜 Conversation history:")
                    for msg in history:
                        role = "You" if msg['role'] == 'user' else "Bot"
                        if msg['role'] in ['user', 'assistant']:
                            print(f"\n{role}: {msg['content']}")
                else:
                    print("No history in current session.")
                continue
            
            elif user_input.lower() == 'clear':
                await manager.clear_history(session_id)
                print("✨ Session history cleared.")
                continue
            
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  • 'new' - Start a new session")
                print("  • 'list' - List all active sessions")
                print("  • 'resume <id>' - Resume a specific session")
                print("  • 'history' - Show conversation history")
                print("  • 'clear' - Clear current session history")
                print("  • 'quit' - End session and exit")
                continue
            
            # Process with agent
            print("🤖 RedisBot: ", end="", flush=True)
            response = await manager.chat(session_id, user_input, stream=True)
            
            # Response is already printed via streaming
            if not config.stream_by_default:
                print(response)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Saving session...")
            await manager.end_session(session_id)
            print("Session saved. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # Cleanup
    if redis_config:
        await agent.shutdown()


async def main():
    """Main entry point"""
    await interactive_demo()


if __name__ == "__main__":
    print("\n🎯 Redis Session Management Demo")
    print("This demo shows persistent conversations that survive restarts.\n")
    
    asyncio.run(main())

