#!/usr/bin/env python3
"""
Example demonstrating MiniAgent with enterprise features:
- Event sourcing and checkpointing
- LangFuse observability
- Redis persistence
- Circuit breaker protection
- Error recovery strategies
"""

import asyncio
import os
from datetime import datetime
from typing import Optional

# Set up environment variables (you can also use .env file)
# os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["LANGFUSE_SECRET_KEY"] = "your-langfuse-secret-key"
# os.environ["LANGFUSE_PUBLIC_KEY"] = "your-langfuse-public-key"
# os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

from miniagent_framework.core import (
    Agent,
    AgentConfig,
    Thread,
    ToolRegistry,
    StreamCallback,
    EventType
)
from miniagent_framework.extensions.redis_client import RedisSessionManager, RedisConfig
from miniagent_framework.core.llm import exponential_retry


# ============== Setup Tools ==============

registry = ToolRegistry()

@registry.tool(description="Calculate mathematical expressions")
async def calculate(expression: str) -> str:
    """Calculate a math expression safely"""
    try:
        # For demo, using eval with restricted namespace
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {e}"


@registry.tool(description="Get current time and date")
async def get_time() -> str:
    """Get the current time"""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@registry.tool(description="Simulate a task that might fail")
async def unstable_task(should_fail: bool = False) -> str:
    """Simulate an unstable service for testing circuit breaker"""
    if should_fail:
        raise Exception("Service temporarily unavailable")
    return "Task completed successfully"


# ============== Setup Callbacks for Monitoring ==============

callbacks = StreamCallback()

# Monitor events
callbacks.on(EventType.LLM_CALL, lambda e: print(f"🤖 LLM Call: {e.content}"))
callbacks.on(EventType.TOOL_EXECUTION, lambda e: print(f"🔧 Tool: {e.content}"))
callbacks.on(EventType.ERROR, lambda e: print(f"❌ Error: {e.content}"))
callbacks.on(EventType.STREAM_CHUNK, lambda e: print(e.content, end="", flush=True))


# ============== Main Example ==============

async def main():
    print("=" * 60)
    print("MiniAgent Enterprise Features Demo")
    print("=" * 60)
    
    # ========== 1. Initialize Components ==========
    
    print("\n1. Initializing components...")
    
    # Redis manager for persistence (optional)
    redis_manager: Optional[RedisSessionManager] = None
    try:
        redis_config = RedisConfig(
            host="localhost",
            port=6379,
            session_ttl=3600  # 1 hour
        )
        redis_manager = RedisSessionManager(redis_config)
        await redis_manager.connect()
        print("✅ Redis connected for persistence")
    except Exception as e:
        print(f"⚠️ Redis not available, continuing without persistence: {e}")
    
    # Agent with enterprise features
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            system_prompt="""You are a helpful assistant with calculation and time tools.
            You can perform math calculations, get the current time, and handle various tasks.
            Be concise in your responses.""",
            retry_policy=exponential_retry(max_retries=3),
            stream_by_default=True
        ),
        tools=registry,
        callbacks=callbacks,
        enable_langfuse=True  # Automatic observability if env vars are set
    )
    
    print(f"✅ Agent initialized (LangFuse: {'enabled' if agent.langfuse else 'disabled'})")
    
    # ========== 2. Load or Create Thread ==========
    
    print("\n2. Managing conversation thread...")
    
    thread_id = "demo_session_001"
    thread: Optional[Thread] = None
    
    # Try to load existing thread from Redis
    if redis_manager:
        thread = await redis_manager.load_thread_enhanced(thread_id)
        if thread:
            print(f"✅ Loaded existing thread with {len(thread.messages)} messages")
            print(f"   Events: {len(thread.events)}, Checkpoints: {len(thread._checkpoints)}")
            
            # Show last checkpoint
            if thread._checkpoints:
                last_checkpoint = thread._checkpoints[-1]
                print(f"   Last checkpoint: {last_checkpoint['timestamp']}")
    
    # Create new thread if not found
    if not thread:
        thread = Thread(thread_id=thread_id)
        print(f"✅ Created new thread: {thread_id}")
    
    # ========== 3. Run Agent with Various Scenarios ==========
    
    print("\n3. Running agent interactions...\n")
    
    # Test 1: Basic conversation with tool use
    print("-" * 40)
    print("Test 1: Basic math and time")
    print("-" * 40)
    
    response = await agent.run(
        "What's 42 * 17? Also, what time is it?",
        thread=thread,
        stream=True
    )
    print(f"\n\n📝 Response: {response}")
    
    # Show thread state
    print(f"\n📊 Thread state: {len(thread.messages)} messages, {len(thread.events)} events")
    last_checkpoint = thread.create_checkpoint()
    print(f"📌 Checkpoint created: {last_checkpoint['state_hash'][:8]}...")
    
    # Save thread to Redis
    if redis_manager:
        await redis_manager.save_thread_enhanced(thread)
        print("💾 Thread saved to Redis")
    
    # Test 2: Error recovery
    print("\n" + "-" * 40)
    print("Test 2: Error recovery and circuit breaker")
    print("-" * 40)
    
    # This might trigger circuit breaker if service fails multiple times
    response = await agent.run(
        "Run the unstable task (don't make it fail)",
        thread=thread,
        stream=False  # Don't stream for this test
    )
    print(f"📝 Response: {response}")
    
    # Check circuit breaker status if available
    if agent.llm.circuit_breaker:
        status = agent.llm.circuit_breaker.get_status()
        print(f"🔌 Circuit breaker: {status['state']} (failures: {status['failure_count']}/{status['threshold']})")
    
    # Test 3: Thread events analysis
    print("\n" + "-" * 40)
    print("Test 3: Analyzing thread events")
    print("-" * 40)
    
    # Count event types
    event_counts = {}
    for event in thread.events:
        event_type = event.type.value
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print("📊 Event summary:")
    for event_type, count in event_counts.items():
        print(f"   {event_type}: {count}")
    
    # Get events from Redis for analysis
    if redis_manager:
        recent_events = await redis_manager.get_thread_events(thread_id, limit=10)
        print(f"\n📜 Last {len(recent_events)} events from Redis:")
        for event in recent_events[-3:]:  # Show last 3
            print(f"   - {event['type']}: {str(event['data'])[:50]}...")
    
    # Test 4: Context too long - test auto-recovery
    print("\n" + "-" * 40)
    print("Test 4: Long conversation handling")
    print("-" * 40)
    
    # Add many messages to test context management
    for i in range(3):
        response = await agent.run(
            f"Tell me fact #{i+1} about Python programming",
            thread=thread,
            stream=False
        )
    
    print(f"📊 Thread now has {len(thread.messages)} messages")
    
    # ========== 4. Observability Summary ==========
    
    print("\n" + "=" * 60)
    print("4. Observability Summary")
    print("=" * 60)
    
    if agent.langfuse:
        print("""
✅ LangFuse Observability Active!
   
   View your traces at: https://cloud.langfuse.com
   
   Automatic tracking includes:
   - All LLM calls with token usage
   - Tool executions with parameters
   - Errors and recovery attempts
   - Latency metrics
   - Full conversation traces
        """)
    else:
        print("""
ℹ️ LangFuse not active. To enable:
   1. pip install langfuse
   2. Set environment variables:
      - LANGFUSE_SECRET_KEY
      - LANGFUSE_PUBLIC_KEY
      - LANGFUSE_HOST
""")
    
    # ========== 5. Persistence Summary ==========
    
    print("\n" + "=" * 60)
    print("5. Persistence & Recovery")
    print("=" * 60)
    
    if redis_manager:
        # List all threads
        all_threads = await redis_manager.list_threads()
        print(f"📚 Total threads in Redis: {len(all_threads)}")
        
        # Show thread metadata
        for thread_info in all_threads[:3]:  # Show first 3
            print(f"   - {thread_info['thread_id']} (TTL: {thread_info['ttl']}s)")
        
        # Demonstrate thread recovery
        print(f"\n🔄 Demonstrating thread recovery...")
        
        # Save current thread
        await redis_manager.save_thread_enhanced(thread)
        
        # Load it back
        recovered_thread = await redis_manager.load_thread_enhanced(thread_id)
        if recovered_thread:
            print(f"✅ Thread recovered successfully!")
            print(f"   Messages: {len(recovered_thread.messages)}")
            print(f"   Events: {len(recovered_thread.events)}")
            print(f"   Checkpoints: {len(recovered_thread._checkpoints)}")
    
    # ========== 6. Enterprise Features Summary ==========
    
    print("\n" + "=" * 60)
    print("6. Enterprise Features Active")
    print("=" * 60)
    
    features = [
        ("✅", "Event Sourcing", f"{len(thread.events)} events logged"),
        ("✅", "Checkpointing", f"{len(thread._checkpoints)} checkpoints"),
        ("✅" if agent.langfuse else "❌", "LangFuse Observability", "Automatic tracing"),
        ("✅" if redis_manager else "❌", "Redis Persistence", "Thread recovery"),
        ("✅", "Circuit Breaker", "Fault tolerance"),
        ("✅", "Error Recovery", "Auto-retry with backoff"),
        ("✅", "Thread Compression", "Efficient storage"),
    ]
    
    for status, feature, detail in features:
        print(f"{status} {feature}: {detail}")
    
    # Cleanup
    if redis_manager:
        await redis_manager.disconnect()
        print("\n👋 Redis disconnected")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          MiniAgent Enterprise Features Demo              ║
║                                                          ║
║  This demo showcases:                                   ║
║  • Event sourcing and audit trails                      ║
║  • LangFuse observability (if configured)               ║
║  • Redis persistence and recovery                       ║
║  • Circuit breaker protection                           ║
║  • Automatic error recovery                             ║
║                                                         ║
║  Requirements:                                          ║
║  • pip install openai langfuse redis                    ║
║  • Redis server running (optional)                      ║
║  • Environment variables set (see code)                 ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
