#!/usr/bin/env python3
"""
Demo: Using different LLM providers (OpenAI, Gemini, Anthropic)
Shows how to switch between providers and compare responses
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent_framework.core import Agent, AgentConfig, Thread
from miniagent_framework.core.tools import ToolRegistry
from miniagent_framework.core.events import StreamCallback, EventType
from miniagent_framework.core.llm import LLMProvider

# Load environment variables
load_dotenv()


# ============== Sample Tools ==============

def create_sample_tools():
    """Create some sample tools for testing"""
    registry = ToolRegistry()
    
    @registry.tool(description="Get current time")
    async def get_time():
        from datetime import datetime
        return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
    
    @registry.tool(description="Calculate simple math expressions")
    async def calculate(expression: str):
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"Result: {expression} = {result}"
        except:
            return f"Error: Invalid expression '{expression}'"
    
    return registry


# ============== Demo Functions ==============

async def test_provider(provider_name: str, api_key: str = None):
    """Test a specific LLM provider"""
    print(f"\n{'='*60}")
    print(f"🤖 Testing {provider_name.upper()} Provider")
    print(f"{'='*60}")
    
    # Configure agent for specific provider
    config = AgentConfig(
        name=f"{provider_name.title()} Assistant",
        provider=provider_name,
        api_key=api_key,  # Use provided key or fall back to env vars
        model=None,  # Use default model for each provider
        temperature=0.7,
        stream_by_default=True
    )
    
    # Print configuration
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model or 'default for provider'}")
    print(f"API Key: {'✓ Configured' if (api_key or os.getenv(f'{provider_name.upper()}_API_KEY')) else '✗ Missing'}")
    
    # Create agent
    try:
        tools = create_sample_tools()
        agent = Agent(config=config, tools=tools)
        
        # Set up streaming callback
        callbacks = StreamCallback()
        callbacks.on(EventType.STREAM_CHUNK, lambda e: print(e.content, end="", flush=True))
        callbacks.on(EventType.STREAM_END, lambda e: print("\n"))
        
        agent.callbacks = callbacks
        
        # Test queries
        test_queries = [
            "What time is it?",
            "Calculate 42 * 17",
            "Tell me a very short joke (one line)"
        ]
        
        thread = Thread()
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("💬 Response: ", end="")
            
            try:
                await agent.run(query, thread=thread, stream=True)
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Failed to initialize {provider_name}: {e}")
        print(f"   Make sure {provider_name.upper()}_API_KEY is set in your environment")
        return False
    
    return True


async def compare_providers():
    """Compare responses from different providers for the same prompt"""
    print("\n" + "="*60)
    print("🔄 COMPARING PROVIDERS")
    print("="*60)
    
    prompt = "Explain quantum computing in exactly one sentence."
    
    providers_to_test = []
    
    # Check which providers have API keys configured
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append("openai")
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers_to_test.append("gemini")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append("anthropic")
    
    if not providers_to_test:
        print("❌ No API keys found. Please set at least one of:")
        print("   - OPENAI_API_KEY")
        print("   - GOOGLE_API_KEY or GEMINI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        return
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"Testing providers: {', '.join(providers_to_test)}\n")
    
    for provider in providers_to_test:
        print(f"\n{provider.upper()}:")
        print("-" * 40)
        
        config = AgentConfig(
            provider=provider,
            temperature=0.7,
            stream_by_default=False  # Non-streaming for comparison
        )
        
        try:
            agent = Agent(config=config)
            response = await agent.run(prompt)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")


async def interactive_mode():
    """Interactive mode to test any provider"""
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE")
    print("="*60)
    
    # List available providers
    print("\nAvailable providers:")
    providers = ["openai", "gemini", "anthropic"]
    for i, p in enumerate(providers, 1):
        api_key_env = f"{p.upper()}_API_KEY"
        if p == "gemini":
            api_key_env = "GOOGLE_API_KEY or GEMINI_API_KEY"
        status = "✓" if (os.getenv(f"{p.upper()}_API_KEY") or 
                        (p == "gemini" and os.getenv("GOOGLE_API_KEY"))) else "✗"
        print(f"  {i}. {p} [{status}] (needs {api_key_env})")
    
    # Get user choice
    try:
        choice = input("\nSelect provider (1-3 or name): ").strip()
        if choice.isdigit():
            provider = providers[int(choice) - 1]
        else:
            provider = choice.lower()
        
        if provider not in providers:
            print("Invalid provider!")
            return
    except:
        print("Invalid choice!")
        return
    
    print(f"\nUsing {provider} provider")
    
    # Create agent
    config = AgentConfig(
        provider=provider,
        temperature=0.7,
        stream_by_default=True
    )
    
    agent = Agent(config=config, tools=create_sample_tools())
    
    # Set up streaming
    callbacks = StreamCallback()
    callbacks.on(EventType.STREAM_CHUNK, lambda e: print(e.content, end="", flush=True))
    callbacks.on(EventType.STREAM_END, lambda e: print())
    agent.callbacks = callbacks
    
    thread = Thread()
    
    print("\nChat started! Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            print(f"{provider.title()}: ", end="")
            await agent.run(user_input, thread=thread, stream=True)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\nGoodbye!")


async def main():
    """Main demo function"""
    print("🚀 Multi-LLM Provider Demo\n")
    
    print("Select demo mode:")
    print("1. Test individual providers")
    print("2. Compare providers")
    print("3. Interactive chat")
    print("4. Run all demos")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        # Test individual providers
        for provider in ["openai", "gemini", "anthropic"]:
            api_key_var = f"{provider.upper()}_API_KEY"
            if provider == "gemini":
                api_key_var = "GOOGLE_API_KEY"
            
            if os.getenv(api_key_var):
                await test_provider(provider)
            else:
                print(f"\n⚠️ Skipping {provider} (no {api_key_var} found)")
    
    elif choice == "2":
        await compare_providers()
    
    elif choice == "3":
        await interactive_mode()
    
    elif choice == "4":
        # Run all demos
        print("\n1️⃣ Testing Individual Providers")
        for provider in ["openai", "gemini", "anthropic"]:
            api_key_var = f"{provider.upper()}_API_KEY"
            if provider == "gemini":
                api_key_var = "GOOGLE_API_KEY"
            
            if os.getenv(api_key_var):
                await test_provider(provider)
        
        print("\n2️⃣ Comparing Providers")
        await compare_providers()
        
        print("\n3️⃣ For interactive mode, run the demo again and select option 3")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-LLM PROVIDER SUPPORT DEMO")
    print("Supports: OpenAI, Google Gemini, Anthropic Claude")
    print("="*60 + "\n")
    
    # Check for API keys
    has_keys = False
    print("API Key Status:")
    if os.getenv("OPENAI_API_KEY"):
        print("  ✓ OpenAI API key found")
        has_keys = True
    else:
        print("  ✗ OpenAI API key not found (set OPENAI_API_KEY)")
    
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        print("  ✓ Google/Gemini API key found")
        has_keys = True
    else:
        print("  ✗ Google API key not found (set GOOGLE_API_KEY or GEMINI_API_KEY)")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        print("  ✓ Anthropic API key found")
        has_keys = True
    else:
        print("  ✗ Anthropic API key not found (set ANTHROPIC_API_KEY)")
    
    if not has_keys:
        print("\n⚠️ Warning: No API keys found. Please set at least one API key to continue.")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export GOOGLE_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)
    
    print()
    asyncio.run(main())


