"""
Interactive demo of MiniAgent framework
"""
import asyncio
import sys
import os
import textwrap
from dotenv import load_dotenv

# Add parent directory to path to import framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent_framework.core import Agent, AgentConfig, Thread
from miniagent_framework.core.tools import (
    ToolRegistry,
    KnowledgeBaseTool,
    WebSearchTool,
    DateTimeTool,
)
from miniagent_framework.core.events import StreamCallback, EventType
from miniagent_framework.core.llm import constant_retry

# Load environment variables
load_dotenv()


async def main():
    """Interactive demo of the framework"""
    
    print("="*60)
    print("🤖 MiniAgent Framework - Interactive Demo")
    print("="*60)
    
    # 1. Set up tools
    registry = ToolRegistry()
    
    # Add knowledge base with more content
    registry.register(KnowledgeBaseTool({
        "pricing": """📊 **Pricing Plans:**
• Starter: $9/month (10 users, 100GB storage)
• Professional: $29/month (50 users, 1TB storage)
• Enterprise: Custom pricing (unlimited users, custom features)
All plans include SSL, daily backups, and 99.9% uptime guarantee.""",
        
        "features": """✨ **Key Features:**
• Real-time collaboration and document sharing
• Advanced security with 2FA and SSO
• Analytics dashboard with custom reports
• 1000+ integrations (Slack, Teams, Google, etc.)
• Mobile apps for iOS and Android
• API access (Pro and Enterprise)""",
        
        "support": """🎧 **Support Options:**
• 24/7 email support (all plans)
• Live chat support (Professional+)
• Phone support (Enterprise)
• Dedicated account manager (Enterprise)
• Community forum and knowledge base""",
        
        "refund": """💰 **Refund Policy:**
• 30-day money-back guarantee
• No questions asked
• Full refund for first-time customers
• Pro-rated refunds for annual plans"""
    }))
    
    # Add web search tool
    registry.register(WebSearchTool())
    
    # Add date/time tool
    registry.register(DateTimeTool())
    
    # Add a simple custom tool
    @registry.tool(description="Calculate math expressions")
    async def calculate(expression: str):
        try:
            # Safe eval for simple math
            allowed_names = {
                k: v for k, v in __builtins__.items() 
                if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"📐 Calculation: {expression} = {result}"
        except:
            return "❌ Invalid expression. Try something like: 2+2, 10*5, 100/4"
    
    # 2. Set up callbacks to show what's happening
    callbacks = StreamCallback()
    stream_state = {"chunks": False}
    
    # Show when agent is thinking
    def thinking_handler(event):
        content = event.content
        if isinstance(content, dict):
            if "plan" in content:
                print("🗺️ Plan:")
                print(textwrap.indent(content["plan"], "   "))
                return
            if "plan_progress" in content:
                total = content.get("total_steps")
                print(f"✅ Plan progress: completed step {content['plan_progress']} of {total}")
                return
            if "tool_calls" in content:
                print("💭 Thinking: analysing tool requirements...")
                return
            if "action" in content:
                print(f"💭 Thinking: {content['action']}")
                return
        if isinstance(content, str):
            print(f"💭 Thinking: {content}")
        else:
            print("💭 Thinking: processing...")

    callbacks.on(EventType.AGENT_THINKING, thinking_handler)
    
    # Show tool usage
    callbacks.on(EventType.TOOL_EXECUTION, 
        lambda e: print(f"🔧 Using tool: {e.content['tool']} {e.content.get('parameters', {})}")
    )
    
    # Show tool results briefly
    callbacks.on(EventType.TOOL_RESULT,
        lambda e: print(f"   ✓ Tool completed" if e.content.get('success') else f"   ✗ Tool failed")
    )
    
    # Show streaming (print each chunk inline)
    def stream_start_handler(event):
        stream_state["chunks"] = False

    def stream_chunk_handler(event):
        stream_state["chunks"] = True
        chunk = event.content or ""
        print(chunk, end="", flush=True)

    def stream_end_handler(event):
        if stream_state["chunks"]:
            print()

    callbacks.on(EventType.STREAM_START, stream_start_handler)
    callbacks.on(EventType.STREAM_CHUNK, stream_chunk_handler)
    callbacks.on(EventType.STREAM_END, stream_end_handler)
    
    # 3. Configure agent
    config = AgentConfig(
        name="MiniBot",
        model="gpt-4.1",
        system_prompt="""You are a helpful AI assistant with access to various tools.

Available tools:
- web_search: Search the web for current information
- knowledge_base: Access internal knowledge about pricing, features, support
- calculate: Perform mathematical calculations
- get_datetime: Get current date and time

IMPORTANT: When you receive tool results, use the actual data provided to give accurate answers. Do not say you don't have access to data if tool results are provided. Extract and summarize the relevant information from tool results to answer user questions directly.

Use the appropriate tools to provide accurate, helpful responses. For product information (pricing, features, support), use the knowledge_base tool. For web searches, use web_search. For calculations, use calculate. For time/date, use get_datetime.""",
        retry_policy=constant_retry(max_retries=2, delay=0.5),
        stream_by_default=True,  # Enable streaming for better UX
        planning_enabled=True,

    )
    
    # 4. Create agent and thread for conversation
    agent = Agent(config=config, tools=registry, callbacks=callbacks)
    thread = Thread()  # Maintain conversation history
    
    # 5. Welcome message
    print("\n🎉 Welcome! I'm MiniBot, your AI assistant.")
    # print("I can help with:")
    # print("  • Product information (pricing, features, support)")
    # print("  • Web searches")
    # print("  • Math calculations")
    # print("  • Current date/time")
    # print("\nCommands:")
    # print("  • Type 'quit' or 'exit' to end")
    # print("  • Type 'reset' to clear conversation history")
    # print("  • Type 'help' for this message")
    # print("  • Type 'stream on/off' to toggle streaming")
    print("-"*60)

    stream_enabled = True  # Enable streaming for better UX
    
    # 6. Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! Thanks for using MiniAgent!")
                break
            
            elif user_input.lower() == 'reset':
                thread = Thread()  # New thread
                print("🔄 Conversation history cleared!")
                continue
            
            elif user_input.lower() == 'help':
                print("\n📚 Available commands:")
                print("  • Ask about pricing, features, or support")
                print("  • Ask me to calculate something (e.g., 'What's 25 * 4?')")
                print("  • Ask for the current date/time")
                print("  • Ask me to search the web for information")
                print("  • Type 'reset' to clear history")
                print("  • Type 'quit' to exit")
                continue
            
            elif user_input.lower() == 'stream on':
                stream_enabled = True
                print("✅ Streaming enabled")
                continue
            
            elif user_input.lower() == 'stream off':
                stream_enabled = False
                print("✅ Streaming disabled")
                continue
            
            elif not user_input:
                continue
            
            # Process with agent
            print("-"*40)
            print("🤖 MiniBot: ", end="")
            
            stream_state["chunks"] = False
            response = await agent.run(
                user_input=user_input,
                thread=thread,  # Maintain conversation context
                stream=stream_enabled
            )

            # Always display the final response (streaming shows intermediate chunks, but final response might not be streamed)
            if (not stream_enabled) or (not stream_state["chunks"]):
                print(response)

            stream_state["chunks"] = False
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'help' for assistance.")
    
    print("\n" + "="*60)
    print("✨ Thanks for trying MiniAgent Framework!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
