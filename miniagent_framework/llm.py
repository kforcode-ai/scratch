"""
LLM client with retry policies and streaming support
"""
import asyncio
import time
from typing import AsyncIterator, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from openai import AsyncOpenAI
import os


class RetryStrategy(Enum):
    """Retry strategies for LLM calls"""
    CONSTANT_DELAY = "constant_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 0.3  # seconds
    multiplier: float = 1.5
    max_delay: float = 10.0
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        if self.strategy == RetryStrategy.CONSTANT_DELAY:
            return self.initial_delay
        else:  # EXPONENTIAL_BACKOFF
            delay = self.initial_delay * (self.multiplier ** attempt)
            return min(delay, self.max_delay)


class LLMClient:
    """
    Unified LLM client with retry logic and streaming
    Currently supports OpenAI, can be extended for Gemini, Anthropic
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        retry_policy: Optional[RetryPolicy] = None
    ):
        self.provider = provider
        self.model = model
        self.retry_policy = retry_policy or RetryPolicy()
        
        # Initialize provider client
        if provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        else:
            raise ValueError(f"Provider {provider} not yet supported")
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Any:
        """
        Complete a chat with retry logic
        """
        last_error = None
        
        for attempt in range(self.retry_policy.max_retries):
            try:
                if stream:
                    # Return the generator directly, don't await it
                    return self._stream_completion(messages, temperature, max_tokens, tools)
                else:
                    return await self._single_completion(messages, temperature, max_tokens, tools)
            
            except Exception as e:
                last_error = e
                if attempt < self.retry_policy.max_retries - 1:
                    delay = self.retry_policy.get_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
        
        raise last_error or Exception("Failed after retries")
    
    async def _single_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Single completion without streaming"""
        if not self.client:
            # Mock response for testing - handle decision requests
            return self._generate_mock_response(messages, tools)
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        response = await self.client.chat.completions.create(**kwargs)
        
        # Check for tool calls
        if response.choices[0].message.tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            }
        
        return response.choices[0].message.content
    
    def _generate_mock_response(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Any:
        """Generate appropriate mock response based on context"""
        # Check if this is a decision request
        if messages and "You can:" in str(messages[0].get("content", "")):
            # This is a decision request from _decide_action
            
            # Find the last user message
            last_user_msg = ""
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_msg = messages[i].get("content", "").lower()
                    last_user_idx = i
                    break
            
            # Check if we've already executed tools AFTER this user message
            has_tool_after_user = False
            if last_user_idx >= 0:
                for msg in messages[last_user_idx + 1:]:
                    content = str(msg.get("content", ""))
                    if "[Tool:" in content or "[Tool Result:" in content:
                        has_tool_after_user = True
                        break
            
            # If we've executed tools for this query, time to respond
            if has_tool_after_user:
                return {"action": "respond"}
            
            # Decide based on keywords
            if any(kw in last_user_msg for kw in ["pricing", "price", "cost", "features", "support", "refund"]):
                return {"action": "use_tools", "tools": [{"name": "knowledge_base", "parameters": {"query": last_user_msg}}]}
            elif any(kw in last_user_msg for kw in ["time", "date", "day"]) and any(q in last_user_msg for q in ["what", "current", "tell"]):
                return {"action": "use_tools", "tools": [{"name": "get_datetime", "parameters": {}}]}
            elif any(op in last_user_msg for op in ["+", "-", "*", "/", "^", "calculate", "square root", "power", "sqrt"]) and not "weather" in last_user_msg.lower():
                import re
                # Look for various math expressions
                math_patterns = [
                    r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic operations
                    r'\d+\s*\*\*\s*\d+',  # Power with **
                    r'square root of\s*\d+',  # Square root
                    r'\d+\s*to the power of\s*\d+',  # Power phrase
                    r'\d+\s*\^\s*\d+',  # Power with ^
                    r'\(\s*[\d\s\+\-\*\/]+\s*\)',  # Parentheses expressions
                ]

                for pattern in math_patterns:
                    match = re.search(pattern, last_user_msg, re.IGNORECASE)
                    if match:
                        expr = match.group().strip()
                        # Convert natural language to Python syntax
                        expr = expr.replace("square root of", "sqrt")
                        expr = expr.replace("to the power of", "**")
                        expr = expr.replace("^", "**")
                        # Add sqrt import if needed
                        if "sqrt" in expr:
                            expr = f"import math; math.{expr}"
                        return {"action": "use_tools", "tools": [{"name": "calculate", "parameters": {"expression": expr}}]}
            elif "weather" in last_user_msg.lower():
                # Extract location from weather query
                import re
                # Look for location names in weather queries
                location_patterns = [
                    r'weather in\s+([A-Za-z\s]+)',
                    r'weather like in\s+([A-Za-z\s]+)',
                    r'weather for\s+([A-Za-z\s]+)',
                ]
                for pattern in location_patterns:
                    match = re.search(pattern, last_user_msg, re.IGNORECASE)
                    if match:
                        location = match.group(1).strip().title()
                        return {"action": "use_tools", "tools": [{"name": "get_weather", "parameters": {"location": location}}]}
            elif any(search_kw in last_user_msg.lower() for search_kw in ["search", "find", "lookup", "news about", "latest", "current"]) and not any(skip_kw in last_user_msg.lower() for skip_kw in ["weather", "time", "calculate"]):
                return {"action": "use_tools", "tools": [{"name": "web_search", "parameters": {"query": last_user_msg}}]}
            elif "nifty" in last_user_msg or "stock" in last_user_msg or "market" in last_user_msg:
                return {"action": "use_tools", "tools": [{"name": "web_search", "parameters": {"query": last_user_msg}}]}
            elif "login" in last_user_msg or "password" in last_user_msg:
                return {"action": "respond"}  # Will generate tech support response
            else:
                return {"action": "respond"}  # Default response
        
        # Find the last user message index
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        
        # Check for tool results AFTER the last user message
        tool_result = None
        if last_user_idx >= 0:
            for msg in messages[last_user_idx + 1:]:
                content = str(msg.get("content", ""))
                if "Tool Result" in content:
                    tool_result = content
                    break
        
        if tool_result:
            # Generate response based on the specific tool result
            if "[Tool Result: calculate]" in tool_result:
                # Extract calculation result
                calc_result = tool_result.replace("[Tool Result: calculate]", "").strip()
                if "📐 Calculation:" in calc_result:
                    return calc_result.replace("📐 Calculation:", "The answer is").strip()
                return calc_result
            elif "[Tool Result: knowledge_base]" in tool_result:
                # Extract knowledge base content
                kb_result = tool_result.replace("[Tool Result: knowledge_base]", "").strip()
                if "**Pricing Plans:**" in kb_result:
                    return "Here's our pricing information:\n\n" + kb_result
                elif "**Key Features:**" in kb_result:
                    return "Here are our key features:\n\n" + kb_result
                elif "**Support Options:**" in kb_result:
                    return "Here are our support options:\n\n" + kb_result
                elif "**Refund Policy:**" in kb_result:
                    return "Here's our refund policy:\n\n" + kb_result
                elif "No information found" in kb_result or "No matching" in kb_result:
                    return "I don't have specific information about that in my knowledge base. Is there something else I can help you with?"
                return kb_result
            elif "[Tool Result: get_datetime]" in tool_result:
                dt_result = tool_result.replace("[Tool Result: get_datetime]", "").strip()
                return dt_result.replace("📅 Current date/time:", "The current date and time is")
            elif "[Tool Result: web_search]" in tool_result:
                search_result = tool_result.replace("[Tool Result: web_search]", "").strip()
                if "No search results" in search_result:
                    return "I wasn't able to find specific information about that. Please try rephrasing your query or check a dedicated financial website for the most current data."
                return f"Based on my web search, here's what I found:\n\n{search_result}"
            else:
                return f"Based on the information I found: {tool_result}"
        
        # Generate contextual response
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break
        
        # Check for context-aware responses
        if "recommend" in last_user_msg and ("team" in last_user_msg or "people" in last_user_msg):
            # Check if there's a number mentioned
            import re
            numbers = re.findall(r'\d+', last_user_msg)
            if numbers:
                team_size = int(numbers[0])
                if team_size <= 10:
                    return "Based on your team size of " + str(team_size) + " people, I'd recommend the Starter plan at $9/month. It supports up to 10 users and includes all essential features."
                elif team_size <= 50:
                    return "For a team of " + str(team_size) + " people, the Professional plan at $29/month would be perfect. It supports up to 50 users and includes advanced features like API access."
                else:
                    return "For a team of " + str(team_size) + " people, you'll need our Enterprise plan with custom pricing. It offers unlimited users and can be tailored to your specific needs. Contact our sales team for a quote."
        elif "nifty" in last_user_msg or "stock" in last_user_msg:
            return "I don't have real-time access to stock market data. For current Nifty50 values, please check a financial website or your broker's platform. The market is typically open from 9:15 AM to 3:30 PM IST on weekdays."
        elif "login" in last_user_msg or "password" in last_user_msg:
            return "For login issues, try these steps: 1) Verify your username/email is correct, 2) Use the 'Forgot Password' option to reset, 3) Clear browser cache and cookies, 4) Try a different browser, 5) If the issue persists, contact IT support with your error details."
        elif "help" in last_user_msg or "how can you" in last_user_msg or "what can you" in last_user_msg:
            return "I can help you with product information (pricing, features, support), perform calculations, check the current date/time, and search for information. Feel free to ask me anything!"
        elif "hi" in last_user_msg or "hello" in last_user_msg or "hey" in last_user_msg:
            return "Hello! I'm here to help. You can ask me about our products, pricing, features, or I can help with calculations and general information. What would you like to know?"
        else:
            return "I'm here to help! You can ask me about our products, pricing, features, or I can help with calculations and general information. What would you like to know?"
    
    async def _stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> AsyncIterator[str]:
        """Stream completion chunks"""
        if not self.client:
            # Mock streaming
            response = self._generate_mock_response(messages, tools)
            if isinstance(response, dict):
                # If it's a dict (decision), convert to string for streaming
                response = str(response)
            for word in response.split():
                yield word + " "
                await asyncio.sleep(0.03)
            return
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        stream = await self.client.chat.completions.create(**kwargs)

        tool_calls_buffer = {}
        has_yielded_tool_calls = False

        async for chunk in stream:
            delta = chunk.choices[0].delta

            # Handle content
            if delta.content:
                yield delta.content

            # Handle tool calls - accumulate them but don't yield yet
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    tc_index = getattr(tool_call_delta, 'index', 0)
                    if tc_index not in tool_calls_buffer:
                        tool_calls_buffer[tc_index] = {
                            "id": "",
                            "name": "",
                            "arguments": ""
                        }

                    tc = tool_calls_buffer[tc_index]

                    if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                        tc["id"] = tool_call_delta.id

                    if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                        if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                            tc["name"] = tool_call_delta.function.name
                        if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                            tc["arguments"] += tool_call_delta.function.arguments

        # After streaming is complete, yield tool calls if any
        if tool_calls_buffer and not has_yielded_tool_calls:
            # Convert buffer to proper format
            tool_calls_list = []
            for tc_data in tool_calls_buffer.values():
                if tc_data["name"]:  # Only include if we have a name
                    tool_calls_list.append({
                        "id": tc_data["id"],
                        "name": tc_data["name"],
                        "arguments": tc_data["arguments"]
                    })

            if tool_calls_list:
                yield {"tool_calls": tool_calls_list}


# Convenience functions for creating retry policies
def constant_retry(max_retries: int = 3, delay: float = 0.2) -> RetryPolicy:
    """Create a constant delay retry policy"""
    return RetryPolicy(
        max_retries=max_retries,
        strategy=RetryStrategy.CONSTANT_DELAY,
        initial_delay=delay
    )


def exponential_retry(
    max_retries: int = 3,
    initial_delay: float = 0.3,
    multiplier: float = 1.5,
    max_delay: float = 10.0
) -> RetryPolicy:
    """Create an exponential backoff retry policy"""
    return RetryPolicy(
        max_retries=max_retries,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=initial_delay,
        multiplier=multiplier,
        max_delay=max_delay
    )
