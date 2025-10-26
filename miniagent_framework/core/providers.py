"""
LLM Provider implementations for OpenAI, Google Gemini, and Anthropic Claude
"""
import asyncio
import json
import time
import os
import logging
from typing import AsyncIterator, Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============== Base Provider Interface ==============

class BaseLLMProvider(ABC):
    """Base interface for LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client"""
        pass
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Union[str, Dict, AsyncIterator]:
        """Complete a chat request"""
        pass
    
    def _convert_tools_to_provider_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to provider-specific format (override if needed)"""
        return tools


# ============== OpenAI Provider ==============

class OpenAIProvider(BaseLLMProvider):
    """OpenAI/GPT provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or "gpt-4o-mini"
        super().__init__(self.api_key, self.model)
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
            self.client = None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Union[str, Dict, AsyncIterator]:
        """Complete using OpenAI API"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Please provide OPENAI_API_KEY.")
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        if stream:
            kwargs["stream"] = True
            response = await self.client.chat.completions.create(**kwargs)
            return self._stream_response(response)
        else:
            response = await self.client.chat.completions.create(**kwargs)
            return self._parse_response(response)
    
    def _parse_response(self, response) -> Union[str, Dict]:
        """Parse OpenAI response"""
        message = response.choices[0].message
        
        # Check for tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                    for tc in message.tool_calls
                ]
            }
        
        return message.content
    
    async def _stream_response(self, stream) -> AsyncIterator:
        """Stream OpenAI response"""
        tool_calls_buffer = {}
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            # Handle content
            if delta.content:
                yield delta.content
            
            # Handle tool calls
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
        
        # Yield tool calls at the end if any
        if tool_calls_buffer:
            tool_calls_list = []
            for tc_data in tool_calls_buffer.values():
                if tc_data["name"]:
                    tool_calls_list.append({
                        "id": tc_data["id"],
                        "name": tc_data["name"],
                        "arguments": tc_data["arguments"]
                    })
            
            if tool_calls_list:
                yield {"tool_calls": tool_calls_list}


# ============== Google/Gemini Provider ==============

class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model or "gemini-1.5-flash"
        super().__init__(self.api_key, self.model)
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            else:
                self.client = None
        except ImportError:
            logger.warning("Google AI library not installed. Install with: pip install google-generativeai")
            self.client = None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Union[str, Dict, AsyncIterator]:
        """Complete using Gemini API"""
        if not self.client:
            raise RuntimeError("Gemini client not initialized. Please provide GOOGLE_API_KEY or GEMINI_API_KEY.")
        
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages(messages)
        
        # Configure generation
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Handle tools if provided
        if tools:
            gemini_tools = self._convert_tools_to_gemini(tools)
            response = await asyncio.to_thread(
                self.client.generate_content,
                gemini_messages,
                generation_config=generation_config,
                tools=gemini_tools,
                stream=stream
            )
        else:
            response = await asyncio.to_thread(
                self.client.generate_content,
                gemini_messages,
                generation_config=generation_config,
                stream=stream
            )
        
        if stream:
            return self._stream_response(response)
        else:
            return self._parse_response(response)
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Gemini format"""
        # Gemini uses a simpler format - we concatenate into a single prompt
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _convert_tools_to_gemini(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Gemini function format"""
        import google.generativeai as genai
        
        gemini_functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                # Create function declaration
                gemini_func = genai.protos.FunctionDeclaration(
                    name=func.get("name"),
                    description=func.get("description"),
                    parameters=func.get("parameters", {})
                )
                gemini_functions.append(gemini_func)
        
        if gemini_functions:
            return [genai.protos.Tool(function_declarations=gemini_functions)]
        return []
    
    def _parse_response(self, response) -> Union[str, Dict]:
        """Parse Gemini response"""
        # Check for function calls
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check for function calls in the response
            if hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        return {
                            "type": "tool_calls",
                            "tool_calls": [{
                                "id": f"gemini_{fc.name}_{time.time()}",
                                "name": fc.name,
                                "arguments": json.dumps(dict(fc.args))
                            }]
                        }
        
        # Return text content
        return response.text
    
    async def _stream_response(self, response_stream) -> AsyncIterator:
        """Stream Gemini response"""
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
            
            # Check for function calls in streaming
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            fc = part.function_call
                            yield {
                                "tool_calls": [{
                                    "id": f"gemini_{fc.name}_{time.time()}",
                                    "name": fc.name,
                                    "arguments": json.dumps(dict(fc.args))
                                }]
                            }


# ============== Anthropic/Claude Provider ==============

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or "claude-3-haiku-20240307"
        super().__init__(self.api_key, self.model)
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.warning("Anthropic library not installed. Install with: pip install anthropic")
            self.client = None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Union[str, Dict, AsyncIterator]:
        """Complete using Anthropic API"""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Please provide ANTHROPIC_API_KEY.")
        
        # Convert messages to Anthropic format
        system_prompt, claude_messages = self._convert_messages(messages)
        
        kwargs = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        # Handle tools if provided
        if tools:
            kwargs["tools"] = self._convert_tools_to_anthropic(tools)
            kwargs["tool_choice"] = {"type": "auto"}
        
        if stream:
            kwargs["stream"] = True
            response = await self.client.messages.create(**kwargs)
            return self._stream_response(response)
        else:
            response = await self.client.messages.create(**kwargs)
            return self._parse_response(response)
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple[str, List[Dict]]:
        """Convert OpenAI-style messages to Anthropic format"""
        system_prompt = ""
        claude_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                claude_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                claude_messages.append({"role": "assistant", "content": content})
        
        return system_prompt, claude_messages
    
    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic tool format"""
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "input_schema": func.get("parameters", {})
                })
        
        return anthropic_tools
    
    def _parse_response(self, response) -> Union[str, Dict]:
        """Parse Anthropic response"""
        # Check for tool use
        if hasattr(response, 'content'):
            for content_block in response.content:
                if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                    return {
                        "type": "tool_calls",
                        "tool_calls": [{
                            "id": content_block.id,
                            "name": content_block.name,
                            "arguments": json.dumps(content_block.input)
                        }]
                    }
        
        # Return text content
        if hasattr(response.content[0], 'text'):
            return response.content[0].text
        return str(response.content[0])
    
    async def _stream_response(self, response_stream) -> AsyncIterator:
        """Stream Anthropic response"""
        async for event in response_stream:
            # Handle text chunks
            if event.type == 'content_block_delta':
                if hasattr(event.delta, 'text'):
                    yield event.delta.text
            
            # Handle tool use in streaming
            elif event.type == 'content_block_start':
                if hasattr(event.content_block, 'type') and event.content_block.type == 'tool_use':
                    # Start accumulating tool call
                    pass
            
            elif event.type == 'content_block_stop':
                if hasattr(event, 'content_block') and event.content_block.type == 'tool_use':
                    yield {
                        "tool_calls": [{
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "arguments": json.dumps(event.content_block.input)
                        }]
                    }


