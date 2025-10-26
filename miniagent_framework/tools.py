"""
Tool system with registry, execution, and error handling
"""
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import asyncio
import json
import inspect
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    display_content: Optional[str] = None  # Human-friendly output
    llm_content: Optional[str] = None      # LLM-friendly output
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "display": self.display_content,
            "llm": self.llm_content
        }


class Tool(ABC):
    """Base class for tools"""
    
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for parameters
    
    def should_confirm(self, params: Dict) -> bool:
        """Whether to ask user confirmation before execution"""
        return False
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def to_function_schema(self) -> Dict:
        """Convert to OpenAI function schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class SimpleTool(Tool):
    """Simple tool wrapper for functions"""
    
    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Execute {func.__name__}"
        self.parameters = self._infer_parameters(func)
    
    def _infer_parameters(self, func: Callable) -> Dict:
        """Infer parameters from function signature"""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function"""
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = await asyncio.to_thread(self.func, **kwargs)
            
            # Handle different return types
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict):
                return ToolResult(
                    success=True,
                    data=result,
                    display_content=json.dumps(result, indent=2),
                    llm_content=json.dumps(result)
                )
            else:
                return ToolResult(
                    success=True,
                    data=result,
                    display_content=str(result),
                    llm_content=str(result)
                )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                display_content=f"❌ Error: {str(e)}",
                llm_content=f"Tool execution failed: {str(e)}"
            )


class ToolRegistry:
    """Registry for managing tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        return tool
    
    def tool(self, name: str = None, description: str = None):
        """Decorator for registering simple tools"""
        def decorator(func):
            tool = SimpleTool(func, name, description)
            self.register(tool)
            return func
        return decorator
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self.tools.keys())
    
    def get_schemas(self) -> List[Dict]:
        """Get all tool schemas for LLM in OpenAI format"""
        return [{
            "type": "function",
            "function": tool.to_function_schema()
        } for tool in self.tools.values()]
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        retry_on_error: bool = True,
        max_retries: int = 2
    ) -> ToolResult:
        """Execute a tool with retry logic"""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found",
                display_content=f"❌ Unknown tool: {tool_name}",
                llm_content=f"Tool {tool_name} does not exist"
            )
        
        last_error = None
        for attempt in range(max_retries if retry_on_error else 1):
            try:
                result = await tool.execute(**parameters)
                if result.success or not retry_on_error:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1 and retry_on_error:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Simple backoff
                    continue
        
        return ToolResult(
            success=False,
            data=None,
            error=last_error or "Unknown error",
            display_content=f"❌ Tool failed after {max_retries} attempts",
            llm_content=f"Tool execution failed: {last_error}"
        )


# Built-in tools
class WebSearchTool(Tool):
    """Web search using Tavily"""
    
    name = "web_search"
    description = "Search the web for current information"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
    
    async def execute(self, query: str) -> ToolResult:
        """Execute web search"""
        if not self.api_key:
            # Mock response
            return ToolResult(
                success=True,
                data={"results": [f"Mock result for: {query}"]},
                display_content=f"🔍 Searched for: {query}\n• Mock result 1\n• Mock result 2",
                llm_content=f"Web search results for '{query}': Mock information about {query}"
            )
        
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)
            response = await asyncio.to_thread(
                client.search,
                query=query,
                max_results=3
            )
            
            results = []
            display_results = []
            for r in response.get("results", [])[:3]:
                # Include more content for LLM to work with
                content = r.get('content', '')[:500]  # Increased from 200 to 500 chars
                results.append(f"{r.get('title', '')}: {content}")
                display_results.append(f"• {r.get('title', '')}")

            return ToolResult(
                success=True,
                data=response,
                display_content=f"🔍 Web Search: {query}\n" + "\n".join(display_results),
                llm_content="\n".join(results)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                display_content=f"❌ Search failed: {str(e)}",
                llm_content=f"Web search failed: {str(e)}"
            )


class KnowledgeBaseTool(Tool):
    """Mock knowledge base tool"""
    
    name = "knowledge_base"
    description = "Search internal knowledge base"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
    
    def __init__(self, knowledge: Optional[Dict[str, str]] = None):
        self.knowledge = knowledge or {
            "pricing": "Starter: $9/mo, Pro: $29/mo, Enterprise: Custom",
            "features": "Collaboration, Security, Analytics, Integrations",
            "support": "24/7 support via email and chat"
        }
    
    async def execute(self, query: str) -> ToolResult:
        """Search knowledge base"""
        query_lower = query.lower()
        
        for key, value in self.knowledge.items():
            if key in query_lower:
                return ToolResult(
                    success=True,
                    data={"result": value},
                    display_content=f"📚 Knowledge Base:\n{value}",
                    llm_content=value
                )
        
        return ToolResult(
            success=True,
            data={"result": "No information found"},
            display_content="📚 No matching information in knowledge base",
            llm_content="No relevant information found in knowledge base"
        )


import os
