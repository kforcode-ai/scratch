"""
Tool system with registry, execution, error handling, and token optimization
"""
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import json
import inspect
import logging
import aiohttp
from enum import Enum

logger = logging.getLogger(__name__)


# ============== Core Data Structures ==============

class ParamType(Enum):
    """Parameter types for validation"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    display_content: Optional[str] = None  # Human-friendly output
    llm_content: Optional[str] = None      # LLM-friendly output
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "display": self.display_content,
            "llm": self.llm_content,
            "metadata": self.metadata
        }


@dataclass
class ToolParameter:
    """
    Enhanced parameter definition with defaults and validation
    """
    name: str
    type: ParamType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    
    def to_schema(self, include_defaults: bool = False) -> Dict[str, Any]:
        """Convert to OpenAI function parameter schema"""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
        
        if self.type == ParamType.NUMBER or self.type == ParamType.INTEGER:
            if self.min_value is not None:
                schema["minimum"] = self.min_value
            if self.max_value is not None:
                schema["maximum"] = self.max_value
        
        if self.pattern and self.type == ParamType.STRING:
            schema["pattern"] = self.pattern
        
        # Only include default in schema if explicitly requested
        if include_defaults and self.default is not None:
            schema["default"] = self.default
        
        return schema
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate parameter value"""
        if value is None and self.required:
            return False, f"Required parameter '{self.name}' is missing"
        
        if value is None:
            return True, None
        
        # Type validation
        type_validators = {
            ParamType.STRING: lambda v: isinstance(v, str),
            ParamType.NUMBER: lambda v: isinstance(v, (int, float)),
            ParamType.INTEGER: lambda v: isinstance(v, int),
            ParamType.BOOLEAN: lambda v: isinstance(v, bool),
            ParamType.OBJECT: lambda v: isinstance(v, dict),
            ParamType.ARRAY: lambda v: isinstance(v, list)
        }
        
        if not type_validators[self.type](value):
            return False, f"Parameter '{self.name}' must be of type {self.type.value}"
        
        # Enum validation
        if self.enum and value not in self.enum:
            return False, f"Parameter '{self.name}' must be one of {self.enum}"
        
        # Range validation
        if self.type in [ParamType.NUMBER, ParamType.INTEGER]:
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"
        
        return True, None


# ============== Base Tool Classes ==============

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
    
    def get_schema(self) -> Dict:
        """Get tool schema for LLM (backward compatibility)"""
        return self.to_function_schema()
    
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


# ============== Enhanced Tool with Token Optimization ==============

class EnhancedTool(Tool):
    """
    Enhanced tool with parameter management and token optimization
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        returns_description: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.tool_parameters = {p.name: p for p in parameters}
        self.returns_description = returns_description
        self._execution_count = 0
        self._total_tokens_saved = 0
        
        # Build legacy parameters format for backward compatibility
        self.parameters = self._build_legacy_parameters()
    
    def _build_legacy_parameters(self) -> Dict[str, Any]:
        """Build parameters in legacy format"""
        properties = {}
        required = []
        
        for param_name, param in self.tool_parameters.items():
            properties[param_name] = param.to_schema(include_defaults=True)
            if param.required:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def get_schema_for_llm(self) -> Dict[str, Any]:
        """
        Get optimized schema for LLM (only required parameters)
        This reduces token usage by excluding optional parameters with defaults
        """
        required_params = []
        properties = {}
        
        for param_name, param in self.tool_parameters.items():
            if param.required or param.default is None:
                # Only include required params or optional without defaults
                properties[param_name] = param.to_schema(include_defaults=False)
                if param.required:
                    required_params.append(param_name)
        
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }
        
        # Add a hint about defaults if there are any
        optional_with_defaults = [
            name for name, p in self.tool_parameters.items() 
            if not p.required and p.default is not None
        ]
        
        if optional_with_defaults:
            schema["description"] += f" (has defaults for: {', '.join(optional_with_defaults)})"
        
        self._calculate_token_savings(properties)
        
        return schema
    
    def get_full_schema(self) -> Dict[str, Any]:
        """Get complete schema including all parameters and defaults"""
        properties = {}
        required_params = []
        
        for param_name, param in self.tool_parameters.items():
            properties[param_name] = param.to_schema(include_defaults=True)
            if param.required:
                required_params.append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            },
            "returns": self.returns_description
        }
    
    def merge_with_defaults(self, llm_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge LLM-provided parameters with defaults
        This is called by the execution layer
        """
        final_params = {}
        
        for param_name, param_def in self.tool_parameters.items():
            if param_name in llm_params:
                # Use LLM-provided value
                final_params[param_name] = llm_params[param_name]
            elif param_def.default is not None:
                # Use default value
                final_params[param_name] = param_def.default
                logger.debug(f"Using default for {param_name}: {param_def.default}")
            elif not param_def.required:
                # Optional parameter without default, skip
                continue
            else:
                # Required parameter missing - this should have been caught by validation
                logger.warning(f"Required parameter {param_name} missing for tool {self.name}")
        
        return final_params
    
    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate all parameters"""
        for param_name, param_def in self.tool_parameters.items():
            value = params.get(param_name)
            valid, error = param_def.validate(value)
            if not valid:
                return False, error
        
        # Check for unexpected parameters
        unexpected = set(params.keys()) - set(self.tool_parameters.keys())
        if unexpected:
            logger.warning(f"Unexpected parameters for {self.name}: {unexpected}")
        
        return True, None
    
    async def execute_with_defaults(self, llm_params: Dict[str, Any]) -> ToolResult:
        """
        Execute tool with merged parameters
        This is the main entry point from the agent
        """
        # Merge with defaults
        final_params = self.merge_with_defaults(llm_params)
        
        # Validate final parameters
        valid, error = self.validate_params(final_params)
        if not valid:
            return ToolResult(
                success=False,
                error=error,
                metadata={"validation_error": True}
            )
        
        # Execute with final parameters
        try:
            self._execution_count += 1
            result = await self.execute(**final_params)
            result.metadata["execution_count"] = self._execution_count
            result.metadata["tokens_saved"] = self._total_tokens_saved
            return result
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"exception": True}
            )
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Implement actual tool execution"""
        pass
    
    def _calculate_token_savings(self, optimized_schema: Dict):
        """Estimate tokens saved by not sending defaults"""
        full_schema = {
            name: p.to_schema(include_defaults=True) 
            for name, p in self.tool_parameters.items()
        }
        
        # Rough token estimation (1 token ≈ 4 chars)
        full_tokens = len(json.dumps(full_schema)) // 4
        optimized_tokens = len(json.dumps(optimized_schema)) // 4
        self._total_tokens_saved += (full_tokens - optimized_tokens)


# ============== Tool Registry ==============

class ToolRegistry:
    """Registry for managing tools (supports both basic and enhanced)"""
    
    def __init__(self, optimize_for_tokens: bool = False):
        self.tools: Dict[str, Tool] = {}
        self.optimize_for_tokens = optimize_for_tokens
        self._total_executions = 0
        self._total_tokens_saved = 0
    
    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({'enhanced' if isinstance(tool, EnhancedTool) else 'basic'})")
        return tool
    
    def tool(self, name: str = None, description: str = None):
        """Decorator for registering simple tools"""
        def decorator(func):
            tool = SimpleTool(func, name, description)
            self.register(tool)
            return func
        return decorator
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM"""
        schemas = []
        for tool in self.tools.values():
            if self.optimize_for_tokens and isinstance(tool, EnhancedTool):
                # Use optimized schema for enhanced tools
                schema = tool.get_schema_for_llm()
            else:
                # Use regular schema
                schema = tool.to_function_schema()
            
            # Wrap in OpenAI format
            schemas.append({
                "type": "function",
                "function": schema
            })
        
        return schemas
    
    def get_schemas_for_llm(self) -> List[Dict]:
        """Alias for get_schemas with token optimization"""
        self.optimize_for_tokens = True
        return self.get_schemas()
    
    def get_full_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get complete schemas for documentation/debugging"""
        schemas = {}
        for name, tool in self.tools.items():
            if isinstance(tool, EnhancedTool):
                schemas[name] = tool.get_full_schema()
            else:
                schemas[name] = tool.to_function_schema()
        return schemas
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                display_content=f"❌ Unknown tool: {tool_name}"
            )
        
        self._total_executions += 1
        
        # Execute based on tool type
        if isinstance(tool, EnhancedTool):
            result = await tool.execute_with_defaults(parameters)
            self._total_tokens_saved += tool._total_tokens_saved
        else:
            result = await tool.execute(**parameters)
        
        logger.info(f"Executed {tool_name} (#{self._total_executions})")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        enhanced_tools = sum(1 for t in self.tools.values() if isinstance(t, EnhancedTool))
        basic_tools = len(self.tools) - enhanced_tools
        
        stats = {
            "total_executions": self._total_executions,
            "total_tokens_saved": self._total_tokens_saved,
            "tools_registered": len(self.tools),
            "enhanced_tools": enhanced_tools,
            "basic_tools": basic_tools,
            "tool_stats": {}
        }
        
        # Add per-tool stats for enhanced tools
        for name, tool in self.tools.items():
            if isinstance(tool, EnhancedTool):
                stats["tool_stats"][name] = {
                    "executions": tool._execution_count,
                    "tokens_saved": tool._total_tokens_saved
                }
        
        return stats


# ============== Specialized Tool Implementations ==============

class APITool(EnhancedTool):
    """
    Generic API tool with authentication and defaults
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        endpoint: str,
        method: str = "POST",
        auth_header: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        default_body: Optional[Dict[str, Any]] = None,
        required_body_fields: Optional[List[str]] = None
    ):
        # Build parameters based on what's required
        parameters = []
        
        # Only add body fields that don't have defaults as required
        if required_body_fields:
            for field in required_body_fields:
                parameters.append(ToolParameter(
                    name=field,
                    type=ParamType.STRING,  # Could be enhanced with type hints
                    description=f"Required field: {field}",
                    required=True
                ))
        
        # Add optional parameters that can be overridden
        if default_body:
            for key, value in default_body.items():
                if not required_body_fields or key not in required_body_fields:
                    param_type = self._infer_type(value)
                    parameters.append(ToolParameter(
                        name=key,
                        type=param_type,
                        description=f"Optional field: {key}",
                        required=False,
                        default=value
                    ))
        
        super().__init__(name, description, parameters)
        
        self.endpoint = endpoint
        self.method = method
        self.auth_header = auth_header
        self.default_headers = default_headers or {}
        self.default_body = default_body or {}
    
    def _infer_type(self, value: Any) -> ParamType:
        """Infer parameter type from value"""
        if isinstance(value, bool):
            return ParamType.BOOLEAN
        elif isinstance(value, int):
            return ParamType.INTEGER
        elif isinstance(value, float):
            return ParamType.NUMBER
        elif isinstance(value, list):
            return ParamType.ARRAY
        elif isinstance(value, dict):
            return ParamType.OBJECT
        else:
            return ParamType.STRING
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute API call with merged parameters"""
        headers = self.default_headers.copy()
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        
        # Build body from kwargs (already merged with defaults)
        body = kwargs
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=self.method,
                    url=self.endpoint,
                    headers=headers,
                    json=body
                ) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        return ToolResult(
                            success=True,
                            data=data,
                            display_content=f"✅ API call successful",
                            llm_content=f"Successfully called {self.name} API"
                        )
                    else:
                        return ToolResult(
                            success=False,
                            error=f"API returned status {response.status}",
                            data=data
                        )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


# ============== Built-in Tools ==============

class WebSearchTool(Tool):
    """Web search using Tavily API"""
    
    def __init__(self, api_key: str = None):
        import os
        self.name = "web_search"
        self.description = "Search the web for information"
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    
    def get_schema(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    async def execute(self, query: str) -> ToolResult:
        """Execute web search"""
        if not self.api_key:
            return ToolResult(
                success=False,
                error="Tavily API key not configured",
                display_content="❌ Web search not configured"
            )
        
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)
            
            results = await asyncio.to_thread(
                client.search,
                query=query,
                search_depth="basic",
                max_results=3
            )
            
            # Format results
            formatted_results = []
            for result in results.get('results', []):
                formatted_results.append({
                    "title": result.get("title"),
                    "content": result.get("content"),
                    "url": result.get("url")
                })
            
            display = f"🔍 Search results for '{query}':\n\n"
            for i, r in enumerate(formatted_results, 1):
                display += f"{i}. **{r['title']}**\n   {r['content'][:200]}...\n   {r['url']}\n\n"
            
            return ToolResult(
                success=True,
                data=formatted_results,
                display_content=display,
                llm_content=json.dumps(formatted_results)
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                display_content=f"❌ Search failed: {str(e)}"
            )


class KnowledgeBaseTool(Tool):
    """Simple knowledge base tool"""
    
    def __init__(self, knowledge: Dict[str, str] = None):
        self.name = "knowledge_base"
        self.description = "Query internal knowledge base"
        self.knowledge = knowledge or {}
        self.parameters = {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to search for in knowledge base"
                }
            },
            "required": ["topic"]
        }
    
    def get_schema(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    async def execute(self, topic: str) -> ToolResult:
        """Search knowledge base"""
        # Find matching topics
        matches = []
        topic_lower = topic.lower()
        
        for key, value in self.knowledge.items():
            if topic_lower in key.lower() or topic_lower in value.lower():
                matches.append((key, value))
        
        if matches:
            display = f"📚 Knowledge base results for '{topic}':\n\n"
            llm_content = []
            
            for key, value in matches[:3]:  # Limit to 3 results
                display += f"**{key}**:\n{value}\n\n"
                llm_content.append({
                    "topic": key,
                    "content": value
                })
            
            return ToolResult(
                success=True,
                data=llm_content,
                display_content=display.strip(),
                llm_content=json.dumps(llm_content)
            )
        else:
            return ToolResult(
                success=True,
                data=[],
                display_content=f"📚 No information found about '{topic}'",
                llm_content=f"No knowledge base entries found for topic: {topic}"
            )