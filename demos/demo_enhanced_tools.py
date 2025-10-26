#!/usr/bin/env python3
"""
Demo: Enhanced tools with default parameters and token optimization
Shows how to save tokens by only sending required params to LLM
"""
import asyncio
import json
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent_framework.core.tools import (
    EnhancedTool, 
    ToolRegistry,  # Use unified registry
    ToolParameter,
    ParamType,
    ToolResult,
    APITool,
    KnowledgeBaseTool
)


# ============== Custom Tools for Demo ==============

class WeatherAPITool(APITool):
    """
    Weather API with mostly default parameters
    Only location is required from LLM
    """
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get weather information for a location",
            endpoint="https://api.weather.com/v1/weather",
            method="GET",
            auth_header="Bearer weather-api-key-123",
            default_headers={
                "Accept": "application/json",
                "X-API-Version": "1.0"
            },
            default_body={
                "units": "metric",
                "include": ["current", "forecast"],
                "language": "en",
                "timezone": "auto"
            },
            required_body_fields=["location"]  # Only location required from LLM
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Mock execution for demo"""
        location = kwargs.get("location", "Unknown")
        units = kwargs.get("units", "metric")
        
        mock_weather = {
            "location": location,
            "temperature": 22 if units == "metric" else 72,
            "condition": "Partly cloudy",
            "humidity": 65
        }
        
        return ToolResult(
            success=True,
            data=mock_weather,
            display_content=f"🌤️ Weather in {location}: {mock_weather['temperature']}° ({mock_weather['condition']})",
            llm_content=f"Weather in {location}: {mock_weather['temperature']} degrees, {mock_weather['condition']}"
        )


class DatabaseQueryTool(EnhancedTool):
    """
    Database query tool with complex defaults
    """
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="query",
                type=ParamType.STRING,
                description="SQL query or natural language query",
                required=True
            ),
            ToolParameter(
                name="database",
                type=ParamType.STRING,
                description="Database to query",
                required=False,
                default="production",
                enum=["production", "staging", "analytics"]
            ),
            ToolParameter(
                name="limit",
                type=ParamType.INTEGER,
                description="Maximum rows to return",
                required=False,
                default=100,
                min_value=1,
                max_value=1000
            ),
            ToolParameter(
                name="timeout",
                type=ParamType.NUMBER,
                description="Query timeout in seconds",
                required=False,
                default=30.0,
                min_value=1.0,
                max_value=300.0
            ),
            ToolParameter(
                name="cache",
                type=ParamType.BOOLEAN,
                description="Whether to use cached results",
                required=False,
                default=True
            )
        ]
        
        super().__init__(
            name="query_database",
            description="Query company database for information",
            parameters=parameters,
            returns_description="Database query results as JSON"
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Mock database query"""
        query = kwargs.get("query")
        database = kwargs.get("database", "production")
        limit = kwargs.get("limit", 100)
        
        # Mock results
        mock_results = [
            {"id": 1, "name": "Product A", "sales": 1000},
            {"id": 2, "name": "Product B", "sales": 1500}
        ]
        
        return ToolResult(
            success=True,
            data=mock_results[:limit],
            display_content=f"📊 Query executed on {database}: {len(mock_results)} rows",
            llm_content=f"Database query results from {database}: {json.dumps(mock_results[:limit])}"
        )


class EmailTool(EnhancedTool):
    """
    Email tool with many defaults
    """
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="to",
                type=ParamType.STRING,
                description="Recipient email address",
                required=True
            ),
            ToolParameter(
                name="subject",
                type=ParamType.STRING,
                description="Email subject",
                required=True
            ),
            ToolParameter(
                name="body",
                type=ParamType.STRING,
                description="Email body content",
                required=True
            ),
            ToolParameter(
                name="cc",
                type=ParamType.ARRAY,
                description="CC recipients",
                required=False,
                default=[]
            ),
            ToolParameter(
                name="bcc",
                type=ParamType.ARRAY,
                description="BCC recipients",
                required=False,
                default=[]
            ),
            ToolParameter(
                name="priority",
                type=ParamType.STRING,
                description="Email priority",
                required=False,
                default="normal",
                enum=["low", "normal", "high", "urgent"]
            ),
            ToolParameter(
                name="signature",
                type=ParamType.STRING,
                description="Email signature",
                required=False,
                default="Best regards,\nAI Assistant"
            ),
            ToolParameter(
                name="track_opens",
                type=ParamType.BOOLEAN,
                description="Track email opens",
                required=False,
                default=False
            ),
            ToolParameter(
                name="schedule_time",
                type=ParamType.STRING,
                description="Schedule send time (ISO format)",
                required=False,
                default=None  # Send immediately
            )
        ]
        
        super().__init__(
            name="send_email",
            description="Send an email message",
            parameters=parameters,
            returns_description="Email send confirmation with message ID"
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Mock email sending"""
        to = kwargs.get("to")
        subject = kwargs.get("subject")
        priority = kwargs.get("priority", "normal")
        
        return ToolResult(
            success=True,
            data={"message_id": "msg-123456", "status": "queued"},
            display_content=f"📧 Email sent to {to}: {subject} (priority: {priority})",
            llm_content=f"Email successfully sent to {to} with subject '{subject}'"
        )


# ============== Demo Functions ==============

class InventoryAPITool(APITool):
    """
    Example: Inventory management API with all parameters pre-configured
    """
    
    def __init__(self, api_key: str, base_url: str):
        super().__init__(
            name="add_to_inventory",
            description="Add a product to inventory system",
            endpoint=f"{base_url}/api/inventory/add",
            method="POST",
            auth_header=f"Bearer {api_key}",
            default_headers={
                "Content-Type": "application/json",
                "X-API-Version": "2.0"
            },
            default_body={
                "warehouse": "main",
                "category": "general",
                "status": "available",
                "added_by": "ai_agent"
            },
            required_body_fields=["product_name", "quantity", "price"]
        )


def print_comparison(registry: ToolRegistry):
    """
    Show the difference between full schemas and optimized schemas
    """
    print("=" * 80)
    print("📊 SCHEMA COMPARISON: Full vs Optimized for LLM")
    print("=" * 80)
    
    for tool_name, tool in registry.tools.items():
        print(f"\n🔧 Tool: {tool_name}")
        print("-" * 40)
        
        # Check if it's an enhanced tool
        if not isinstance(tool, EnhancedTool):
            continue
            
        # Full schema
        full_schema = tool.get_full_schema()
        full_params = full_schema["parameters"]["properties"]
        
        # Optimized schema for LLM
        optimized_schema = tool.get_schema_for_llm()
        optimized_params = optimized_schema["parameters"]["properties"]
        
        print(f"Full schema parameters: {len(full_params)}")
        for param_name, param_def in full_params.items():
            default = param_def.get("default", "N/A")
            print(f"  • {param_name}: {param_def['type']} (default: {default})")
        
        print(f"\nOptimized schema for LLM: {len(optimized_params)} parameters")
        for param_name in optimized_params:
            print(f"  • {param_name}: {optimized_params[param_name]['type']} (REQUIRED)")
        
        # Calculate token savings
        full_json = json.dumps(full_schema)
        optimized_json = json.dumps(optimized_schema)
        
        full_tokens = len(full_json) // 4  # Rough estimate: 1 token ≈ 4 chars
        optimized_tokens = len(optimized_json) // 4
        savings = full_tokens - optimized_tokens
        savings_percent = (savings / full_tokens) * 100 if full_tokens > 0 else 0
        
        print(f"\n💰 Token Savings:")
        print(f"  Full schema: ~{full_tokens} tokens")
        print(f"  Optimized: ~{optimized_tokens} tokens")
        print(f"  Saved: ~{savings} tokens ({savings_percent:.1f}%)")


async def simulate_llm_interaction(registry: ToolRegistry):
    """
    Simulate how LLM would interact with optimized tools
    """
    print("\n" + "=" * 80)
    print("🤖 SIMULATING LLM INTERACTIONS")
    print("=" * 80)
    
    # Test cases simulating LLM tool calls
    test_cases = [
        {
            "description": "LLM sends email with only required fields",
            "tool": "send_email",
            "llm_params": {
                "to": "user@example.com",
                "subject": "Monthly Report",
                "body": "Please find attached the monthly report."
            }
        },
        {
            "description": "LLM queries database with just the query",
            "tool": "query_database",
            "llm_params": {
                "query": "SELECT * FROM sales WHERE date >= '2024-01-01'"
            }
        },
        {
            "description": "LLM gets weather for a location",
            "tool": "get_weather",
            "llm_params": {
                "location": "San Francisco"
            }
        },
        {
            "description": "LLM searches knowledge base",
            "tool": "knowledge_base",
            "llm_params": {
                "query": "How to reset password?"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print("-" * 40)
        
        tool_name = test_case["tool"]
        llm_params = test_case["llm_params"]
        
        if tool_name not in registry.tools:
            print(f"  ❌ Tool '{tool_name}' not found")
            continue
        
        tool = registry.tools[tool_name]
        
        print(f"  LLM provides: {json.dumps(llm_params, indent=4)}")
        
        # Show what gets merged
        final_params = tool.merge_with_defaults(llm_params)
        
        print(f"\n  After merging with defaults:")
        for key, value in final_params.items():
            if key in llm_params:
                print(f"    • {key}: {value} (from LLM)")
            else:
                print(f"    • {key}: {value} (DEFAULT)")
        
        # Execute
        result = await registry.execute_tool(tool_name, llm_params)
        
        print(f"\n  Result: {'✅ Success' if result.success else '❌ Failed'}")
        if result.display_content:
            print(f"  {result.display_content}")


async def main():
    """
    Main demo showing token optimization with enhanced tools
    """
    print("🚀 Enhanced Tools Demo - Token Optimization\n")
    
    # Create registry with token optimization enabled
    registry = ToolRegistry(optimize_for_tokens=True)
    
    # Register tools with many default parameters
    registry.register(EmailTool())
    registry.register(DatabaseQueryTool())
    registry.register(WeatherAPITool())
    registry.register(KnowledgeBaseTool(
        kb_endpoint="http://kb.internal.com",
        default_limit=5,
        default_threshold=0.8
    ))
    registry.register(InventoryAPITool(
        api_key="inv-key-123",
        base_url="https://inventory.api.com"
    ))
    
    # Show schema comparison
    print_comparison(registry)
    
    # Show what LLM actually receives
    print("\n" + "=" * 80)
    print("📤 WHAT LLM RECEIVES (Optimized Schemas)")
    print("=" * 80)
    
    llm_schemas = registry.get_schemas_for_llm()
    for schema in llm_schemas:
        func = schema["function"]
        print(f"\n{func['name']}:")
        print(f"  Description: {func['description']}")
        print(f"  Required params: {func['parameters'].get('required', [])}")
    
    # Simulate LLM interactions
    await simulate_llm_interaction(registry)
    
    # Show final statistics
    stats = registry.get_stats()
    print("\n" + "=" * 80)
    print("📈 FINAL STATISTICS")
    print("=" * 80)
    print(f"Total executions: {stats['total_executions']}")
    print(f"Total tokens saved: {stats['total_tokens_saved']}")
    print(f"\nPer-tool statistics:")
    for tool_name, tool_stats in stats['tool_stats'].items():
        if tool_stats['executions'] > 0:
            print(f"  • {tool_name}: {tool_stats['executions']} executions, "
                  f"{tool_stats['tokens_saved']} tokens saved")
    
    # Calculate overall savings
    total_tools = len(registry.tools)
    avg_savings_per_tool = stats['total_tokens_saved'] / total_tools if total_tools > 0 else 0
    
    print(f"\n💡 Summary:")
    print(f"  Average tokens saved per tool schema: ~{avg_savings_per_tool:.0f}")
    print(f"  With 100 LLM calls, you'd save: ~{avg_savings_per_tool * 100:.0f} tokens")
    print(f"  With GPT-4 pricing ($0.03/1K tokens), that's: ~${(avg_savings_per_tool * 100 * 0.03 / 1000):.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ENHANCED TOOLS WITH DEFAULT PARAMETERS")
    print("Demonstrating token optimization by sending only required params to LLM")
    print("=" * 80 + "\n")
    
    asyncio.run(main())
