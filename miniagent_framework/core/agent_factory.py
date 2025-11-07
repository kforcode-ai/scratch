"""
Factory helpers for composing slim agents with layered system prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence

from .agent import Agent, AgentConfig
from .tools import Tool, ToolRegistry


DEFAULT_BASE_PROMPT = (
    "You are a capable assistant. Use tools when they help and answer directly when you can."
)
DEFAULT_TOOL_HEADER = "Available tools:"


def compose_system_prompt(
    *,
    base_prompt: str,
    creator_prompt: Optional[str],
    tool_registry: ToolRegistry,
    tool_section_header: str = DEFAULT_TOOL_HEADER,
) -> str:
    """
    Build a system prompt from the framework base, creator instructions, and tool summaries.
    """
    base = (base_prompt or "").strip()
    creator = (creator_prompt or "").strip()

    tool_lines = []
    for tool in tool_registry.tools.values():
        description = getattr(tool, "description", "") or "No description provided."
        tool_lines.append(f"- {tool.name}: {description.strip()}")

    tool_section = ""
    if tool_lines:
        tool_section = "\n".join([tool_section_header, *tool_lines]).strip()

    parts = [part for part in (base, creator, tool_section) if part]
    return "\n\n".join(parts)


def _clone_config(config: AgentConfig) -> AgentConfig:
    """
    Create a shallow clone of the provided config to avoid mutating shared instances.
    """
    if hasattr(config, "model_copy"):
        return config.model_copy()
    if hasattr(config, "__dict__"):
        data = dict(config.__dict__)
        return AgentConfig(**data)
    return replace(config)


@dataclass
class AgentFactory:
    """
    Utility for constructing slim agents with consistent prompt composition.
    """

    base_config: Optional[AgentConfig] = None
    base_prompt: str = DEFAULT_BASE_PROMPT
    creator_prompt: str = ""
    tool_section_header: str = DEFAULT_TOOL_HEADER

    def create(
        self,
        *,
        config: Optional[AgentConfig] = None,
        overrides: Optional[Dict[str, object]] = None,
        tools: Optional[Sequence[Tool]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        creator_prompt: Optional[str] = None,
        base_prompt: Optional[str] = None,
        llm_client=None,
        callbacks=None,
    ) -> Agent:
        """
        Instantiate an Agent with optional config overrides and registered tools.
        """
        registry = tool_registry or ToolRegistry()
        if tools:
            for tool in tools:
                registry.register(tool)

        source_config = config or self.base_config or AgentConfig()
        working_config = _clone_config(source_config)

        if overrides:
            for key, value in overrides.items():
                setattr(working_config, key, value)

        resolved_base_prompt = (base_prompt or self.base_prompt).strip()
        resolved_creator_prompt = (
            self.creator_prompt if creator_prompt is None else creator_prompt.strip()
        )
        working_config.system_prompt = compose_system_prompt(
            base_prompt=resolved_base_prompt,
            creator_prompt=resolved_creator_prompt,
            tool_registry=registry,
            tool_section_header=self.tool_section_header,
        )

        agent = Agent(
            config=working_config,
            tools=registry,
            llm_client=llm_client,
            callbacks=callbacks,
        )
        self._stamp_agent(agent, resolved_base_prompt, resolved_creator_prompt)
        return agent

    def register_tool(
        self,
        agent: Agent,
        tool: Tool,
        *,
        creator_prompt: Optional[str] = None,
        base_prompt: Optional[str] = None,
    ) -> Tool:
        """
        Register a tool on an existing agent and rebuild the system prompt.
        """
        registered_tool = agent.tools.register(tool)
        self.refresh_system_prompt(
            agent,
            creator_prompt=creator_prompt,
            base_prompt=base_prompt,
        )
        return registered_tool

    def refresh_system_prompt(
        self,
        agent: Agent,
        *,
        creator_prompt: Optional[str] = None,
        base_prompt: Optional[str] = None,
    ) -> None:
        """
        Rebuild the agent system prompt based on the current tool registry.
        """
        base = (
            base_prompt.strip()
            if base_prompt is not None
            else getattr(agent, "_base_prompt", self.base_prompt)
        )
        creator = (
            creator_prompt.strip()
            if creator_prompt is not None
            else getattr(agent, "_creator_prompt", self.creator_prompt)
        )
        agent.config.system_prompt = compose_system_prompt(
            base_prompt=base,
            creator_prompt=creator,
            tool_registry=agent.tools,
            tool_section_header=getattr(
                agent, "_tool_section_header", self.tool_section_header
            ),
        )
        self._stamp_agent(agent, base, creator)

    def _stamp_agent(self, agent: Agent, base_prompt: str, creator_prompt: str) -> None:
        """
        Store the prompts on the agent for subsequent refresh operations.
        """
        agent._base_prompt = base_prompt
        agent._creator_prompt = creator_prompt
        agent._tool_section_header = self.tool_section_header
