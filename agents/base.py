"""
Agent metadata module.

Note: With LangGraph 2025 best practices, you typically don't need
a complex agent registry. Instead:

1. Single agent with tools: Use create_react_agent directly
2. Multi-agent: Use tool-based handoff pattern

This module provides minimal utilities for API introspection
(e.g., /agents endpoint to list available agents).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentInfo:
    """
    Metadata about an agent for API introspection.
    This is NOT an executable agent - just metadata for documentation.
    """
    name: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Simple registry for agent metadata
_agent_registry: dict[str, AgentInfo] = {}


def register_agent(name: str, description: str, **metadata: Any) -> AgentInfo:
    """Register agent metadata for API introspection."""
    info = AgentInfo(name=name, description=description, metadata=metadata)
    _agent_registry[name] = info
    return info


def get_agent(name: str) -> AgentInfo | None:
    """Get agent metadata by name."""
    return _agent_registry.get(name)


def get_all_agents() -> list[AgentInfo]:
    """Get all registered agent metadata."""
    return list(_agent_registry.values())


def clear_registry() -> None:
    """Clear the registry. Useful for testing."""
    _agent_registry.clear()


# Register default agent
register_agent(
    name="assistant",
    description="Main AI assistant with access to all registered tools",
)
