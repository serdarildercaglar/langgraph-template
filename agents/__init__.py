"""Agents module - Agent metadata for API introspection."""

from agents.base import (
    AgentInfo,
    clear_registry,
    get_agent,
    get_all_agents,
    register_agent,
)

__all__ = [
    "AgentInfo",
    "clear_registry",
    "get_agent",
    "get_all_agents",
    "register_agent",
]
