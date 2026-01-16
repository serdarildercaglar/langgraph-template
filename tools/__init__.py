"""Tools module - Tool registration and built-in tools."""

from tools.base import (
    clear_registry,
    get_all_tools,
    get_tool,
    get_tools_by_tag,
    register_tool,
)

# Import builtin tools to register them
from tools.builtin import calculator, current_time, echo

__all__ = [
    # Base utilities
    "clear_registry",
    "get_all_tools",
    "get_tool",
    "get_tools_by_tag",
    "register_tool",
    # Built-in tools
    "calculator",
    "current_time",
    "echo",
]
