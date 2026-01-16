"""
Tool base module.
Provides decorator and registry for creating LangGraph tools.
"""

from typing import Any, Callable, TypeVar

from langchain_core.tools import StructuredTool, tool

T = TypeVar("T", bound=Callable[..., Any])

# Registry for all registered tools
_tool_registry: dict[str, StructuredTool] = {}


def register_tool(
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> Callable[[T], StructuredTool]:
    """
    Decorator to register a function as a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        tags: Tags for categorizing/filtering tools

    Example:
        @register_tool(name="search", tags=["web"])
        def search_web(query: str) -> str:
            '''Search for information on the web.'''
            return f"Results for: {query}"

        @register_tool()
        def calculate(expression: str) -> str:
            '''Evaluate a math expression.'''
            return str(eval(expression))
    """
    def decorator(func: T) -> StructuredTool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        # Create the tool using LangChain's tool decorator
        structured_tool = tool(
            name=tool_name,
            description=tool_desc,
        )(func)

        # Store tags as metadata
        if tags:
            structured_tool.metadata = {"tags": tags}

        # Add to registry
        _tool_registry[tool_name] = structured_tool

        return structured_tool

    return decorator


def get_tool(name: str) -> StructuredTool | None:
    """Get a registered tool by name."""
    return _tool_registry.get(name)


def get_tools_by_tag(tag: str) -> list[StructuredTool]:
    """Get all tools with a specific tag."""
    return [
        t for t in _tool_registry.values()
        if t.metadata and tag in t.metadata.get("tags", [])
    ]


def get_all_tools() -> list[StructuredTool]:
    """Get all registered tools."""
    return list(_tool_registry.values())


def clear_registry() -> None:
    """Clear the tool registry. Useful for testing."""
    _tool_registry.clear()
