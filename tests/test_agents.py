"""Tests for agent and tool functionality."""

import pytest
from unittest.mock import MagicMock

from agents.base import AgentInfo, register_agent, get_agent, get_all_agents, clear_registry
from tools.base import register_tool, get_tool, get_all_tools, clear_registry as clear_tool_registry


class TestAgentRegistry:
    """Tests for agent metadata registry."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def test_register_and_get_agent(self):
        """Test registering and retrieving agent metadata."""
        info = register_agent(
            name="test_agent",
            description="A test agent",
            custom_field="custom_value",
        )

        assert info.name == "test_agent"
        assert info.description == "A test agent"
        assert info.metadata["custom_field"] == "custom_value"

        retrieved = get_agent("test_agent")
        assert retrieved is info

    def test_get_nonexistent_agent(self):
        """Test getting a non-existent agent returns None."""
        result = get_agent("nonexistent_agent")
        assert result is None

    def test_get_all_agents(self):
        """Test getting all registered agents."""
        register_agent("agent1", "First agent")
        register_agent("agent2", "Second agent")

        agents = get_all_agents()
        assert len(agents) == 2
        names = [a.name for a in agents]
        assert "agent1" in names
        assert "agent2" in names


class TestToolRegistry:
    """Tests for tool registry."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_tool_registry()

    def test_register_tool_decorator(self):
        """Test registering a tool with decorator."""
        @register_tool(name="test_tool", tags=["test"])
        def my_tool(query: str) -> str:
            """Test tool description."""
            return f"Result: {query}"

        tool = get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert "test" in tool.metadata.get("tags", [])

    def test_register_tool_without_name(self):
        """Test tool name defaults to function name."""
        @register_tool()
        def another_tool(x: int) -> int:
            """Another tool."""
            return x * 2

        tool = get_tool("another_tool")
        assert tool is not None
        assert tool.name == "another_tool"

    def test_get_all_tools(self):
        """Test getting all registered tools."""
        @register_tool()
        def tool_a() -> str:
            """Tool A."""
            return "a"

        @register_tool()
        def tool_b() -> str:
            """Tool B."""
            return "b"

        tools = get_all_tools()
        assert len(tools) == 2


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_agent_info_creation(self):
        """Test creating AgentInfo."""
        info = AgentInfo(
            name="my_agent",
            description="My agent description",
            metadata={"version": "1.0"},
        )

        assert info.name == "my_agent"
        assert info.description == "My agent description"
        assert info.metadata["version"] == "1.0"

    def test_agent_info_default_metadata(self):
        """Test AgentInfo with default metadata."""
        info = AgentInfo(name="agent", description="desc")
        assert info.metadata == {}
