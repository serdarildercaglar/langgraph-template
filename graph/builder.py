"""
Graph builder utilities for LangGraph.
Thin wrapper around create_react_agent with Langfuse prompt integration.

Updated for LangGraph 1.0.x (January 2026):
- Pre/Post model hooks support for context management and guardrails
"""

from typing import Callable, Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from config.settings import get_settings
from llm import get_chat_model
from observability import get_agent_prompt


# Type alias for hook functions
HookFunction = Callable[[dict], dict | None]


async def get_checkpointer() -> BaseCheckpointSaver:
    """Get configured checkpointer based on settings."""
    settings = get_settings().checkpoint

    if settings.backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        return AsyncSqliteSaver.from_conn_string(settings.sqlite_path)
    elif settings.backend == "memory":
        return MemorySaver()
    else:
        raise ValueError(f"Unsupported checkpoint backend: {settings.backend}")


async def create_agent(
    tools: Sequence[BaseTool | Callable] | None = None,
    prompt: str | None = None,
    langfuse_prompt_name: str | None = None,
    model_name: str | None = None,
    checkpointer: BaseCheckpointSaver | bool = True,
    pre_model_hook: HookFunction | None = None,
    post_model_hook: HookFunction | None = None,
) -> CompiledStateGraph:
    """
    Create a ReAct agent with Langfuse prompt integration.

    This is a thin wrapper around LangGraph's create_react_agent that adds:
    - Langfuse prompt management
    - Configurable checkpointer from settings
    - Pre/Post model hooks for context management and guardrails (LangGraph 1.0+)

    Args:
        tools: Tools available to the agent
        prompt: System prompt (overrides Langfuse)
        langfuse_prompt_name: Fetch prompt from Langfuse by this name
        model_name: Override the default model
        checkpointer: True=use settings, False=none, or pass custom
        pre_model_hook: Hook called before model invocation (LangGraph 1.0+)
            - Use for context management, message summarization
            - See: create_message_summarization_hook()
        post_model_hook: Hook called after model invocation (LangGraph 1.0+)
            - Use for guardrails, human-in-the-loop checks
            - See: create_guardrail_hook()

    Returns:
        Compiled ReAct agent

    Example:
        # Simple usage
        agent = await create_agent(
            tools=get_all_tools(),
            langfuse_prompt_name="main-assistant",
        )

        # With guardrails (LangGraph 1.0+)
        from guardrails import create_input_guardrail, create_output_guardrail

        agent = await create_agent(
            tools=get_all_tools(),
            pre_model_hook=create_input_guardrail([...]),
            post_model_hook=create_output_guardrail([...]),
        )
    """
    # Resolve prompt: explicit > Langfuse > default
    system_prompt = prompt
    if not system_prompt and langfuse_prompt_name:
        system_prompt = get_agent_prompt(langfuse_prompt_name)
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant."

    # Resolve checkpointer
    resolved_checkpointer = None
    if checkpointer is True:
        resolved_checkpointer = await get_checkpointer()
    elif checkpointer is not False:
        resolved_checkpointer = checkpointer

    # Build kwargs for create_react_agent
    agent_kwargs: dict = {
        "model": get_chat_model(model_name=model_name),
        "tools": list(tools) if tools else [],
        "prompt": system_prompt,
        "checkpointer": resolved_checkpointer,
    }

    # Add hooks if provided (LangGraph 1.0+ feature)
    if pre_model_hook is not None:
        agent_kwargs["pre_model_hook"] = pre_model_hook
    if post_model_hook is not None:
        agent_kwargs["post_model_hook"] = post_model_hook

    # Create agent using LangGraph native function
    return create_react_agent(**agent_kwargs)
