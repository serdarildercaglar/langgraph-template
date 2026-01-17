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


async def create_agent(
    tools: Sequence[BaseTool | Callable] | None = None,
    prompt: str | None = None,
    langfuse_prompt_name: str | None = None,
    model_name: str | None = None,
    checkpointer: BaseCheckpointSaver | bool = True,
    pre_model_hook: Callable[[dict], dict | None] | None = None,
    post_model_hook: Callable[[dict], dict | None] | None = None,
) -> CompiledStateGraph:
    """
    Create a ReAct agent with Langfuse prompt integration.

    Args:
        tools: Tools available to the agent
        prompt: System prompt (overrides Langfuse)
        langfuse_prompt_name: Fetch prompt from Langfuse by this name
        model_name: Override the default model
        checkpointer: True=use settings, False=none, or pass custom
        pre_model_hook: Hook called before model invocation (context management)
        post_model_hook: Hook called after model invocation (guardrails)

    Returns:
        Compiled ReAct agent

    Example:
        agent = await create_agent(
            tools=get_all_tools(),
            langfuse_prompt_name="main-assistant",
        )
    """
    # Resolve prompt: explicit > Langfuse > default
    if prompt:
        system_prompt = prompt
    elif langfuse_prompt_name:
        system_prompt = get_agent_prompt(langfuse_prompt_name)
    else:
        system_prompt = get_settings().agent.default_prompt

    # Resolve checkpointer inline (no separate function needed)
    resolved_checkpointer: BaseCheckpointSaver | None = None
    if checkpointer is True:
        settings = get_settings().checkpoint
        if settings.backend == "sqlite":
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            resolved_checkpointer = AsyncSqliteSaver.from_conn_string(settings.sqlite_path)
        elif settings.backend == "memory":
            resolved_checkpointer = MemorySaver()
    elif isinstance(checkpointer, BaseCheckpointSaver):
        resolved_checkpointer = checkpointer

    # Build and return agent
    return create_react_agent(
        model=get_chat_model(model_name),
        tools=list(tools) if tools else [],
        prompt=system_prompt,
        checkpointer=resolved_checkpointer,
        **({"pre_model_hook": pre_model_hook} if pre_model_hook else {}),
        **({"post_model_hook": post_model_hook} if post_model_hook else {}),
    )
