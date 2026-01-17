"""
Base guardrail types and utilities.
Designed to work with LangGraph 1.0 pre_model_hook and post_model_hook.
"""

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    message: str | None = None
    triggered_rule: str | None = None


# Type aliases
InputGuardrail = Callable[[str], GuardrailResult]
OutputGuardrail = Callable[[str], GuardrailResult]


def _get_content_as_str(msg) -> str:
    """Extract content as string from a message."""
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def create_input_guardrail(
    guardrails: list[InputGuardrail],
    block_message: str = "I cannot process this request.",
) -> Callable[[dict], dict | None]:
    """
    Create a pre_model_hook from input guardrails.

    Args:
        guardrails: List of input guardrail functions
        block_message: Message to return when blocked

    Returns:
        Hook function for use with create_react_agent's pre_model_hook
    """
    def hook(state: dict) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get last user message using next()
        last_user_msg = next(
            (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            None
        )
        if not last_user_msg:
            return None

        content = _get_content_as_str(last_user_msg)

        # Run guardrails - return on first failure
        for guardrail in guardrails:
            if not guardrail(content).passed:
                return {"messages": messages + [AIMessage(content=block_message)]}

        return None

    return hook


def create_output_guardrail(
    guardrails: list[OutputGuardrail],
    block_message: str = "I cannot provide this response.",
) -> Callable[[dict], dict | None]:
    """
    Create a post_model_hook from output guardrails.

    Args:
        guardrails: List of output guardrail functions
        block_message: Message to return when blocked

    Returns:
        Hook function for use with create_react_agent's post_model_hook
    """
    def hook(state: dict) -> dict | None:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        content = _get_content_as_str(messages[-1])

        # Run guardrails - replace message on first failure
        for guardrail in guardrails:
            if not guardrail(content).passed:
                return {"messages": messages[:-1] + [AIMessage(content=block_message)]}

        return None

    return hook


def chain_hooks(*hooks: Callable[[dict], dict | None]) -> Callable[[dict], dict | None]:
    """Chain multiple hooks together."""
    def chained(state: dict) -> dict | None:
        current = state
        modified = False
        for hook in hooks:
            result = hook(current)
            if result is not None:
                current = {**current, **result}
                modified = True
        return current if modified else None
    return chained
