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


# Type aliases for guardrail functions
InputGuardrail = Callable[[str], GuardrailResult]
OutputGuardrail = Callable[[str], GuardrailResult]


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

    Example:
        from guardrails import create_input_guardrail, prompt_injection_check

        agent = create_react_agent(
            model=model,
            tools=tools,
            pre_model_hook=create_input_guardrail([prompt_injection_check]),
        )
    """

    def hook(state: dict) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                break

        if not last_user_msg:
            return None

        content = (
            last_user_msg.content
            if isinstance(last_user_msg.content, str)
            else str(last_user_msg.content)
        )

        # Run all guardrails
        for guardrail in guardrails:
            result = guardrail(content)
            if not result.passed:
                # Replace with block message
                blocked_response = AIMessage(content=block_message)
                return {"messages": messages + [blocked_response]}

        return None  # All checks passed

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

    Example:
        from guardrails import create_output_guardrail, pii_output_check

        agent = create_react_agent(
            model=model,
            tools=tools,
            post_model_hook=create_output_guardrail([pii_output_check]),
        )
    """

    def hook(state: dict) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage):
            return None

        content = (
            last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        )

        # Run all guardrails
        for guardrail in guardrails:
            result = guardrail(content)
            if not result.passed:
                # Replace last message with block message
                messages[-1] = AIMessage(content=block_message)
                return {"messages": messages}

        return None  # All checks passed

    return hook


def chain_guardrails(*hooks: Callable[[dict], dict | None]) -> Callable[[dict], dict | None]:
    """
    Chain multiple hooks together.

    Example:
        combined_hook = chain_guardrails(
            create_input_guardrail([prompt_injection_check]),
            create_message_summarization_hook(max_messages=20),
        )
    """

    def chained_hook(state: dict) -> dict | None:
        current_state = state
        for hook in hooks:
            result = hook(current_state)
            if result is not None:
                # Update state with hook result
                current_state = {**current_state, **result}
        # Return None if no changes, otherwise return modified state
        return None if current_state is state else current_state

    return chained_hook
