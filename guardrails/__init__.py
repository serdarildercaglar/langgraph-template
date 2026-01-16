"""
Guardrails module for input/output validation.
Uses LangGraph 1.0 native pre_model_hook and post_model_hook.
"""

from guardrails.base import (
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    create_input_guardrail,
    create_output_guardrail,
    chain_guardrails,
)
from guardrails.builtin import (
    # Input guardrails
    prompt_injection_check,
    toxic_content_check,
    pii_input_check,
    # Output guardrails
    pii_output_check,
)

__all__ = [
    # Core
    "GuardrailResult",
    "InputGuardrail",
    "OutputGuardrail",
    "create_input_guardrail",
    "create_output_guardrail",
    "chain_guardrails",
    # Built-in input
    "prompt_injection_check",
    "toxic_content_check",
    "pii_input_check",
    # Built-in output
    "pii_output_check",
]
