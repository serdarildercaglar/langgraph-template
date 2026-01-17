"""
Built-in guardrail implementations.
Simple pattern-based checks. For production, integrate with LLM-based or specialized APIs.
"""

import re
from typing import Callable

from guardrails.base import GuardrailResult


def create_pattern_guardrail(
    patterns: list[tuple[str, str]],
    message: str,
    rule_prefix: str,
) -> Callable[[str], GuardrailResult]:
    """
    Factory to create pattern-based guardrails.

    Args:
        patterns: List of (regex_pattern, rule_name) tuples
        message: Message to return when pattern matches
        rule_prefix: Prefix for triggered_rule field

    Returns:
        Guardrail function
    """
    def check(content: str) -> GuardrailResult:
        content_lower = content.lower()
        for pattern, rule_name in patterns:
            if re.search(pattern, content_lower):
                return GuardrailResult(
                    passed=False,
                    message=message,
                    triggered_rule=f"{rule_prefix}_{rule_name}",
                )
        return GuardrailResult(passed=True)
    return check


# =============================================================================
# INPUT GUARDRAILS
# =============================================================================

prompt_injection_check = create_pattern_guardrail(
    patterns=[
        (r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)", "ignore_instructions"),
        (r"disregard\s+(your|the|all)\s+(rules?|instructions?)", "disregard_rules"),
        (r"you\s+are\s+now\s+", "role_override"),
        (r"new\s+instructions?:", "new_instructions"),
        (r"system\s*:\s*", "system_prefix"),
        (r"<\s*system\s*>", "system_tag"),
        (r"forget\s+(everything|all|your)", "forget_command"),
        (r"pretend\s+(you|to\s+be)", "pretend_command"),
        (r"act\s+as\s+(if|a|an)", "act_as"),
        (r"jailbreak", "jailbreak"),
        (r"DAN\s+mode", "dan_mode"),
    ],
    message="Potential prompt injection detected",
    rule_prefix="prompt_injection",
)
prompt_injection_check.__doc__ = """
Check for common prompt injection patterns.
For production: Use LLM-based classification or specialized services.
"""

toxic_content_check = create_pattern_guardrail(
    patterns=[
        (r"\b(kill|murder|harm)\s+(yourself|someone|people)\b", "violence"),
        (r"\bhow\s+to\s+(make|build)\s+(a\s+)?(bomb|weapon|explosive)\b", "weapons"),
        (r"\b(hate|attack)\s+(all\s+)?(jews|muslims|christians|blacks|whites)\b", "hate_speech"),
    ],
    message="Potentially harmful content detected",
    rule_prefix="toxic",
)
toxic_content_check.__doc__ = """
Basic toxic content detection.
For production: Use moderation APIs (OpenAI, Perspective API, etc.)
"""

pii_input_check = create_pattern_guardrail(
    patterns=[
        (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
        (r"\b\d{16}\b", "credit_card"),
        (r"\b[A-Z]{2}\d{6}[A-Z]?\b", "passport"),
    ],
    message="PII detected in input",
    rule_prefix="pii_input",
)
pii_input_check.__doc__ = """
Check if user is sharing PII that shouldn't be processed.
For production: Use dedicated PII detection services.
"""


# =============================================================================
# OUTPUT GUARDRAILS
# =============================================================================

pii_output_check = create_pattern_guardrail(
    patterns=[
        (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
        (r"\b\d{16}\b", "credit_card"),
        (r"\bpassword\s*[:=]\s*\S+", "password"),
        (r"\bapi[_-]?key\s*[:=]\s*\S+", "api_key"),
        (r"\bsecret\s*[:=]\s*\S+", "secret"),
    ],
    message="Output contains sensitive data",
    rule_prefix="pii_output",
)
pii_output_check.__doc__ = """Prevent model from outputting PII."""
