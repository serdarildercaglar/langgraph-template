"""
Built-in guardrail implementations.
Simple pattern-based checks. For production, integrate with LLM-based or specialized APIs.
"""

import re

from guardrails.base import GuardrailResult


# =============================================================================
# INPUT GUARDRAILS
# =============================================================================


def prompt_injection_check(content: str) -> GuardrailResult:
    """
    Check for common prompt injection patterns.
    For production: Use LLM-based classification or specialized services.
    """
    patterns = [
        r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"disregard\s+(your|the|all)\s+(rules?|instructions?)",
        r"you\s+are\s+now\s+",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"forget\s+(everything|all|your)",
        r"pretend\s+(you|to\s+be)",
        r"act\s+as\s+(if|a|an)",
        r"jailbreak",
        r"DAN\s+mode",
    ]

    content_lower = content.lower()
    for pattern in patterns:
        if re.search(pattern, content_lower):
            return GuardrailResult(
                passed=False,
                message="Potential prompt injection detected",
                triggered_rule="prompt_injection",
            )

    return GuardrailResult(passed=True)


def toxic_content_check(content: str) -> GuardrailResult:
    """
    Basic toxic content detection.
    For production: Use moderation APIs (OpenAI, Perspective API, etc.)
    """
    # Very basic - production should use proper moderation
    toxic_patterns = [
        r"\b(kill|murder|harm)\s+(yourself|someone|people)\b",
        r"\bhow\s+to\s+(make|build)\s+(a\s+)?(bomb|weapon|explosive)\b",
        r"\b(hate|attack)\s+(all\s+)?(jews|muslims|christians|blacks|whites)\b",
    ]

    content_lower = content.lower()
    for pattern in toxic_patterns:
        if re.search(pattern, content_lower):
            return GuardrailResult(
                passed=False,
                message="Potentially harmful content detected",
                triggered_rule="toxic_content",
            )

    return GuardrailResult(passed=True)


def pii_input_check(content: str) -> GuardrailResult:
    """
    Check if user is sharing PII that shouldn't be processed.
    For production: Use dedicated PII detection services.
    """
    patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),  # US SSN
        (r"\b\d{16}\b", "credit_card"),  # Credit card (basic)
        (r"\b[A-Z]{2}\d{6}[A-Z]?\b", "passport"),  # Passport (basic)
    ]

    for pattern, pii_type in patterns:
        if re.search(pattern, content):
            return GuardrailResult(
                passed=False,
                message=f"PII detected: {pii_type}",
                triggered_rule=f"pii_input_{pii_type}",
            )

    return GuardrailResult(passed=True)


# =============================================================================
# OUTPUT GUARDRAILS
# =============================================================================


def pii_output_check(content: str) -> GuardrailResult:
    """
    Prevent model from outputting PII.
    """
    patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b\d{16}\b", "credit_card"),
        (r"\bpassword\s*[:=]\s*\S+", "password"),
        (r"\bapi[_-]?key\s*[:=]\s*\S+", "api_key"),
        (r"\bsecret\s*[:=]\s*\S+", "secret"),
    ]

    content_lower = content.lower()
    for pattern, pii_type in patterns:
        if re.search(pattern, content_lower):
            return GuardrailResult(
                passed=False,
                message=f"Output contains sensitive data: {pii_type}",
                triggered_rule=f"pii_output_{pii_type}",
            )

    return GuardrailResult(passed=True)


def hallucination_check(content: str) -> GuardrailResult:
    """
    Basic hallucination indicators check.
    For production: Use factuality verification with RAG or external APIs.
    """
    # Check for fake citations/sources
    fake_patterns = [
        r"according\s+to\s+(?:a\s+)?(?:recent\s+)?study(?:\s+by)?\s+\[?\d+\]?",
        r"research\s+shows\s+that\s+\d+%",
        r"scientists\s+(?:have\s+)?discovered\s+that",
    ]

    # These are just indicators, not definitive
    # Real hallucination detection requires fact-checking
    for pattern in fake_patterns:
        if re.search(pattern, content.lower()):
            return GuardrailResult(
                passed=True,  # Just flag, don't block
                message="Potential unverified claim detected",
                triggered_rule="hallucination_indicator",
            )

    return GuardrailResult(passed=True)


def off_topic_check(content: str, allowed_topics: list[str] | None = None) -> GuardrailResult:
    """
    Check if response stays on allowed topics.
    This is a placeholder - real implementation needs semantic similarity.

    Args:
        content: The output content to check
        allowed_topics: List of allowed topic keywords (optional)
    """
    if not allowed_topics:
        return GuardrailResult(passed=True)

    # Very basic keyword check
    # Production: Use embeddings and semantic similarity
    content_lower = content.lower()
    has_relevant_topic = any(topic.lower() in content_lower for topic in allowed_topics)

    if not has_relevant_topic:
        return GuardrailResult(
            passed=False,
            message="Response appears off-topic",
            triggered_rule="off_topic",
        )

    return GuardrailResult(passed=True)
