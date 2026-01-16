"""
Example built-in tools.
Demonstrates tool creation patterns.
"""

from datetime import datetime

from tools.base import register_tool


@register_tool(
    name="calculator",
    description="Perform mathematical calculations. Supports basic arithmetic operations.",
    tags=["math", "utility"],
)
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    
    Returns:
        Result of the calculation
    """
    # Safe evaluation of mathematical expressions
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression. Only numbers and basic operators allowed."
    
    try:
        # Use eval with restricted builtins for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@register_tool(
    name="current_time",
    description="Get the current date and time.",
    tags=["utility", "datetime"],
)
def current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current datetime in ISO format
    """
    now = datetime.now()
    return f"Current time: {now.isoformat()}"


@register_tool(
    name="echo",
    description="Echo back the input message. Useful for testing.",
    tags=["utility", "test"],
)
def echo(message: str) -> str:
    """
    Echo back the provided message.
    
    Args:
        message: Message to echo
    
    Returns:
        The same message
    """
    return f"Echo: {message}"
