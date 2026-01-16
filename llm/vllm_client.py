"""
vLLM OpenAI-compatible client.
Provides ChatOpenAI configured for vLLM endpoint.
"""

from langchain_openai import ChatOpenAI

from config.settings import get_settings


def get_chat_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance configured for vLLM.

    Creates a LangChain ChatOpenAI client pointing to vLLM's OpenAI-compatible
    /v1/chat/completions endpoint. Uses settings from environment/config.

    Args:
        model_name: Override the default model name from settings.
        temperature: Override the default temperature (0.0-2.0) from settings.

    Returns:
        ChatOpenAI instance configured for vLLM endpoint.

    Example:
        model = get_chat_model()
        response = await model.ainvoke([HumanMessage(content="Hello")])

        # With overrides
        model = get_chat_model(model_name="qwen2.5-72b", temperature=0.0)
    """
    settings = get_settings().vllm
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),
        model=model_name or settings.model_name,
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
