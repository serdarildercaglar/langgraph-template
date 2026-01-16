"""
vLLM OpenAI-compatible client wrapper.
Provides async-first LLM client with retry logic and streaming support.
"""

from typing import Any, AsyncIterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import VLLMSettings, get_settings


class VLLMClient:
    """
    Wrapper around ChatOpenAI configured for vLLM.
    Provides consistent interface with retry logic.
    """
    
    def __init__(
        self,
        settings: VLLMSettings | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self._settings = settings or get_settings().vllm
        self._model_name = model_name or self._settings.model_name
        self._temperature = temperature if temperature is not None else self._settings.temperature
        self._max_tokens = max_tokens or self._settings.max_tokens
        
        self._client: ChatOpenAI | None = None
    
    @property
    def client(self) -> ChatOpenAI:
        """Lazy initialization of the ChatOpenAI client."""
        if self._client is None:
            self._client = ChatOpenAI(
                base_url=self._settings.base_url,
                api_key=self._settings.api_key.get_secret_value(),
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._settings.timeout,
                max_retries=self._settings.max_retries,
            )
        return self._client
    
    @property
    def model(self) -> BaseChatModel:
        """Return the underlying LangChain model for use in graphs."""
        return self.client
    
    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke with retry logic."""
        result = await self.client.ainvoke(messages, **kwargs)
        return result  # type: ignore
    
    async def astream(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming with token-level output."""
        async for chunk in self.client.astream(messages, **kwargs):
            if chunk.content:
                yield chunk.content  # type: ignore
    
    async def astream_events(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream events for detailed observability.
        Yields events like: on_llm_start, on_llm_new_token, on_llm_end
        """
        async for event in self.client.astream_events(
            messages,
            version="v2",
            **kwargs,
        ):
            yield event


def create_vllm_client(
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> VLLMClient:
    """Factory function for creating vLLM clients."""
    return VLLMClient(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_chat_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance configured for vLLM.
    Use this when you need direct LangChain model access.
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
