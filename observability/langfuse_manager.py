"""
Langfuse integration for observability and prompt management.
Provides callback handler and prompt fetching from Langfuse.
"""

import os
from functools import lru_cache
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from config.settings import LangfuseSettings, get_settings


class LangfuseManager:
    """
    Manages Langfuse client and callback handlers.
    Handles prompt management and tracing configuration.
    """
    
    def __init__(self, settings: LangfuseSettings | None = None):
        self._settings = settings or get_settings().langfuse
        self._client: Langfuse | None = None
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Set environment variables for Langfuse SDK."""
        if self._settings.enabled:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self._settings.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self._settings.secret_key.get_secret_value()
            os.environ["LANGFUSE_HOST"] = self._settings.host
    
    @property
    def enabled(self) -> bool:
        return self._settings.enabled and bool(self._settings.public_key)
    
    @property
    def client(self) -> Langfuse | None:
        """Lazy initialization of Langfuse client."""
        if not self.enabled:
            return None
        
        if self._client is None:
            self._client = Langfuse(
                public_key=self._settings.public_key,
                secret_key=self._settings.secret_key.get_secret_value(),
                host=self._settings.host,
            )
        return self._client
    
    def create_handler(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        trace_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> CallbackHandler | None:
        """
        Create a Langfuse callback handler for tracing.
        Returns None if Langfuse is disabled.
        """
        if not self.enabled:
            return None
        
        return CallbackHandler(
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name,
            metadata=metadata,
            tags=tags,
        )
    
    def get_prompt(
        self,
        name: str,
        version: int | None = None,
        label: str | None = None,
    ) -> str | None:
        """
        Fetch a prompt from Langfuse prompt management.
        
        Args:
            name: Prompt name in Langfuse
            version: Specific version (optional)
            label: Label like 'production', 'staging' (optional)
        
        Returns:
            Prompt text or None if not found/disabled
        """
        if not self.client:
            return None
        
        try:
            prompt = self.client.get_prompt(
                name=name,
                version=version,
                label=label,
            )
            return prompt.prompt if hasattr(prompt, 'prompt') else str(prompt)
        except Exception:
            return None
    
    def get_prompt_with_variables(
        self,
        name: str,
        variables: dict[str, Any],
        version: int | None = None,
        label: str | None = None,
    ) -> str | None:
        """
        Fetch and compile a prompt with variables.
        
        Args:
            name: Prompt name in Langfuse
            variables: Variables to interpolate
            version: Specific version (optional)
            label: Label like 'production', 'staging' (optional)
        
        Returns:
            Compiled prompt text or None
        """
        if not self.client:
            return None
        
        try:
            prompt = self.client.get_prompt(
                name=name,
                version=version,
                label=label,
            )
            return prompt.compile(**variables)
        except Exception:
            return None
    
    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if self._client:
            self._client.flush()


# Module-level singleton
_manager: LangfuseManager | None = None


def get_langfuse_manager() -> LangfuseManager:
    """Get or create the Langfuse manager singleton."""
    global _manager
    if _manager is None:
        _manager = LangfuseManager()
    return _manager


def create_trace_handler(
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CallbackHandler | None:
    """
    Convenience function to create a trace handler.
    Returns None if Langfuse is disabled.
    """
    return get_langfuse_manager().create_handler(
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        metadata=metadata,
    )


def get_agent_prompt(
    agent_name: str,
    label: str | None = None,
) -> str | None:
    """
    Fetch an agent's system prompt from Langfuse.
    Convention: prompts are named 'agent-{agent_name}'
    """
    return get_langfuse_manager().get_prompt(
        name=f"agent-{agent_name}",
        label=label or ("production" if get_settings().is_production else "development"),
    )
