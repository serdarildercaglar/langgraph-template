"""
API Request/Response models.
vLLM multimodal compatible, OpenAI-style message format.
"""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# Content types for multimodal messages
class TextContent(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = Field(..., min_length=1)


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""
    url: str = Field(..., description="URL or base64 data URI (data:image/jpeg;base64,...)")
    detail: Literal["auto", "low", "high"] = Field(default="auto")


class ImageContent(BaseModel):
    """Image content block."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class AudioUrl(BaseModel):
    """Audio URL."""
    url: str = Field(..., description="URL or base64 data URI (data:audio/wav;base64,...)")


class AudioContent(BaseModel):
    """Audio content block."""
    type: Literal["audio_url"] = "audio_url"
    audio_url: AudioUrl


class VideoUrl(BaseModel):
    """Video URL."""
    url: str = Field(..., description="URL or base64 data URI (data:video/mp4;base64,...)")


class VideoContent(BaseModel):
    """Video content block."""
    type: Literal["video_url"] = "video_url"
    video_url: VideoUrl


# Union type for all content types
ContentPart = TextContent | ImageContent | AudioContent | VideoContent


class Message(BaseModel):
    """
    Single message in conversation.
    Supports multimodal content (text, image, audio, video).
    """
    role: Literal["user", "assistant", "system"] = Field(...)
    content: str | list[ContentPart] = Field(
        ...,
        description="Text string or array of content parts for multimodal"
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> str | list[ContentPart]:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if not v:
                raise ValueError("Content list cannot be empty")
            return v
        raise ValueError("Content must be string or list of content parts")


class ChatRequest(BaseModel):
    """
    Chat request payload.
    Supports multimodal input via vLLM-compatible format.
    """
    user_id: str = Field(..., min_length=1, description="User identifier")
    session_id: str = Field(..., min_length=1, description="Session/conversation identifier")
    messages: list[Message] = Field(..., min_length=1, description="Conversation messages")
    agent_name: str | None = Field(default=None, description="Specific agent to route to")
    stream: bool = Field(default=False, description="Enable streaming response")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("At least one message is required")
        return v


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ToolCallInfo(BaseModel):
    """Information about a tool call."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    status: Literal["pending", "running", "completed", "failed"] = "completed"


class ChatResponse(BaseModel):
    """
    Chat response payload.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    agent_name: str | None = None
    content: str = Field(..., description="Response content")
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    """
    Server-Sent Event for streaming responses.
    """
    event: Literal["message_start", "content_delta", "tool_start", "tool_end", "message_end", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class ErrorResponse(BaseModel):
    """Error response payload."""
    error: str
    code: str = Field(default="internal_error")
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# =============================================================================
# RAG / Retrieve Models
# =============================================================================


class SearchRequest(BaseModel):
    """Single collection search request."""
    collection: str = Field(..., description="Collection name")
    queries: list[str] = Field(..., min_length=1, description="Search queries")
    dense_fields: list[str] = Field(default_factory=list, description="Dense vector fields")
    fts_fields: list[str] = Field(default_factory=list, description="FTS/BM25 sparse fields")
    top_k: int = Field(default=5, ge=1, le=100, description="Results per query")
    filter: str | None = Field(default=None, description="Pre-filter expression")
    rerank: Literal["rrf", "weighted"] = Field(default="rrf", description="Reranking strategy")
    weights: list[float] | None = Field(default=None, description="Weights for weighted rerank")

    def model_post_init(self, __context: Any) -> None:
        if not self.dense_fields and not self.fts_fields:
            raise ValueError("At least one of dense_fields or fts_fields required")


class RetrieveRequest(BaseModel):
    """
    Retrieve request payload. Supports single/multi collection and single/multi query.

    Examples:
        # Single collection, single query
        {
            "requests": [{
                "collection": "docs",
                "queries": ["how to authenticate"],
                "dense_fields": ["content_vector"]
            }]
        }

        # Single collection, multiple queries (batch)
        {
            "requests": [{
                "collection": "docs",
                "queries": ["OAuth setup", "JWT token", "API key"],
                "dense_fields": ["content_vector"],
                "fts_fields": ["content_sparse"],
                "filter": "category == 'api'"
            }]
        }

        # Multiple collections (parallel search)
        {
            "requests": [
                {
                    "collection": "docs",
                    "queries": ["authentication"],
                    "dense_fields": ["content_vector"]
                },
                {
                    "collection": "faq",
                    "queries": ["authentication"],
                    "dense_fields": ["question_vector"]
                }
            ]
        }
    """
    requests: list[SearchRequest] = Field(..., min_length=1)


class Hit(BaseModel):
    """Single search hit."""
    id: str
    score: float
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Results for a single query."""
    query: str
    hits: list[Hit]


class CollectionResult(BaseModel):
    """Results for a single collection."""
    collection: str
    results: list[QueryResult]
    search_type: Literal["dense", "fts", "hybrid"]


class RetrieveResponse(BaseModel):
    """Retrieve response payload."""
    collections: list[CollectionResult]
    total_hits: int
    latency_ms: float
