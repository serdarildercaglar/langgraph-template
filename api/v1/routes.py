"""
API v1 routes.
Provides chat and stream endpoints with multimodal support.
"""

from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.models import (
    ChatRequest,
    ChatResponse,
    StreamEvent,
    TokenUsage,
    ToolCallInfo,
    ContentPart,
    TextContent,
    ImageContent,
    AudioContent,
    VideoContent,
)
from graph import create_agent
from guardrails import (
    create_input_guardrail,
    create_output_guardrail,
    prompt_injection_check,
    toxic_content_check,
    pii_output_check,
)
from observability import create_trace_handler
from tools import get_all_tools

router = APIRouter(prefix="/api/v1", tags=["v1"])

# Graph singleton
_compiled_graph = None


def _convert_content(content: str | list[ContentPart]) -> str | list[dict[str, Any]]:
    """Convert API content format to LangChain message content format."""
    if isinstance(content, str):
        return content

    result = []
    for part in content:
        if isinstance(part, TextContent):
            result.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            result.append({
                "type": "image_url",
                "image_url": {"url": part.image_url.url, "detail": part.image_url.detail}
            })
        elif isinstance(part, AudioContent):
            result.append({"type": "audio_url", "audio_url": {"url": part.audio_url.url}})
        elif isinstance(part, VideoContent):
            result.append({"type": "video_url", "video_url": {"url": part.video_url.url}})
    return result


def _to_langchain_messages(messages: list) -> list:
    """Convert API messages to LangChain message objects."""
    msg_map = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
    return [msg_map[m.role](content=_convert_content(m.content)) for m in messages]


def _create_invoke_config(request: ChatRequest) -> dict:
    """Create config dict for graph invocation."""
    handler = create_trace_handler(
        session_id=request.session_id,
        user_id=request.user_id,
        trace_name="chat",
        metadata=request.metadata,
    )
    config = {"configurable": {"thread_id": request.session_id}}
    if handler:
        config["callbacks"] = [handler]
    return config


async def get_graph():
    """Get or create the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = await create_agent(
            tools=get_all_tools(),
            langfuse_prompt_name="main-assistant",
            pre_model_hook=create_input_guardrail([prompt_injection_check, toxic_content_check]),
            post_model_hook=create_output_guardrail([pii_output_check]),
        )
    return _compiled_graph


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Synchronous chat endpoint with multimodal support."""
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /stream endpoint for streaming or set stream=false"
        )

    graph = await get_graph()
    config = _create_invoke_config(request)

    try:
        result = await graph.ainvoke(
            {"messages": _to_langchain_messages(request.messages)},
            config=config
        )

        # Get last AI message
        last_ai = next(
            (m for m in reversed(result.get("messages", [])) if isinstance(m, AIMessage)),
            None
        )

        tool_calls = []
        if last_ai and getattr(last_ai, "tool_calls", None):
            tool_calls = [
                ToolCallInfo(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("args", {}),
                )
                for tc in last_ai.tool_calls
            ]

        return ChatResponse(
            session_id=request.session_id,
            content=last_ai.content if last_ai else "",
            tool_calls=tool_calls,
            usage=TokenUsage(),
            metadata={"user_id": request.user_id},
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/stream")
async def stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint (SSE) with multimodal support."""

    async def event_generator() -> AsyncIterator[str]:
        graph = await get_graph()
        config = _create_invoke_config(request)

        try:
            yield f"data: {StreamEvent(event='message_start', data={'session_id': request.session_id}).model_dump_json()}\n\n"

            async for event in graph.astream_events(
                {"messages": _to_langchain_messages(request.messages)},
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and getattr(chunk, "content", None):
                        yield f"data: {StreamEvent(event='content_delta', data={'content': chunk.content}).model_dump_json()}\n\n"

                elif event_type == "on_tool_start":
                    yield f"data: {StreamEvent(event='tool_start', data={'tool': event.get('name', 'unknown'), 'input': event.get('data', {}).get('input', {})}).model_dump_json()}\n\n"

                elif event_type == "on_tool_end":
                    yield f"data: {StreamEvent(event='tool_end', data={'tool': event.get('name', 'unknown'), 'output': str(event.get('data', {}).get('output', ''))}).model_dump_json()}\n\n"

            yield f"data: {StreamEvent(event='message_end', data={'session_id': request.session_id}).model_dump_json()}\n\n"

        except Exception as e:
            yield f"data: {StreamEvent(event='error', data={'error': str(e)}).model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "v1"}


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List all available agent configurations."""
    from agents.base import get_all_agents
    return {"agents": [{"name": a.name, "description": a.description} for a in get_all_agents()]}


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    """List all available tools."""
    from tools.base import get_all_tools
    return {"tools": [{"name": t.name, "description": t.description} for t in get_all_tools()]}
