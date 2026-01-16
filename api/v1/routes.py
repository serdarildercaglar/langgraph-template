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
from observability import create_trace_handler

router = APIRouter(prefix="/api/v1", tags=["v1"])


def convert_content_to_langchain(content: str | list[ContentPart]) -> str | list[dict[str, Any]]:
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
            result.append({
                "type": "audio_url",
                "audio_url": {"url": part.audio_url.url}
            })
        elif isinstance(part, VideoContent):
            result.append({
                "type": "video_url",
                "video_url": {"url": part.video_url.url}
            })
    return result


def convert_messages_to_langchain(messages: list) -> list:
    """Convert API messages to LangChain message objects."""
    result = []
    for msg in messages:
        content = convert_content_to_langchain(msg.content)
        if msg.role == "user":
            result.append(HumanMessage(content=content))
        elif msg.role == "assistant":
            result.append(AIMessage(content=content))
        elif msg.role == "system":
            result.append(SystemMessage(content=content))
    return result


# Graph singleton
_compiled_graph = None


async def get_graph():
    """Get or create the compiled graph."""
    global _compiled_graph

    if _compiled_graph is None:
        from graph import create_agent
        from tools import get_all_tools
        from guardrails import (
            create_input_guardrail,
            create_output_guardrail,
            prompt_injection_check,
            toxic_content_check,
            pii_output_check,
        )

        _compiled_graph = await create_agent(
            tools=get_all_tools(),
            langfuse_prompt_name="main-assistant",
            pre_model_hook=create_input_guardrail([
                prompt_injection_check,
                toxic_content_check,
            ]),
            post_model_hook=create_output_guardrail([
                pii_output_check,
            ]),
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

    langfuse_handler = create_trace_handler(
        session_id=request.session_id,
        user_id=request.user_id,
        trace_name="chat",
        metadata=request.metadata,
    )

    config = {"configurable": {"thread_id": request.session_id}}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    try:
        lc_messages = convert_messages_to_langchain(request.messages)
        result = await graph.ainvoke({"messages": lc_messages}, config=config)

        messages = result.get("messages", [])
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break

        content = last_ai_message.content if last_ai_message else ""

        tool_calls = []
        if last_ai_message and hasattr(last_ai_message, "tool_calls") and last_ai_message.tool_calls:
            for tc in last_ai_message.tool_calls:
                tool_calls.append(ToolCallInfo(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("args", {}),
                ))

        return ChatResponse(
            session_id=request.session_id,
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(),
            metadata={"user_id": request.user_id},
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/stream")
async def stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint (SSE) with multimodal support."""

    async def event_generator() -> AsyncIterator[str]:
        graph = await get_graph()

        langfuse_handler = create_trace_handler(
            session_id=request.session_id,
            user_id=request.user_id,
            trace_name="stream",
            metadata=request.metadata,
        )

        config = {"configurable": {"thread_id": request.session_id}}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        try:
            lc_messages = convert_messages_to_langchain(request.messages)

            start_event = StreamEvent(
                event="message_start",
                data={"session_id": request.session_id},
            )
            yield f"data: {start_event.model_dump_json()}\n\n"

            async for event in graph.astream_events(
                {"messages": lc_messages},
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        stream_event = StreamEvent(
                            event="content_delta",
                            data={"content": chunk.content},
                        )
                        yield f"data: {stream_event.model_dump_json()}\n\n"

                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    stream_event = StreamEvent(
                        event="tool_start",
                        data={"tool": tool_name, "input": tool_input},
                    )
                    yield f"data: {stream_event.model_dump_json()}\n\n"

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    tool_output = event.get("data", {}).get("output", "")
                    stream_event = StreamEvent(
                        event="tool_end",
                        data={"tool": tool_name, "output": str(tool_output)},
                    )
                    yield f"data: {stream_event.model_dump_json()}\n\n"

            end_event = StreamEvent(
                event="message_end",
                data={"session_id": request.session_id},
            )
            yield f"data: {end_event.model_dump_json()}\n\n"

        except Exception as e:
            error_event = StreamEvent(
                event="error",
                data={"error": str(e)},
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "v1"}


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List all available agent configurations."""
    from agents.base import get_all_agents

    agents = get_all_agents()
    return {
        "agents": [
            {"name": a.name, "description": a.description}
            for a in agents
        ]
    }


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    """List all available tools."""
    from tools.base import get_all_tools

    tools = get_all_tools()
    return {
        "tools": [
            {"name": t.name, "description": t.description}
            for t in tools
        ]
    }
