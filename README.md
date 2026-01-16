# LangGraph Modular Agentic Framework

Production-ready, modular framework for building agentic AI systems with LangGraph.

## Features

- **LangGraph Native**: Uses `create_react_agent` - no unnecessary abstractions
- **vLLM Integration**: OpenAI-compatible API for on-premise LLM deployment
- **Multimodal Support**: Text, image, audio, and video input via vLLM
- **Langfuse Observability**: Full tracing and prompt management
- **Milvus RAG**: Async batch processing for vector search
- **FastAPI Backend**: Production-grade async API with streaming support

## Architecture Philosophy

### Why Single Agent + Tools (Not Multi-Agent)?

This framework uses a **single ReAct agent with tools** instead of a multi-agent supervisor pattern. Here's why:

#### The Problem with Supervisor Pattern

```
┌─────────────┐
│  Supervisor │ ──▶ Parses text to decide routing
└──────┬──────┘     (fragile, error-prone)
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐ ┌─────┐
│Agent│ │Agent│
│  A  │ │  B  │
└─────┘ └─────┘
```

Problems:
- **Text parsing is fragile**: Supervisor must parse "AGENT_A" or "FINISH" from LLM output
- **Context clutter**: Sub-agents see supervisor's routing logic in their context
- **Unnecessary complexity**: Custom state management, edge routing, etc.
- **Debugging difficulty**: Hard to trace decisions across multiple agents

#### The LangGraph 2025 Recommendation: Tool-Based Handoff

```
┌──────────────────────────────┐
│      Single ReAct Agent      │
│  ┌────┐ ┌────┐ ┌──────────┐  │
│  │Tool│ │Tool│ │SubAgent  │  │
│  │ A  │ │ B  │ │ as Tool  │  │
│  └────┘ └────┘ └──────────┘  │
└──────────────────────────────┘
```

Benefits:
- **Tool calling is reliable**: LLMs are trained for structured tool calls
- **Clean context**: Each tool call is isolated
- **Simple debugging**: Clear tool invocation traces in Langfuse
- **Native LangGraph**: Uses `create_react_agent` directly

#### When You Actually Need Multi-Agent

If you truly need multiple specialized agents, use **tool-based handoff**:

```python
from langgraph.prebuilt import create_react_agent

# Create specialized agent as a tool
def call_research_agent(query: str) -> str:
    """Call the research specialist for in-depth analysis."""
    research_graph = create_react_agent(model, research_tools)
    result = research_graph.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

# Main agent uses sub-agent as a tool
main_agent = create_react_agent(
    model=model,
    tools=[call_research_agent, other_tools...],
)
```

This gives you multi-agent capability **without** the supervisor pattern's complexity.

## Quick Start

### 1. Installation

```bash
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Run the Server

```bash
python main.py
```

### 3. Test the API

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "session_id": "session-abc",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Project Structure

```
langgraph_framework/
├── graph/
│   ├── builder.py      # create_agent() - thin wrapper for Langfuse integration
│   └── state.py        # MessagesState re-export, RAGState for custom graphs
├── tools/
│   ├── base.py         # @register_tool decorator
│   └── builtin/        # Example tools (calculator, current_time)
├── agents/
│   └── base.py         # AgentInfo metadata (for /agents endpoint only)
├── retrieval/          # RAG: embeddings, Milvus, pipeline
├── llm/                # vLLM client
├── observability/      # Langfuse integration
├── api/                # FastAPI routes
└── config/             # Pydantic settings
```

## Creating Tools

Tools are the primary way to extend agent capabilities:

```python
from tools import register_tool

@register_tool(tags=["database"])
def search_documents(query: str, limit: int = 10) -> str:
    """Search the document database.

    Args:
        query: Search query
        limit: Max results
    """
    # Your implementation
    return f"Found {limit} results for: {query}"
```

Tools are automatically registered and available to the agent.

## Creating the Agent

### Simple Usage (Recommended)

```python
from graph import create_agent
from tools import get_all_tools

# Creates ReAct agent with all registered tools
agent = await create_agent(
    tools=get_all_tools(),
    langfuse_prompt_name="main-assistant",  # Fetches prompt from Langfuse
)

# Use it
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "session-123"}},
)
```

### Direct LangGraph Usage (No Wrapper)

```python
from langgraph.prebuilt import create_react_agent
from llm import get_chat_model
from tools import get_all_tools

# Direct LangGraph - no framework wrapper needed
agent = create_react_agent(
    model=get_chat_model(),
    tools=get_all_tools(),
    prompt="You are a helpful assistant.",
)
```

## Langfuse Integration

### Prompt Management

Prompts are fetched from Langfuse by name:

```python
agent = await create_agent(
    langfuse_prompt_name="main-assistant",  # Fetches from Langfuse
)
```

Create prompts in Langfuse dashboard with matching names.

### Tracing

All agent invocations are automatically traced:

```python
from observability import create_trace_handler

handler = create_trace_handler(
    session_id="session-123",
    user_id="user-456",
    trace_name="chat",
)

result = await agent.ainvoke(
    {"messages": messages},
    config={"callbacks": [handler]},
)
```

## RAG Integration

```python
from retrieval import get_rag_pipeline

rag = get_rag_pipeline()
await rag.initialize()

# Ingest
await rag.ingest_documents([
    {"id": "doc1", "content": "Document text..."},
])

# Retrieve
results = await rag.retrieve("search query", top_k=5)
```

### RAG as a Tool

```python
from tools import register_tool
from retrieval import get_rag_pipeline

@register_tool(tags=["rag"])
async def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    rag = get_rag_pipeline()
    results = await rag.retrieve(query, top_k=5)
    return rag.format_context(results)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Synchronous chat |
| `/api/v1/stream` | POST | Streaming chat (SSE) |
| `/api/v1/health` | GET | Health check |
| `/api/v1/agents` | GET | List agent metadata |
| `/api/v1/tools` | GET | List available tools |

### Request Format

```json
{
  "user_id": "user-123",
  "session_id": "session-abc",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

### Multimodal Request

```json
{
  "user_id": "user-123",
  "session_id": "session-abc",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

## Configuration

Key environment variables:

| Setting | Description | Default |
|---------|-------------|---------|
| `VLLM_BASE_URL` | vLLM API endpoint | `http://localhost:8000/v1` |
| `VLLM_MODEL_NAME` | Model to use | `default-model` |
| `LANGFUSE_ENABLED` | Enable observability | `true` |
| `MILVUS_HOST` | Milvus host | `localhost` |
| `CHECKPOINT_BACKEND` | Persistence backend | `sqlite` |

## Development

```bash
# Tests
pytest tests/ -v

# Linting
ruff check .

# Type checking
mypy .
```

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph create_react_agent](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
- [LangGraph Supervisor Pattern](https://github.com/langchain-ai/langgraph-supervisor-py) - "We now recommend using the supervisor pattern directly via tools rather than this library"
- [Langfuse Documentation](https://langfuse.com/docs)

## License

MIT License
