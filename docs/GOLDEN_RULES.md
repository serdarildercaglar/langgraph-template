# 🏆 GOLDEN RULES - LangGraph Framework Code Review (2026)

Bu doküman, projenin code review sürecinde kullanılacak altın kuralları tanımlar. Tüm kurallar 2026 yılı best practices araştırmasına dayanmaktadır.

---

## İçindekiler

1. [LangGraph Native Yaklaşım](#1-langgraph-native-yaklaşım)
2. [FastAPI & Async Kuralları](#2-fastapi--async-kuralları)
3. [Python Asyncio Patterns](#3-python-asyncio-patterns)
4. [Pydantic v2 Kuralları](#4-pydantic-v2-kuralları)
5. [vLLM OpenAI-Compatible API](#5-vllm-openai-compatible-api)
6. [Milvus Vector Database](#6-milvus-vector-database)
7. [Tool Tasarımı](#7-tool-tasarımı)
8. [Güvenlik (Guardrails)](#8-güvenlik-guardrails)
9. [Observability](#9-observability)
10. [Code Style](#10-code-style)

---

## 1. LangGraph Native Yaklaşım

| Kural | Açıklama | Kaynak |
|-------|----------|--------|
| ✅ `create_react_agent` kullan | Custom ReAct loop yazma, native kullan | [LangGraph 1.0](https://www.blog.langchain.com/langchain-langgraph-1dot0/) |
| ✅ Tool-based handoff | Supervisor pattern yerine sub-agent'ları tool olarak tanımla | [LangGraph Design](https://www.blog.langchain.com/building-langgraph/) |
| ✅ `MessagesState` (TypedDict) | Pydantic state yerine TypedDict - her update'te validation yok | [State of Agent Engineering](https://www.langchain.com/state-of-agent-engineering) |
| ✅ Durable state + checkpointer | Production için persistence zorunlu (SQLite/Postgres) | [LangGraph 1.0](https://www.blog.langchain.com/langchain-langgraph-1dot0/) |
| ✅ Human-in-the-loop hooks | Pre/post model hooks ile güvenlik ve onay | [LangGraph 1.0](https://www.blog.langchain.com/langchain-langgraph-1dot0/) |
| ❌ Custom state reducers | `add_messages` reducer yeterli, custom yazma | Best Practice |

### Örnek: Doğru Agent Oluşturma

```python
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState

# ✅ Doğru
agent = create_react_agent(
    model=get_chat_model(),
    tools=get_all_tools(),
    prompt="You are a helpful assistant.",
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
)

# ❌ Yanlış - Custom ReAct loop
class CustomAgent:
    def process(self, state):
        response = self.model.invoke(messages)
        if response.tool_calls:
            # Manuel tool handling...
```

---

## 2. FastAPI & Async Kuralları

| Kural | Açıklama | Kaynak |
|-------|----------|--------|
| ✅ `async def` + `await` I/O için | Database, API çağrıları için async kullan | [FastAPI Async](https://fastapi.tiangolo.com/async/) |
| ✅ `def` blocking/CPU için | CPU-bound işlemleri sync fonksiyonda yap | [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices) |
| ❌ `async def` + blocking kod | Asla! Event loop'u bloklar | [FastAPI Async](https://fastapi.tiangolo.com/async/) |
| ✅ Async DB drivers | `asyncpg`, `aiomysql`, `databases` kullan | [FastAPI Production](https://render.com/articles/fastapi-production-deployment-best-practices) |
| ✅ Connection pooling | `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True` | [FastAPI Production](https://render.com/articles/fastapi-production-deployment-best-practices) |
| ✅ Background tasks | Uzun işler için `BackgroundTasks` veya Celery | [FastAPI Production](https://render.com/articles/fastapi-production-deployment-best-practices) |
| ✅ Uvicorn + Gunicorn | Dev: uvicorn, Prod: gunicorn + uvicorn workers | [FastAPI Production](https://render.com/articles/fastapi-production-deployment-best-practices) |

### Örnek: Async/Sync Kullanımı

```python
# ✅ Doğru - I/O bound işlem async
@app.post("/chat")
async def chat(request: ChatRequest):
    result = await agent.ainvoke({"messages": request.messages})
    return result

# ✅ Doğru - CPU bound işlem sync (threadpool'da çalışır)
@app.post("/process-image")
def process_image(image: UploadFile):
    # Heavy CPU computation
    result = expensive_cpu_operation(image)
    return result

# ❌ Yanlış - async fonksiyonda blocking çağrı
@app.post("/bad-example")
async def bad_example():
    result = requests.get("https://api.example.com")  # Blocking!
    return result
```

---

## 3. Python Asyncio Patterns

| Kural | Açıklama | Kaynak |
|-------|----------|--------|
| ✅ `asyncio.run()` kullan | `run_until_complete()` yerine modern API | [Python asyncio](https://docs.python.org/3/library/asyncio.html) |
| ✅ `asyncio.create_task()` + `gather()` | Paralel execution için | [Real Python](https://realpython.com/async-io-python/) |
| ✅ `Semaphore` ile concurrency limit | Kaynak aşımını önle | [Elastic Blog](https://www.elastic.co/blog/async-patterns-building-python-service) |
| ✅ CPU işlerini offload et | `ThreadPoolExecutor` veya `ProcessPoolExecutor` | [Python asyncio](https://docs.python.org/3/library/asyncio.html) |
| ✅ Graceful shutdown | Signal handling ile temiz kapanma | [Elastic Blog](https://www.elastic.co/blog/async-patterns-building-python-service) |
| ❌ `await coroutine` direkt | Task'a sarmadan await event loop'a dönmez | [Python asyncio](https://docs.python.org/3/library/asyncio.html) |

### Örnek: Paralel Execution

```python
import asyncio

# ✅ Doğru - Paralel execution
async def fetch_all_data():
    task1 = asyncio.create_task(fetch_users())
    task2 = asyncio.create_task(fetch_orders())
    task3 = asyncio.create_task(fetch_products())

    users, orders, products = await asyncio.gather(task1, task2, task3)
    return users, orders, products

# ✅ Doğru - Semaphore ile rate limiting
async def fetch_with_limit(urls: list[str], max_concurrent: int = 10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str):
        async with semaphore:
            return await http_client.get(url)

    tasks = [asyncio.create_task(fetch_one(url)) for url in urls]
    return await asyncio.gather(*tasks)

# ❌ Yanlış - Sequential execution (yavaş)
async def fetch_all_slow():
    users = await fetch_users()      # Bekle
    orders = await fetch_orders()    # Sonra bekle
    products = await fetch_products() # Sonra bekle
    return users, orders, products
```

---

## 4. Pydantic v2 Kuralları

| Kural | Açıklama | Kaynak |
|-------|----------|--------|
| ✅ API boundary'de Pydantic | Request/Response modelleri için | [Pydantic v2](https://pydantic.dev/articles/pydantic-v2) |
| ✅ TypedDict internal state için | Agent state için Pydantic overhead'i gereksiz | Best Practice |
| ✅ Declarative constraints | Python validator yerine `Field(ge=0, le=100)` | [Pydantic Performance](https://docs.pydantic.dev/latest/concepts/performance/) |
| ✅ `FailFast` sequences için | Sequence validasyonunda erken çık (v2.8+) | [Pydantic Performance](https://docs.pydantic.dev/latest/concepts/performance/) |
| ✅ TypeAdapter reuse | Her seferinde yeniden oluşturma (cold validation) | [Pydantic v2 at Scale](https://medium.com/@connect.hashblock/pydantic-v2-at-scale-7-tricks-for-2-faster-validation-9bd95bf27232) |
| ✅ `model_validator` kullan | `root_validator` deprecated | [Pydantic Migration](https://docs.pydantic.dev/latest/migration/) |

### Örnek: Pydantic Kullanımı

```python
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

# ✅ Doğru - API boundary'de Pydantic
class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    messages: list[dict]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @model_validator(mode='after')
    def validate_messages(self):
        if not self.messages:
            raise ValueError("Messages cannot be empty")
        return self

# ✅ Doğru - Internal state için TypedDict
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ❌ Yanlış - Internal state için Pydantic (gereksiz overhead)
class AgentStatePydantic(BaseModel):
    messages: list  # Her update'te validation çalışır
```

---

## 5. vLLM OpenAI-Compatible API

| Kural | Açıklama | Kaynak |
|-------|----------|--------|
| ✅ `/v1/chat/completions` kullan | `/v1/completions` legacy, kullanma | [vLLM Docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) |
| ✅ Tool calling için `--enable-auto-tool-choice` | Model desteğine göre parser seç | [vLLM Docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) |
| ✅ `gpu-memory-utilization=0.90` | Daha fazla batch için yüksek tut | [vLLM Quickstart](https://www.glukhov.org/post/2026/01/vllm-quickstart/) |
| ✅ Streaming SSE format | OpenAI uyumlu streaming | [vLLM Docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) |
| ⚠️ Token counting farklı | OpenAI ile aynı olmayabilir | [vLLM Docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) |

### Örnek: vLLM Client Yapılandırması

```python
from langchain_openai import ChatOpenAI

# ✅ Doğru - Chat completions endpoint
def get_chat_model():
    return ChatOpenAI(
        base_url=settings.vllm_base_url,  # http://localhost:8000/v1
        api_key=settings.vllm_api_key,
        model=settings.vllm_model_name,
        temperature=settings.vllm_temperature,
        max_tokens=settings.vllm_max_tokens,
        streaming=True,  # SSE streaming
    )

# vLLM server başlatma (önerilen parametreler)
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-72B-Instruct \
#     --gpu-memory-utilization 0.90 \
#     --max-model-len 8192 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes
```

---

## 6. Milvus Vector Database

| Kural | Açıklama | Kaynak |
|-------|----------|--------|
| ✅ `AsyncMilvusClient` kullan | SDK v2 native async (**pymilvus >= 2.5.3** gerekli) | [Milvus SDK v2](https://medium.com/vector-database/introducing-milvus-sdk-v2-native-async-support-unified-apis-and-superior-performance-388c3eb6fa2d) |
| ✅ `asyncio.gather()` ile batch | Paralel insert/query | [Milvus SDK v2](https://medium.com/vector-database/introducing-milvus-sdk-v2-native-async-support-unified-apis-and-superior-performance-388c3eb6fa2d) |
| ✅ HNSW index | High recall, low latency | [Milvus Best Practices](https://milvus.io/ai-quick-reference/what-are-vector-database-best-practices) |
| ✅ Metadata filtering | Search öncesi filtrele, search space küçült | [Milvus Best Practices](https://milvus.io/ai-quick-reference/what-are-vector-database-best-practices) |
| ✅ Schema Cache | İlk fetch sonrası cache'le | [Milvus SDK v2](https://medium.com/vector-database/introducing-milvus-sdk-v2-native-async-support-unified-apis-and-superior-performance-388c3eb6fa2d) |
| ⚠️ SDK v1 deprecated | Milvus 3.0'da v1 desteği bitecek | [Milvus SDK v2](https://medium.com/vector-database/introducing-milvus-sdk-v2-native-async-support-unified-apis-and-superior-performance-388c3eb6fa2d) |

### Örnek: Async Milvus Kullanımı

```python
from pymilvus import AsyncMilvusClient
import asyncio

# ✅ Doğru - Async client
async def search_vectors(queries: list[list[float]]):
    client = AsyncMilvusClient(uri="http://localhost:19530")

    # Paralel search
    tasks = [
        client.search(
            collection_name="documents",
            data=[query],
            limit=10,
            output_fields=["content", "metadata"]
        )
        for query in queries
    ]

    results = await asyncio.gather(*tasks)
    return results

# ✅ Doğru - Batch insert
async def batch_insert(documents: list[dict]):
    client = AsyncMilvusClient(uri="http://localhost:19530")

    # Chunk'lara böl ve paralel insert
    chunk_size = 1000
    chunks = [documents[i:i+chunk_size] for i in range(0, len(documents), chunk_size)]

    tasks = [client.insert("documents", chunk) for chunk in chunks]
    await asyncio.gather(*tasks)
```

---

## 7. Tool Tasarımı

| Kural | Açıklama |
|-------|----------|
| ✅ Single Responsibility | Her tool tek bir iş yapsın |
| ✅ Docstring zorunlu | LLM tool description olarak görür |
| ✅ Type hints zorunlu | Schema generation için |
| ✅ String return | Exception yerine hata mesajı döndür |
| ✅ Async tercih | I/O-bound işlemler için |
| ❌ Generic tool'lar | "do_everything" gibi tool'lar yazma |

### Örnek: Tool Tanımlama

```python
from tools.base import register_tool

# ✅ Doğru - Tek sorumluluk, açık docstring
@register_tool(tags=["database", "users"])
async def get_user_by_id(user_id: str) -> str:
    """Get user information by their unique ID.

    Args:
        user_id: The unique identifier of the user (e.g., "usr_12345")

    Returns:
        User information including name, email, and account status.
    """
    try:
        user = await db.users.find_one({"id": user_id})
        if not user:
            return f"User with ID '{user_id}' not found."
        return f"User: {user['name']}, Email: {user['email']}, Status: {user['status']}"
    except Exception as e:
        return f"Error retrieving user: {str(e)}"

# ❌ Yanlış - Çok amaçlı, belirsiz
@register_tool()
def do_database_stuff(action: str, data: dict) -> str:
    """Do various database operations."""  # Çok belirsiz!
    if action == "get":
        ...
    elif action == "create":
        ...
    elif action == "delete":
        ...
```

---

## 8. Güvenlik (Guardrails)

| Kural | Açıklama |
|-------|----------|
| ✅ Input validation | Prompt injection, toxic content, PII |
| ✅ Output validation | Sensitive data leak kontrolü |
| ✅ Pre/post model hooks | LangGraph 1.0 native |
| ✅ Rate limiting | API seviyesinde (nginx/traefik) |
| ✅ Max input length | Token/karakter limiti |

### Örnek: Guardrail Implementasyonu

```python
from guardrails.base import GuardrailResult
import re

# ✅ Input guardrail
def prompt_injection_check(content: str) -> GuardrailResult:
    """Detect potential prompt injection attempts."""
    injection_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?prior\s+instructions",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"jailbreak",
        r"DAN\s+mode",
    ]

    content_lower = content.lower()
    for pattern in injection_patterns:
        if re.search(pattern, content_lower):
            return GuardrailResult(
                passed=False,
                message="I cannot process this request.",
                triggered_rule="prompt_injection",
            )

    return GuardrailResult(passed=True)

# ✅ Output guardrail
def pii_output_check(content: str) -> GuardrailResult:
    """Prevent PII leakage in responses."""
    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",              # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ]

    for pattern in pii_patterns:
        if re.search(pattern, content):
            return GuardrailResult(
                passed=False,
                message="Response contained sensitive information.",
                triggered_rule="pii_leak",
            )

    return GuardrailResult(passed=True)
```

---

## 9. Observability

| Kural | Açıklama |
|-------|----------|
| ✅ Langfuse tracing | Tüm agent çağrıları trace edilmeli |
| ✅ External prompt management | Langfuse'dan prompt yönetimi |
| ✅ Session/User ID zorunlu | Her request'te olmalı |
| ✅ Structured logging | JSON format, structlog |
| ✅ p95/p99 latency monitoring | Average yerine percentile |

### Örnek: Observability Setup

```python
import structlog
from observability import create_trace_handler

# ✅ Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
)

logger = structlog.get_logger()

# ✅ Langfuse tracing
async def chat_with_tracing(request: ChatRequest):
    handler = create_trace_handler(
        session_id=request.session_id,
        user_id=request.user_id,
        trace_name="chat",
        metadata={"source": request.metadata.get("source")},
    )

    logger.info(
        "chat_request_received",
        user_id=request.user_id,
        session_id=request.session_id,
        message_count=len(request.messages),
    )

    result = await agent.ainvoke(
        {"messages": request.messages},
        config={"callbacks": [handler]},
    )

    logger.info(
        "chat_request_completed",
        user_id=request.user_id,
        session_id=request.session_id,
    )

    return result
```

---

## 10. Code Style

| Kural | Açıklama |
|-------|----------|
| ✅ Type hints zorunlu | Tüm fonksiyonlarda |
| ✅ Google-style docstrings | Args, Returns, Raises |
| ✅ Async-first | Sync sadece CPU-bound için |
| ❌ Circular imports | Modül yapısına dikkat |
| ❌ `Any` type | Mümkün olduğunca kaçın |
| ✅ `ruff` linting | Modern, hızlı linter |

### Örnek: Kod Standartları

```python
from typing import Optional
from pydantic import BaseModel

# ✅ Doğru - Type hints, docstring, async
async def process_message(
    message: str,
    user_id: str,
    session_id: str,
    temperature: float = 0.7,
) -> dict[str, str]:
    """Process a user message and return agent response.

    Args:
        message: The user's input message.
        user_id: Unique identifier for the user.
        session_id: Current conversation session ID.
        temperature: LLM temperature setting (0.0-2.0).

    Returns:
        Dictionary containing the agent's response and metadata.

    Raises:
        ValueError: If message is empty.
        ConnectionError: If LLM service is unavailable.
    """
    if not message.strip():
        raise ValueError("Message cannot be empty")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": session_id}},
    )

    return {
        "content": result["messages"][-1].content,
        "session_id": session_id,
    }

# ❌ Yanlış - Type hints yok, docstring yok
def process(msg, uid, sid, temp=0.7):
    result = agent.invoke({"messages": [{"role": "user", "content": msg}]})
    return result
```

---

## Referanslar

### LangGraph
- [LangGraph 1.0 Announcement](https://www.blog.langchain.com/langchain-langgraph-1dot0/)
- [Building LangGraph](https://www.blog.langchain.com/building-langgraph/)
- [State of Agent Engineering](https://www.langchain.com/state-of-agent-engineering)
- [Agent Orchestration 2026](https://iterathon.tech/blog/ai-agent-orchestration-frameworks-2026)

### FastAPI
- [FastAPI Async Documentation](https://fastapi.tiangolo.com/async/)
- [FastAPI Production Deployment](https://render.com/articles/fastapi-production-deployment-best-practices)
- [FastAPI Best Practices GitHub](https://github.com/zhanymkanov/fastapi-best-practices)

### Python Asyncio
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python Async IO](https://realpython.com/async-io-python/)
- [Elastic Async Patterns](https://www.elastic.co/blog/async-patterns-building-python-service)

### Pydantic
- [Pydantic v2 Features](https://pydantic.dev/articles/pydantic-v2)
- [Pydantic Performance](https://docs.pydantic.dev/latest/concepts/performance/)
- [Pydantic Migration Guide](https://docs.pydantic.dev/latest/migration/)

### vLLM
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [vLLM Quickstart 2026](https://www.glukhov.org/post/2026/01/vllm-quickstart/)

### Milvus
- [Milvus SDK v2 Async](https://medium.com/vector-database/introducing-milvus-sdk-v2-native-async-support-unified-apis-and-superior-performance-388c3eb6fa2d)
- [Milvus Best Practices](https://milvus.io/ai-quick-reference/what-are-vector-database-best-practices)
- [Milvus Documentation](https://milvus.io/docs)

---

## Versiyon Geçmişi

| Tarih | Versiyon | Değişiklikler |
|-------|----------|---------------|
| 2026-01-16 | 1.0.0 | İlk sürüm - 2026 best practices araştırmasına dayalı |
