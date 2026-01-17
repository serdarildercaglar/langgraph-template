# LangGraph 2026 Best Practices

Bu doküman, LangGraph framework'ünü kullanırken izlenen best practice'leri ve bu projedeki mimari kararların gerekçelerini açıklar.

> **Kaynak**: Bu best practice'ler LangGraph resmi dokümantasyonu, GitHub tartışmaları ve LangChain ekibinin önerilerine dayanmaktadır (Ocak 2026, LangGraph 1.0.6).

## İçindekiler

1. [Agent Oluşturma](#1-agent-oluşturma)
2. [State Yönetimi](#2-state-yönetimi)
3. [Multi-Agent vs Single Agent](#3-multi-agent-vs-single-agent)
4. [Tool Tanımlama](#4-tool-tanımlama)
5. [Checkpointer Kullanımı](#5-checkpointer-kullanımı)
6. [Prompt Yönetimi](#6-prompt-yönetimi)
7. [Pre/Post Model Hooks (Yeni - 2026)](#7-prepost-model-hooks-yeni---2026)
8. [Node Caching (Yeni - 2026)](#8-node-caching-yeni---2026)
9. [Deferred Nodes (Yeni - 2026)](#9-deferred-nodes-yeni---2026)
10. [Anti-Patterns](#10-anti-patterns)

---

## 1. Agent Oluşturma

### ✅ Önerilen: `create_react_agent` Kullanımı

LangGraph'ın prebuilt `create_react_agent` fonksiyonu, ReAct pattern'ini implement eder ve çoğu use case için yeterlidir.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=model,
    tools=[tool1, tool2],
    prompt="You are a helpful assistant.",
    checkpointer=checkpointer,  # Opsiyonel: persistence için
)
```

**Neden?**
- Tool calling loop otomatik yönetilir
- Message state otomatik güncellenir
- Error handling dahil
- LangGraph ekibi tarafından optimize edilmiş

### ❌ Kaçınılması Gereken: Custom ReAct Loop

```python
# YAPMAYIN - gereksiz karmaşıklık
class ToolAgent:
    def process(self, state):
        response = self.model.invoke(messages)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool = self.find_tool(tool_call["name"])
                result = tool.invoke(tool_call["args"])
                # ... manuel tool result handling
```

**Sorun**: `create_react_agent` bunu zaten yapıyor. Tekrar implement etmek:
- Hata yapma riski artırır
- Maintenance yükü getirir
- LangGraph güncellemelerinden yararlanmayı engeller

### Bu Projede

```python
# graph/builder.py
async def create_agent(...) -> CompiledStateGraph:
    return create_react_agent(
        model=get_chat_model(model_name),
        tools=list(tools) if tools else [],
        prompt=system_prompt,
        checkpointer=resolved_checkpointer,
    )
```

Wrapper'ımız sadece şunları ekliyor:
- Langfuse'dan prompt fetch etme
- Settings'den checkpointer yapılandırma
- vLLM model entegrasyonu

---

## 2. State Yönetimi

### ✅ Önerilen: TypedDict + `add_messages` Reducer

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

**Neden TypedDict?**
- Lightweight, runtime overhead yok
- LangGraph ile native uyumluluk
- Partial updates kolay (sadece değişen key'leri döndür)

**Neden Pydantic Değil?**
- Pydantic validation her update'te çalışır → performans maliyeti
- Agent workflow'larında state sık güncellenir
- Validation gerekliyse sadece input/output boundary'de kullanın

### ✅ Önerilen: Minimal State

```python
# Çoğu case için bu yeterli
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### ❌ Kaçınılması Gereken: Aşırı State Field'ları

```python
# YAPMAYIN - çoğu field gereksiz
class OverEngineeredState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_agent: str           # LangGraph routing ile çözülür
    tool_results: list[dict]     # Messages içinde zaten var
    iteration_count: int         # create_react_agent yönetiyor
    error_state: str             # Exception handling ile çözülür
```

### Bu Projede

```python
# graph/state.py
from langgraph.graph import MessagesState  # Re-export

class RAGState(TypedDict):
    """Sadece RAG workflow'ları için extended state"""
    messages: Annotated[list[AnyMessage], add_messages]
    context: list[str]
    query: str
```

Standart agent için `MessagesState` (LangGraph built-in) kullanıyoruz.

---

## 3. Multi-Agent vs Single Agent

### LangGraph Ekibinin Önerisi (Şubat 2025)

> "We now recommend using the supervisor pattern directly via tools rather than this library."
> — [langgraph-supervisor-py README](https://github.com/langchain-ai/langgraph-supervisor-py)

### Supervisor Pattern'in Sorunları

```
┌─────────────┐
│  Supervisor │ ──▶ "Route to AGENT_A" (text parsing)
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐ ┌─────┐
│Agent│ │Agent│
└─────┘ └─────┘
```

| Sorun | Açıklama |
|-------|----------|
| **Text Parsing Kırılganlığı** | LLM "AGENT_A" yerine "Agent A" veya "the first agent" yazabilir |
| **Context Clutter** | Sub-agent'lar supervisor'ın routing mantığını context'te görür |
| **State Karmaşıklığı** | `SupervisorState`, `next`, `agents_called` gibi ek field'lar gerekir |
| **Debugging Zorluğu** | Hangi agent'ın ne zaman çağrıldığını izlemek zor |

### ✅ Önerilen: Tool-Based Handoff

```python
from graph.builder import create_agent
from langchain_core.tools import tool

# Sub-agent'ı tool olarak tanımla (Langfuse'dan prompt çeker)
@tool
async def call_research_agent(query: str) -> str:
    """Derinlemesine araştırma için research agent'ı çağır."""
    research_agent = await create_agent(
        tools=research_tools,
        langfuse_prompt_name="research-agent",  # Langfuse'dan prompt çek
    )
    result = await research_agent.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

@tool
async def call_code_agent(task: str) -> str:
    """Kod yazma görevi için code agent'ı çağır."""
    code_agent = await create_agent(
        tools=code_tools,
        langfuse_prompt_name="code-agent",  # Langfuse'dan prompt çek
    )
    result = await code_agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content

# Ana agent tüm sub-agent'ları tool olarak kullanır
main_agent = await create_agent(
    tools=[
        call_research_agent,
        call_code_agent,
        regular_tool_1,
        regular_tool_2,
    ],
    langfuse_prompt_name="main-assistant",
)
```

**Avantajları:**
- Tool calling güvenilir (LLM'ler bunun için eğitilmiş)
- Her tool call izole context'te
- Langfuse'da net trace'ler
- Supervisor state yönetimi yok
- **Her sub-agent kendi Langfuse prompt'unu kullanır** (versiyon kontrolü, A/B test)

> **Not:** Sub-agent prompt'ları Langfuse'da tanımlanmalıdır:
> - `research-agent`: Araştırma uzmanı rolü, kaynak gösterme formatı
> - `code-agent`: Kod yazma uzmanı rolü, kod standartları

### Ne Zaman Multi-Agent Gerekir?

| Senaryo | Öneri |
|---------|-------|
| 5-10 tool | Single agent yeterli |
| 10-20 tool | Tool'ları kategorize et, hala single agent |
| 20+ tool veya farklı domain'ler | Tool-based handoff ile sub-agent'lar |
| Agent'lar arası conversation | O zaman gerçek multi-agent (nadir) |

### Bu Projede

Single agent + tools yaklaşımı kullanıyoruz:

```python
# api/routes.py
_compiled_graph = await create_agent(
    tools=get_all_tools(),  # Tüm tool'lar tek agent'a
    langfuse_prompt_name="main-assistant",
)
```

---

## 4. Tool Tanımlama

### ✅ Önerilen: LangChain `@tool` Decorator

```python
from langchain_core.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the database for records.

    Args:
        query: Search query string
        limit: Maximum number of results
    """
    return f"Found {limit} results"
```

**Önemli:**
- Docstring = Tool description (LLM bunu görür)
- Type hints = Schema generation
- Default values = Optional parameters

### ✅ Önerilen: Registry Pattern (Opsiyonel)

```python
# tools/base.py
_tool_registry: dict[str, StructuredTool] = {}

def register_tool(name=None, tags=None):
    def decorator(func):
        structured_tool = tool(name=name or func.__name__)(func)
        _tool_registry[structured_tool.name] = structured_tool
        return structured_tool
    return decorator
```

**Faydası:**
- `get_all_tools()` ile tüm tool'lara erişim
- Tag'lerle filtreleme (`get_tools_by_tag("rag")`)
- Test'lerde `clear_registry()`

### ❌ Kaçınılması Gereken: Overengineered Tool Wrapper'ları

```python
# YAPMAYIN
class ToolResult(BaseModel):
    success: bool
    data: Any
    error: str | None

def safe_tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(str(e))
    return wrapper
```

**Sorun:**
- LangGraph zaten error handling yapıyor
- Ekstra serialization overhead
- Tool output'u karmaşıklaştırıyor

---

## 5. Checkpointer Kullanımı

### ✅ Önerilen: Thread-Based Persistence

```python
# Checkpointer ile agent oluştur
agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
)

# Her conversation için unique thread_id
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "session-123"}},
)
```

### Checkpointer Seçenekleri

| Backend | Use Case |
|---------|----------|
| `MemorySaver` | Development, testing |
| `SqliteSaver` | Single-instance production |
| `AsyncSqliteSaver` | Async single-instance |
| `PostgresSaver` | Multi-instance production |

### Bu Projede

```python
# graph/builder.py
async def get_checkpointer():
    settings = get_settings().checkpoint
    if settings.backend == "sqlite":
        return AsyncSqliteSaver.from_conn_string(settings.sqlite_path)
    elif settings.backend == "memory":
        return MemorySaver()
```

---

## 6. Prompt Yönetimi

### ✅ Önerilen: External Prompt Management

Prompt'ları kod dışında yönetmek:
- A/B testing kolaylaşır
- Deployment olmadan prompt güncellenebilir
- Version control ve rollback

### Langfuse Entegrasyonu

```python
from langfuse import Langfuse

langfuse = Langfuse()

def get_prompt(name: str) -> str | None:
    try:
        prompt = langfuse.get_prompt(name)
        return prompt.compile()
    except:
        return None

# Kullanım
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=get_prompt("main-assistant") or "Default prompt",
)
```

### Bu Projede

```python
# graph/builder.py
async def create_agent(langfuse_prompt_name=None, prompt=None, ...):
    system_prompt = prompt
    if not system_prompt and langfuse_prompt_name:
        system_prompt = get_agent_prompt(langfuse_prompt_name)
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant."

    return create_react_agent(model=model, prompt=system_prompt, ...)
```

Öncelik sırası:
1. Explicit `prompt` parametresi
2. Langfuse'dan fetch (`langfuse_prompt_name`)
3. Default prompt

---

## 7. Pre/Post Model Hooks (Yeni - 2026)

LangGraph 1.0 ile gelen `pre_model_hook` ve `post_model_hook` parametreleri, model çağrılarından önce ve sonra özel mantık eklemenizi sağlar.

### ✅ Pre-Model Hook: Context Yönetimi

Uzun konuşmalarda context bloat'ı önlemek için mesaj özetleme:

```python
from graph.builder import create_agent, create_message_summarization_hook

agent = await create_agent(
    tools=get_all_tools(),
    langfuse_prompt_name="main-assistant",
    pre_model_hook=create_message_summarization_hook(max_messages=20),
)
```

**Custom Hook Örneği:**

```python
def summarize_context_hook(state: dict) -> dict | None:
    """Uzun conversation'larda eski mesajları özetle."""
    messages = state.get("messages", [])
    if len(messages) <= 15:
        return None  # Değişiklik yok

    # İlk (system) ve son 10 mesajı koru
    # Ortadaki mesajları özetle (production'da LLM ile yapılabilir)
    summary = f"[Previous {len(messages) - 11} messages summarized]"
    from langchain_core.messages import SystemMessage

    kept = [messages[0]] + [SystemMessage(content=summary)] + messages[-10:]
    return {"messages": kept}
```

### ✅ Post-Model Hook: Guardrails

Model çıktısını kontrol etmek ve filtrelemek için:

```python
from graph.builder import create_agent, create_guardrail_hook

def no_sensitive_data(state: dict) -> bool:
    """PII veya hassas veri kontrolü."""
    content = state["messages"][-1].content
    # Basit örnek - production'da daha sofistike kontrol gerekir
    sensitive_patterns = ["SSN:", "Credit Card:", "Password:"]
    return not any(p in content for p in sensitive_patterns)

agent = await create_agent(
    tools=get_all_tools(),
    post_model_hook=create_guardrail_hook(
        check_fn=no_sensitive_data,
        block_message="I cannot share sensitive information.",
    ),
)
```

### ✅ Human-in-the-Loop Hook

Kritik kararlar için insan onayı:

```python
def require_approval_for_actions(state: dict) -> dict | None:
    """Belirli tool çağrıları için onay iste."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls"):
        dangerous_tools = ["delete_record", "send_email", "execute_query"]
        for tc in last_msg.tool_calls:
            if tc["name"] in dangerous_tools:
                # Human-in-the-loop interrupt
                from langgraph.types import interrupt
                interrupt(f"Approve action: {tc['name']}?")
    return None
```

---

## 8. Node Caching (Yeni - 2026)

Node caching, pahalı hesaplamaları önbelleğe alarak performansı artırır.

### ✅ Önerilen: Cache Policy ile Node Tanımlama

```python
from langgraph.graph import StateGraph
from langgraph.cache import SqliteCache, InMemoryCache, RedisCache

graph = StateGraph(AgentState)

# Pahalı node için caching
graph.add_node(
    "retrieve_documents",
    retrieve_fn,
    cache_policy={
        "ttl": 3600,  # 1 saat cache
        "cache": SqliteCache("cache.db"),
    }
)

# Hızlı node'lar için cache gerekmez
graph.add_node("format_response", format_fn)
```

### Cache Backend Seçenekleri

| Backend | Use Case |
|---------|----------|
| `InMemoryCache` | Development, tek process |
| `SqliteCache` | Single-instance production |
| `RedisCache` | Multi-instance, distributed |

### Ne Zaman Kullanmalı?

- ✅ RAG retrieval işlemleri
- ✅ Dış API çağrıları
- ✅ Pahalı embedding hesaplamaları
- ❌ State'e bağımlı dinamik işlemler
- ❌ Her zaman güncel olması gereken veriler

---

## 9. Deferred Nodes (Yeni - 2026)

Deferred node'lar, tüm upstream path'ler tamamlanana kadar çalışmayı erteler. Map-reduce ve consensus pattern'leri için idealdir.

### ✅ Önerilen: Map-Reduce Pattern

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)

# Paralel işlemler
graph.add_node("analyze_source_1", analyze_fn_1)
graph.add_node("analyze_source_2", analyze_fn_2)
graph.add_node("analyze_source_3", analyze_fn_3)

# Deferred: Tüm analizler tamamlandıktan sonra çalışır
graph.add_node("aggregate_results", aggregate_fn, defer=True)

# Edges
graph.add_edge(START, "analyze_source_1")
graph.add_edge(START, "analyze_source_2")
graph.add_edge(START, "analyze_source_3")
graph.add_edge("analyze_source_1", "aggregate_results")
graph.add_edge("analyze_source_2", "aggregate_results")
graph.add_edge("analyze_source_3", "aggregate_results")
graph.add_edge("aggregate_results", END)
```

### ✅ Multi-Agent Consensus

```python
# Birden fazla agent'ın aynı soruya cevap vermesi
graph.add_node("agent_1_response", agent_1_fn)
graph.add_node("agent_2_response", agent_2_fn)
graph.add_node("agent_3_response", agent_3_fn)

# Consensus: Tüm cevapları değerlendir
graph.add_node("reach_consensus", consensus_fn, defer=True)
```

### Ne Zaman Kullanmalı?

- ✅ Paralel veri toplama ve birleştirme
- ✅ Multi-agent voting/consensus
- ✅ Fan-out/fan-in pattern'leri
- ❌ Sequential workflow'lar
- ❌ Tek path'li grafikler

---

## 10. Anti-Patterns

### ❌ Class-Based Agent Hierarchy

```python
# YAPMAYIN
class BaseAgent(ABC):
    @abstractmethod
    def process(self, state): pass

class SimpleAgent(BaseAgent):
    def process(self, state):
        return self.model.invoke(messages)

class ToolAgent(BaseAgent):
    def process(self, state):
        # Custom tool loop
```

**Sorun:** `create_react_agent` bunu zaten yapıyor.

### ❌ Custom State Reducers (Çoğu Case'de)

```python
# YAPMAYIN (genellikle gereksiz)
def custom_message_reducer(current, new):
    # Custom merge logic
    ...

class CustomState(TypedDict):
    messages: Annotated[list, custom_message_reducer]
```

**Sorun:** `add_messages` reducer çoğu case'i handle ediyor.

### ❌ Supervisor Text Parsing

```python
# YAPMAYIN
def route_supervisor(state):
    content = state["messages"][-1].content.upper()
    if "AGENT_A" in content:
        return "agent_a"
    elif "FINISH" in content:
        return END
```

**Sorun:** Kırılgan, tool-based handoff kullanın.

### ❌ Gereksiz State Field'ları

```python
# YAPMAYIN
class State(TypedDict):
    messages: ...
    current_agent: str      # Routing ile çözülür
    agents_called: list     # Debugging için log kullanın
    iteration: int          # create_react_agent yönetiyor
```

---

## Özet Karar Tablosu

| Karar | Bu Projede | Gerekçe |
|-------|------------|---------|
| Agent oluşturma | `create_react_agent` | Native, optimize edilmiş |
| State | `MessagesState` (TypedDict) | Minimal, performanslı |
| Multi-agent | Tool-based handoff | Güvenilir, basit |
| Tool tanımlama | `@tool` + registry | Standart + keşfedilebilirlik |
| Prompt yönetimi | Langfuse | External, versionable |
| Persistence | Configurable checkpointer | Flexible backend |
| Context yönetimi | Pre-model hooks (1.0+) | Uzun conversation desteği |
| Güvenlik | Post-model hooks (1.0+) | Guardrails, HITL |
| Performans | Node caching (1.0+) | Pahalı işlemler için |

---

## Versiyon Geçmişi

| Tarih | LangGraph Versiyonu | Değişiklikler |
|-------|---------------------|---------------|
| Ocak 2026 | 1.0.6 | Pre/post hooks, node caching, deferred nodes eklendi |
| Ocak 2025 | 0.2.x | İlk versiyon, temel best practices |

---

## Referanslar

- [LangGraph How-to: ReAct Agent from Scratch](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
- [LangGraph Prebuilt Agents Discussion](https://github.com/langchain-ai/langgraph/discussions/4390)
- [LangGraph Supervisor Library](https://github.com/langchain-ai/langgraph-supervisor-py)
- [LangGraph State Management](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [TypedDict vs Pydantic for State](https://shazaali.substack.com/p/type-safety-in-langgraph-when-to)
- [LangGraph Workflow Updates (Python & JS)](https://changelog.langchain.com/announcements/langgraph-workflow-updates-python-js)
- [Node-level Caching in LangGraph](https://changelog.langchain.com/announcements/node-level-caching-in-langgraph)
- [LangGraph Release Week Recap](https://blog.langchain.com/langgraph-release-week-recap/)
- [LangGraph Releases](https://github.com/langchain-ai/langgraph/releases)
