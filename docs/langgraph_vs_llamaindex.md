# LangGraph vs LlamaIndex Workflows — A Study Guide

> Based on the real architecture of the AI Tutor project.  
> Both branches run the same pipeline: BKT → Diagnose → Recommend.  
> The only difference is *how* they orchestrate the steps.

---

## 1. What problem are these frameworks solving?

When you build an AI pipeline with multiple steps, you need something to:
- Run Step A, wait for it to finish, then pass its result to Step B
- Handle errors gracefully
- Give you observability (what happened in each step?)
- Let you run steps concurrently when they don't depend on each other

Both LangGraph and LlamaIndex Workflows solve this — just with different philosophies.

---

## 2. The Core Mental Model

### LangGraph — "Think like a flowchart"

LangGraph models your pipeline as a **directed graph** (like a flowchart).
- **Nodes** = functions that do work
- **Edges** = connections between nodes (what runs next)
- **State** = a shared dictionary that every node reads from and writes to

```
START → [run_bkt] → [diagnose] → [recommend] → END
           ↑            ↑              ↑
        (node)       (node)         (node)
```

All nodes share one `AgentState` dictionary. Each node reads what it needs and writes its result back in.

```python
# LangGraph approach (from langgraph_direct branch)
graph = StateGraph(AgentState)
graph.add_node("run_bkt",   run_bkt_node)
graph.add_node("diagnose",  diagnose_node)
graph.add_node("recommend", recommend_node)

graph.add_edge(START,       "run_bkt")
graph.add_edge("run_bkt",   "diagnose")
graph.add_edge("diagnose",  "recommend")
graph.add_edge("recommend", END)

tutor_graph = graph.compile()
await tutor_graph.ainvoke(state)
```

### LlamaIndex Workflows — "Think like events at a concert"

LlamaIndex models your pipeline as **event-driven steps**.
- **Steps** = async functions decorated with `@step`
- **Events** = typed messages passed between steps (like tickets at a concert)
- No shared state dictionary — each step receives its input via an event and emits a new event

```
StartEvent → [run_bkt step] → BKTDoneEvent
                                    ↓
                          [diagnose step] → DiagnosisDoneEvent
                                                  ↓
                                        [recommend step] → StopEvent
```

Each event carries exactly what the next step needs — nothing more.

```python
# LlamaIndex approach (from llamaindex-experiment branch)
class TutorWorkflow(Workflow):

    @step
    async def run_bkt(self, ev: StartEvent) -> BKTDoneEvent:
        result = await run_bkt_node(...)
        return BKTDoneEvent(diagnosis=result["diagnosis"], ...)

    @step
    async def diagnose(self, ev: BKTDoneEvent) -> DiagnosisDoneEvent:
        result = await diagnose_node(...)
        return DiagnosisDoneEvent(analysis=result["analysis"], ...)

    @step
    async def recommend(self, ev: DiagnosisDoneEvent) -> StopEvent:
        result = await recommend_node(...)
        return StopEvent(result=result["feedback"])
```

---

## 3. Key Differences at a Glance

| Feature | LangGraph | LlamaIndex Workflows |
|---|---|---|
| **Mental model** | Graph (nodes + edges) | Event bus (steps + events) |
| **Data passing** | Shared state dict (`AgentState`) | Typed events between steps |
| **Step definition** | Regular async function | Class method with `@step` decorator |
| **Routing** | Explicit `add_edge()` calls | Inferred from event types (return type → next step) |
| **Concurrency** | You manage it (`asyncio.gather`) | Same — you manage it |
| **Type safety** | TypedDict (loose) | Pydantic Events (strict) |
| **Observability** | Manual with Langfuse `@observe` | Native Langfuse LlamaIndex handler |
| **Origin** | LangChain ecosystem | LlamaIndex ecosystem |

---

## 4. How Data Flows — Side by Side

Both pipelines do the same thing. Here's how data moves through each.

### LangGraph (shared state)

```
state = {
    "student_id": "student_1",
    "obs": tensor,
    "output": tensor,
    "skill_id_to_name": {...},
    "diagnosis": {},    ← written by run_bkt_node
    "analysis": [],     ← written by diagnose_node
    "feedback": [],     ← written by recommend_node
}

run_bkt_node(state)   → adds state["diagnosis"]
diagnose_node(state)  → adds state["analysis"]
recommend_node(state) → adds state["feedback"]
```

Every node sees the entire state. A node can accidentally overwrite something another node wrote. This is flexible but requires discipline.

### LlamaIndex (typed events)

```
StartEvent(student_id, obs, output, skill_id_to_name)
    ↓
BKTDoneEvent(student_id, obs, output, skill_id_to_name, diagnosis)
    ↓
DiagnosisDoneEvent(student_id, analysis)
    ↓
StopEvent(result=feedback)
```

Each step only receives what it declared it needs. There's no way to accidentally read stale data from an earlier step — the event type enforces what's available.

---

## 5. State Definition

### LangGraph

```python
# state.py — langgraph_direct branch
class AgentState(TypedDict):
    student_id: str
    obs: Any
    output: Any
    skill_id_to_name: dict
    diagnosis: dict
    analysis: list
    feedback: list
```

One flat dictionary. All fields must be defined upfront.

### LlamaIndex

```python
# state.py — llamaindex-experiment branch
from llama_index.core.workflow import Event

class BKTDoneEvent(Event):
    student_id: str
    obs: Any
    output: Any
    skill_id_to_name: dict
    diagnosis: dict

class DiagnosisDoneEvent(Event):
    student_id: str
    analysis: list
```

Events are Pydantic models. Each event only contains what the *next* step needs. If you forget to include a field, Python raises an error at the call site — not silently at runtime.

---

## 6. Observability with Langfuse

This is where the two approaches differ most practically.

### LangGraph + Langfuse

You must manually decorate every function you want to trace:

```python
@observe(name="run_bkt")
async def run_bkt_node(state): ...

@observe(name="diagnose")
async def diagnose_node(state): ...

@observe(name="recommend")
async def recommend_node(state): ...
```

Each decorator creates a span in Langfuse. You control exactly what gets traced.

### LlamaIndex + Langfuse

LlamaIndex has a native Langfuse integration. Register the handler once:

```python
# graph.py — llamaindex-experiment branch
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager

_langfuse_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([_langfuse_handler])
```

After this, **every `@step` execution is automatically traced** — no `@observe` needed on individual functions. You still add `@observe` for extra named spans, but the framework-level tracing is free.

---

## 7. When to Use Which

### Use LangGraph when:

- You need **conditional routing** — "if proficiency is low, go to remediation node; if high, go to challenge node"
- You want to add **cycles** — "retry this step if the LLM output is invalid"
- You're already using LangChain tools and want tight integration
- Your team thinks in flowcharts

```python
# LangGraph can do conditional edges — LlamaIndex cannot as easily
graph.add_conditional_edges(
    "diagnose",
    lambda state: "remediate" if state["analysis"]["level"] == "하" else "advance"
)
```

### Use LlamaIndex Workflows when:

- You want **strict typed interfaces** between steps (fewer surprises)
- You're building a pipeline where steps are clearly sequential with well-defined inputs/outputs
- You want automatic Langfuse tracing with minimal setup
- You're already using LlamaIndex for RAG (document loading, vector search, etc.)

---

## 8. The Async Bottleneck Problem

Both branches had the same bug: `run_bkt_node` and `diagnose_node` were **synchronous** inside async workflow steps.

### Why this is a problem

Python's async event loop is single-threaded. When a sync function runs, it **blocks the entire event loop** — no other coroutine can run until it finishes.

```
Event loop timeline (BEFORE fix):
─────────────────────────────────────────────────────────
run_bkt (async step)
  └─ run_bkt_node() [SYNC — blocks 2 seconds]
                    ────────────────────────
                    nothing else can run here
─────────────────────────────────────────────────────────
```

### The fix: use async HTTP client

```python
# BEFORE (blocks event loop)
with httpx.Client(timeout=60.0) as client:
    resp = client.post(...)

# AFTER (yields control while waiting for network)
async with httpx.AsyncClient(timeout=60.0) as client:
    resp = await client.post(...)
```

The `await` tells the event loop: "I'm waiting for a network response, go do something else." This is critical when handling multiple students concurrently via the FastAPI endpoint.

### The same issue existed for OpenAI calls

```python
# BEFORE
client = OpenAI()                           # sync client
response = client.chat.completions.create(...)

# AFTER
client = AsyncOpenAI()                      # async client
response = await client.chat.completions.create(...)
```

---

## 9. The Neo4j Tool: MCP Toolbox → Direct Driver

The `llamaindex-experiment` branch originally used **Google GenAI Toolbox** to talk to Neo4j. This was replaced with a direct async Neo4j driver.

### Why we removed Toolbox

| | MCP Toolbox | Direct Driver |
|---|---|---|
| **Extra processes needed** | Yes (Go binary server + Docker) | No |
| **Port conflicts** | Yes (macOS AirPlay on port 5000) | No |
| **Protocol versions** | Must match client + server | Not applicable |
| **Cold start** | Toolbox server must be running | Connect on first query |
| **Code complexity** | `ToolboxClient` + `Protocol` + `load_toolset` | `AsyncGraphDatabase.driver()` |
| **Production value** | Useful for multi-database/multi-tool setups | Sufficient for single Neo4j |

### The async tool interface

The direct driver was designed to keep `recommendation_node.py` unchanged:

```python
# neo4j_tool.py

class _Tool:
    async def __call__(self, **kwargs) -> list[dict]:
        return await _run_query(self._query, kwargs)

def get_tool(name: str) -> _Tool:
    # sync — just a cache lookup, no I/O
    ...

# recommendation_node.py — same call pattern as before
tool = get_tool(tool_name)          # sync (was: await get_tool)
raw  = await tool(skill_id=skill_id)  # async — actual Neo4j query
```

The key insight: `get_tool` is sync because it only looks up a cache. The actual network I/O happens in `await tool(...)` which is async.

---

## 10. Cloud Architecture Summary

| | `langgraph_direct` | `llamaindex-experiment` |
|---|---|---|
| `ai-tutor` deployment | **Cloud Run Job** (batch) | **Cloud Run Service** (API) |
| Triggered by | CLI / cron | HTTP POST /tutor |
| Multi-student | Sequential | Concurrent (async) |
| Frontend integration | ❌ | ✅ |
| Entry point | `ai_tutor.main:cli` | `ai_tutor.api:serve` |

The FastAPI endpoint in `llamaindex-experiment` accepts:
```json
POST /tutor
{
  "student_id": "student_1",
  "sequence": [[5, 1], [5, 0], [12, 1], ...],
  "skill_id_to_name": {"5": "덧셈", "12": "뺄셈"}
}
```

And returns per-skill feedback that a frontend can display directly.

---

## Further Reading

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Workflows Docs](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)
- [Langfuse LlamaIndex Integration](https://langfuse.com/docs/integrations/llama-index)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
- [httpx AsyncClient](https://www.python-httpx.org/async/)
- [neo4j Python Driver (async)](https://neo4j.com/docs/python-manual/current/async/)
