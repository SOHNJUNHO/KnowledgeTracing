# Agentic GraphRAG Tutor

A production-grade AI tutoring system that combines a custom Neural Bayesian Knowledge Tracing model with an Agentic GraphRAG pipeline to deliver personalized, explainable study recommendations.

Based on the master's thesis: **TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성** — Junho Son, University of Seoul, 2026. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

---

## Why This Exists

LLMs are powerful feedback generators, but they have three fundamental problems in educational settings:

1. **Hallucination** — they produce inconsistent or factually wrong feedback
2. **Omitted Relationships** — they cannot reflect the curriculum or the relationships between knowledge concepts
3. **Opacity** — they cannot explain *why* they gave a particular response

This project addresses all three:
- **Transparent parameter extraction**: BKTransformer outputs BKT parameters (P(know), P(learn), P(guess), P(slip)) grounded in Bayes' theorem, making the knowledge state transparent instead of black-box.
- **Interpretable diagnosis**: The diagnosis agent logically evaluates BKT parameters to assign a proficiency level (High / Mid / Low) per knowledge concept.
- **Hallucination-free recommendations**: Based on the diagnosis, the recommendation agent queries a knowledge graph (Neo4j) that encodes curriculum relationships, producing grounded feedback without hallucination.

---

## Architecture

### Pipeline

```
Student Interaction History (skill_id, correct) × T timesteps
                        │
                        ▼
         ┌──────────────────────────┐
         │   BKTransformer (PyTorch)│  ← bkt-service (port 8001)
         │       RoPE · SwiGLU      │
         └──────────────────────────┘
                        │
          Per-skill BKT Parameters:
          P(know), P(learn), P(guess), P(slip)
                        │
                        ▼
         ┌──────────────────────────┐
         │     Diagnosis Agent      │  ← LLM via orchestrator-api
         │   (OpenAI / vLLM)        │
         └──────────────────────────┘
                        │
              Proficiency Level (상 / 중 / 하)
                        │
                        ▼
         ┌──────────────────────────┐
         │   Recommendation Agent   │  ← LLM + direct Neo4j access
         │   (OpenAI / vLLM)        │
         │   Neo4j GraphRAG         │
         │   Predefined Cypher Query│
         └──────────────────────────┘
                        │
                        ▼
            Personalized Study Feedback
```

**Workflow:** `StartEvent → run_bkt → BKTDoneEvent → diagnose → DiagnosisDoneEvent → recommend → StopEvent`

**Cypher Query Selection by Proficiency:**
| Level | Query | Purpose |
|---|---|---|
| 하 (Low) | `get_prerequisites` | Retrieve prerequisite concepts to study first |
| 중 (Mid) | `get_current_concept` | Reinforce the current concept |
| 상 (High) | `get_advanced_concepts` | Surface next concepts the student is ready for |

### Microservices

The current runtime is split into three primary services, plus an optional local LLM server:

```
orchestrator-api   src/ai_tutor/api.py          port 8000   FastAPI — drives the tutor workflow
bkt-service        services/bkt_service.py       port 8001   BKTransformer inference (PyTorch)
vllm-service       vllm/vllm-openai (image)      port 8080   Local LLM serving (opt-in, GPU)
```

Services communicate over HTTP where appropriate. The orchestrator calls the BKT service over HTTP and connects to Neo4j directly via the official async driver.

---

## Key Design Decisions

**1. BKT Parameters as LLM Input**
Unlike conventional deep learning KT models that output proficiency directly through uninterpretable neural computations, BKTransformer derives proficiency by applying Bayes' theorem to explicit BKT parameters. This provides transparency into how and why each correctness probability was produced.

**2. Predefined Tool Selection**
The recommendation agent does not decide which Cypher query to run. The proficiency level from the diagnosis agent determines the query. This guarantees transparency in the GraphRAG retrieval step.

**3. LLM Backend Toggle**
`LLM_BACKEND=openai` routes calls to the OpenAI API (with Langfuse tracing). `LLM_BACKEND=vllm` routes to a local vLLM server with no external dependency. Switching requires no code change — only a config update.

**4. PyTorch over ONNX Runtime for BKT Inference**
ONNX Runtime was evaluated as an alternative to PyTorch for the BKT service. The transformer encoder (`_encode`) is well-suited to ONNX — it is a parallel tensor computation with a static graph. However, `_run_bkt_loop` iterates `range(block_size)` sequentially, where each step depends on the latent state from the previous step. ONNX tracing unrolls this loop entirely into the static graph, producing one copy of every BKT operation per timestep. With `block_size=818`, this generates an unmanageably large graph (minutes to export, multi-GB model file) with no inference speedup, because the bottleneck is the sequential dependency — not the per-step compute. PyTorch runs the loop dynamically and is therefore faster and lighter for this architecture. `torch.quantization.quantize_dynamic` is applied at startup to compress Linear weights to int8, giving meaningful speedup without the ONNX tradeoffs.

**5. Langfuse for Full Observability**
Every pipeline run is traced end-to-end: BKT output, LLM prompts/responses, graph context, and final feedback. This supports human expert evaluation and iterative prompt improvement.

---

## LlamaIndex Workflows vs LangGraph

The pipeline was initially prototyped using **LangGraph** and later migrated to **LlamaIndex Workflows**. This section explains what changed and why.

### LangGraph — "Think like a flowchart"

LangGraph models the pipeline as a directed graph. Nodes are functions, edges are connections, and all nodes share a single mutable `AgentState` dictionary.

```python
graph = StateGraph(AgentState)
graph.add_node("run_bkt",  run_bkt_node)
graph.add_node("diagnose", diagnose_node)
graph.add_node("recommend",recommend_node)
graph.add_edge(START,       "run_bkt")
graph.add_edge("run_bkt",   "diagnose")
graph.add_edge("diagnose",  "recommend")
graph.add_edge("recommend", END)
tutor_graph = graph.compile()
await tutor_graph.ainvoke(state)
```

Every node reads whatever it wants from `AgentState` and writes back into it. This is flexible but has a cost: it is easy for a node to silently read a key that was never set, or overwrite a key another node depends on. The edges and state schema must be kept in sync manually.

### LlamaIndex Workflows — "Think like typed events"

LlamaIndex Workflows model the pipeline as event-driven steps. Each step receives a typed event and emits a typed event. There is no shared mutable dictionary.

```python
class TutorWorkflow(Workflow):

    @step
    async def run_bkt(self, ev: StartEvent) -> BKTDoneEvent:
        result = await run_bkt_node(ev.state)
        return BKTDoneEvent(diagnosis=result["diagnosis"], ...)

    @step
    async def diagnose(self, ev: BKTDoneEvent) -> DiagnosisDoneEvent:
        result = await diagnose_node(ev.state)
        return DiagnosisDoneEvent(analysis=result["analysis"], ...)

    @step
    async def recommend(self, ev: DiagnosisDoneEvent) -> StopEvent:
        result = await recommend_node(ev.state)
        return StopEvent(result=result["feedback"])
```

The workflow infers execution order from the event types — if `diagnose` takes a `BKTDoneEvent`, it will automatically run after the step that emits one. There are no manual `add_edge` calls.

### Why the migration was made

The pipeline is strictly linear — BKT → Diagnose → Recommend, always in that order, no branching. LangGraph's strengths (conditional routing, agentic loops, human-in-the-loop) add no value here.

The decisive reason was **typed events as data contracts**. In LangGraph every node reads from and writes to a single shared `AgentState` dict. A node can silently read a key that was never set, or overwrite a key another node depends on — nothing prevents it at runtime. In LlamaIndex Workflows each step receives a typed Pydantic event containing exactly what it needs and nothing else. A missing or mistyped field raises an error immediately at the step boundary rather than producing silent wrong output downstream.

| | LangGraph | LlamaIndex Workflows |
|---|---|---|
| **Best for** | Agentic loops, branching, conditional routing | Linear deterministic pipelines |
| **Data passing** | Shared mutable `AgentState` dict | Typed Pydantic events per step |
| **Schema enforcement** | Optional | Enforced at each step boundary |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LlamaIndex Workflows |
| Diagnostic Model | PyTorch (custom BKTransformer) |
| LLM | OpenAI GPT-4o-mini (or self-hosted via vLLM) |
| Knowledge Graph | Neo4j Aura |
| Observability | Langfuse |
| Services | FastAPI + uvicorn |
| Deployment | Docker Compose (local) · Cloud Run (production) |

---

## Project Structure

```
src/ai_tutor/
├── agents/
│   ├── graph.py              # LlamaIndex TutorWorkflow definition
│   ├── state.py              # AgentState TypedDict + event types
│   ├── schemas.py            # Pydantic models (BKTTimestep, AnalysisRecord, FeedbackRecord)
│   ├── diagnosis_node.py     # BKT inference + LLM proficiency diagnosis
│   └── recommendation_node.py# GraphRAG + LLM feedback generation
├── bkt/
│   ├── model.py              # BKTransformer (RoPE, SwiGLU)
│   └── config.py             # BKT hyperparameters
├── tools/
│   └── neo4j_tool.py         # Direct Neo4j query wrapper
├── llm_client.py             # LLM backend factory (openai | vllm)
└── api.py                    # FastAPI orchestrator entry point

services/
├── bkt_service.py            # BKTransformer inference microservice (port 8001)
├── Dockerfile.bkt
├── neo4j_service.py          # Legacy Neo4j Cypher microservice (not used by current app path)
└── Dockerfile.neo4j_svc
```

---

## How to Run

### Local — Docker Compose

1. Copy and fill in credentials:
   ```bash
   cp .env.example .env
   # edit .env with your API keys and Neo4j connection details
   ```

2. Start all services (OpenAI backend):
   ```bash
   LLM_BACKEND=openai LLM_MODEL=gpt-4o-mini docker compose up --build
   ```

3. Start with local vLLM instead (requires NVIDIA GPU):
   ```bash
   LLM_BACKEND=vllm LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
     docker compose --profile vllm up --build
   ```

4. Call the API:
   ```bash
   curl -X POST localhost:8000/tutor \
     -H "Content-Type: application/json" \
     -d '{"student_id": "s1", "sequence": [[1,1],[2,0],[1,1]], "skill_id_to_name": {"1": "순환소수", "2": "유리수"}}'
   ```

---

## Observability

All pipeline runs are traced in Langfuse with the following span hierarchy:

```
tutor_pipeline  [trace]
  ├── run_bkt      — BKT parameters per timestep
  ├── diagnose     — LLM generation: proficiency level + reasoning per skill
  └── recommend    — LLM generation per skill: graph context + feedback
```

Note: Langfuse tracing is active only when `LLM_BACKEND=openai`. The vLLM path uses the raw OpenAI-compatible client without a Langfuse wrapper.

**1. Pipeline Trace**
![Pipeline Trace](./assets/trace-tree.png)

**2. BKT Parameter-Based Proficiency Diagnosis**

[BKT Parameters as Input]
![Diagnosis Node Input](./assets/diagnose-input.png)

[Agent's Diagnosis as Output]
![Diagnosis Node](./assets/diagnose-output.png)

**3. Agentic GraphRAG Personalized Feedback Generation**
![Recommendation Node](./assets/recommend.png)

---

## References

**This work:**
> 손준호 (2026). TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성. 서울시립대학교 일반대학원 석사학위논문. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

**Original BKTransformer:**
> Badrinath, A., & Pardos, Z. (2025). Optimizing Bayesian Knowledge Tracing with Neural Network Parameter Generation. *Journal of Educational Data Mining*, 17(1), 41–65.

**RAG Survey:**
> Gao, Y., et al. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.

---

## Dataset
- **AI-Hub** — 수학분야 학습자 역량 측정 데이터, https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=133
