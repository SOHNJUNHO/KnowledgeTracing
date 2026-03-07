# Agentic GraphRAG Tutor

A production-grade AI tutoring system that combines a custom Neural Bayesian Knowledge Tracing model with an Agentic GraphRAG pipeline to deliver personalized, explainable study recommendations.

Based on the master's thesis: **TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성** — Junho Son, University of Seoul, 2026. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

---

## Why This Exists

LLMs are powerful feedback generators, but they have three fundamental problems in educational settings:

1. **Hallucination** — they produce inconsistent or factually wrong feedback
2. **Omitted Relationships** — they cannot reflect the curriculum, the relationships between knowledge concepts
3. **Opacity** — they cannot explain *why* they gave a particular response

This project addresses all three with a step-by-step approach:
- **Transparent parameter extraction**: Unlike the black-box computations of conventional deep learning KT models, BKTransformer outputs BKT parameters (P(know), P(learn), P(guess), P(slip)) grounded in Bayes' theorem, making the knowledge state transparent.
- **Interpretable diagnosis**: The diagnosis agent logically evaluates the extracted BKT parameters to determine a student's proficiency level (High / Mid / Low) per knowledge concept.
- **Hallucination-free personalized recommendation**: Based on the diagnosis, the recommendation agent queries a knowledge graph (Neo4j) that encodes curriculum relationships, producing accurate, grounded feedback without LLM hallucination.

---

## Architecture

```
Student Interaction History (skill_id, correct) × T timesteps
                        │
                        ▼
         ┌──────────────────────────┐
         │   BKTransformer (PyTorch)│
         │       RoPE · SwiGLU      │
         └──────────────────────────┘
                        │
          Per-skill BKT Parameters:
          P(know), P(learn), P(guess), P(slip)
                        │
                        ▼
         ┌──────────────────────────┐
         │     Diagnosis Agent      │
         │       GPT-4o-mini        │
         └──────────────────────────┘
                        │
              Proficiency Level (상 / 중 / 하)
                        │
                        ▼
         ┌──────────────────────────┐
         │   Recommendation Agent   │
         │   Neo4j GraphRAG (MCP)   │
         │   Predefined Cypher Query│
         │   GPT-4o-mini            │
         └──────────────────────────┘
                        │
                        ▼
            Personalized Study Feedback
```

**LangGraph Flow:** `START → run_bkt → diagnose → recommend → END`

**Cypher Query Selection by Proficiency:**
| Level | Query | Purpose |
|---|---|---|
| 하 (Low) | `get_prerequisites` | Retrieve prerequisite concepts to study first |
| 중 (Mid) | `get_current_concept` | Reinforce the current concept |
| 상 (High) | `get_advanced_concepts` | Surface next concepts the student is ready for |

---

## Key Design Decisions

**1. BKT Parameters as LLM Input**
Unlike conventional deep learning KT models that output proficiency directly through uninterpretable neural computations, BKTransformer derives proficiency by applying Bayes' theorem to explicit BKT parameters. This provides transparency into how and why each correctness probability was produced.

**2. Predefined Tool Selection**
The recommendation agent does not decide which Cypher query to run. The proficiency level from the diagnosis agent determines the query. This guarantees transparency in the GraphRAG retrieval step.

**3. MCP for Graph Access**
Neo4j is accessed via Google GenAI Toolbox (MCP pattern) rather than a direct driver. This decouples the graph interface from the agent code and makes the tool layer independently testable and replaceable.

**4. Langfuse for Full Observability**
Every pipeline run is traced end-to-end: BKT output, LLM prompts/responses, graph context, and final feedback. This supports human expert evaluation and iterative prompt improvement.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Diagnostic Model | PyTorch (custom BKTransformer) |
| LLM | OpenAI GPT-4o-mini |
| Knowledge Graph | Neo4j Aura |
| Graph Access | Google GenAI Toolbox (MCP) |
| Observability | Langfuse |

---

## Project Structure

```
src/
├── agents/
│   ├── graph.py              # LangGraph StateGraph definition
│   ├── state.py              # AgentState TypedDict
│   ├── diagnosis_node.py     # BKT inference + LLM proficiency diagnosis
│   └── recommendation_node.py# GraphRAG + LLM feedback generation
├── bkt/
│   └── model.py              # BKTransformer (RoPE, SwiGLU)
├── tools/
│   └── neo4j_tool.py         # MCP toolbox client wrapper
└── run.py                    # CLI entry point

tools.yaml                    # Predefined Cypher queries (MCP tool definitions)
```

---

## How to Run Locally

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your credentials:
   ```
   OPENAI_API_KEY=...
   LANGFUSE_HOST=...
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_SECRET_KEY=...
   NEO4J_URI=...
   NEO4J_USERNAME=...
   NEO4J_PASSWORD=...
   TOOLBOX_URL=http://localhost:5001
   ```

3. Start the Google GenAI Toolbox server:
   ```bash
   toolbox --tools-file tools.yaml --port 5001
   ```

4. Run the pipeline:
   ```bash
   python src/run.py
   ```

---

## Observability

All pipeline runs are traced in Langfuse with the following span hierarchy:

```
tutor_pipeline  [trace]
  ├── run_bkt      — BKT parameters per timestep
  ├── diagnose     — LLM generation: diagnosis in natural language, proficiency
  └── recommend    — LLM generation per skill: graph context + feedback
```

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
