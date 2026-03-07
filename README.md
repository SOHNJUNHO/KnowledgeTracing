# Agentic GraphRAG Tutor

> A production-grade AI tutoring system that combines a custom Neural Bayesian Knowledge Tracing model with an Agentic GraphRAG pipeline to deliver personalized, explainable study recommendations.

---

## Why This Exists

Most AI tutoring systems treat student performance as a black box — feeding raw interaction logs into an LLM and hoping for useful output. This project takes a different approach.

A custom **Neural BKT Transformer** first processes a student's interaction history and outputs interpretable cognitive parameters per skill: latent knowledge probability, learning rate, guess rate, and slip rate. These parameters — not raw data — are handed to an LLM agent. The agent then reasons over them to diagnose proficiency, queries a **Neo4j knowledge graph** for curriculum context, and generates a grounded study recommendation.

The result is a system where every output is traceable to a theoretical model of learning.

---

## Architecture

```
Student Interaction History (skill_id, correct) × T timesteps
                        │
                        ▼
         ┌──────────────────────────┐
         │   BKTransformer (PyTorch)│
         │   RoPE · SwiGLU · Causal │
         │   Attention · 3 Layers   │
         └──────────────────────────┘
                        │
          Per-skill BKT Parameters:
          P(know), P(learn), P(guess), P(slip)
                        │
                        ▼
         ┌──────────────────────────┐
         │     Diagnosis Agent      │
         │     GPT-4o-mini          │
         │     Proficiency: 상/중/하 │
         └──────────────────────────┘
                        │
              Proficiency Level
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

**1. BKT Parameters as LLM Input, Not Raw Logs**
Passing `prior=0.82, slip=0.31` to the LLM is both more compact and more meaningful than a sequence of 0s and 1s. The LLM receives a cognitive fingerprint, not a data dump. This is what makes the diagnosis explainable and theoretically grounded.

**2. Deterministic Tool Selection**
The recommendation agent does not decide which Cypher query to run — the proficiency level from the diagnosis node deterministically selects it. This keeps the graph traversal predictable and auditable, which matters in educational contexts.

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
│   └── model.py              # BKTransformer (RoPE, SwiGLU, causal attn)
├── tools/
│   └── neo4j_tool.py         # MCP toolbox client wrapper
└── run.py                    # CLI entry point

tools.yaml                    # Predefined Cypher queries (MCP tool definitions)
```

---

## Observability

All pipeline runs are traced in Langfuse with the following span hierarchy:

```
tutor_pipeline  [trace]
  ├── run_bkt      — BKT parameters per timestep, device metadata
  ├── diagnose     — LLM generation: prompt + proficiency output
  └── recommend    — LLM generation per skill: graph context + feedback
```

---

## Related

- **Paper:** *(link to paper)*
- **Dataset:** ASSISTments 2009 (skill_builder), Korean math curriculum (icecream 8th grade)
