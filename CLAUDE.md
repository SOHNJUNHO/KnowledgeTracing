# Project Manifesto: Agentic GraphRAG Tutor

## 1. Context & Background
- **Developer Background:** Research-focused (Thesis) transitioning to AI Agent Engineering.
- **Goal:** Build a production-ready system that aligns with  2026 Agentic patterns.
- **Legacy Code:** Old training scripts (`.py`) and messy GraphRAG experiments (`.ipynb`).

## 2. The New Architecture (The Target)
- **Framework:** LangGraph (State-based orchestration).
- **Brain:** Custom PyTorch BKT model (Diagnostic).
- **Data Layer:** Google GenAI Toolbox (MCP) + Neo4j Aura (Cloud).
- **Observability:** Langfuse (OTEL-based).

## 3. High-Priority Missions
1.  **Extract:** Move the BKT model logic from the old scripts into a clean `src/models/` directory.
2.  **Refactor:** Turn the `.ipynb` GraphRAG logic into formal LangGraph nodes.
3.  **Modernize:** Replace direct Neo4j drivers with the MCP Toolbox pattern.
