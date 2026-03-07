# Agentic GraphRAG 튜터

> 커스텀 Neural Bayesian Knowledge Tracing 모델과 Agentic GraphRAG 파이프라인을 결합하여, 개인화되고 설명 가능한 학습 추천을 제공하는 프로덕션 수준의 AI 튜터링 시스템입니다.

---

## 개발 배경

대부분의 AI 튜터링 시스템은 학생의 학습 이력을 블랙박스처럼 다룹니다. 원시 상호작용 로그를 LLM에 그대로 넣고 유용한 출력을 기대하는 방식입니다. 이 프로젝트는 다른 접근법을 택합니다.

커스텀 **Neural BKT Transformer**가 학생의 문제 풀이 기록을 처리하여, 각 지식 요소(Knowledge Component)별로 해석 가능한 인지 파라미터를 출력합니다: 잠재 지식 확률(prior), 학습률(learning rate), 추측 확률(guess), 실수 확률(slip). 원시 데이터가 아닌 이 파라미터들이 LLM 에이전트에 전달됩니다. 에이전트는 이를 바탕으로 숙련도를 진단하고, **Neo4j 지식 그래프**에서 교육과정 맥락을 조회한 뒤, 근거 있는 학습 추천을 생성합니다.

모든 출력이 학습의 이론적 모델(BKT)로 추적 가능한 시스템입니다.

---

## 아키텍처

```
학생 상호작용 기록 (skill_id, correct) × T 타임스텝
                        │
                        ▼
         ┌──────────────────────────┐
         │  BKTransformer (PyTorch)  │
         │  RoPE · SwiGLU · Causal   │
         │  Attention · 3 Layers     │
         └──────────────────────────┘
                        │
          지식 요소별 BKT 파라미터:
          P(know), P(learn), P(guess), P(slip)
                        │
                        ▼
         ┌──────────────────────────┐
         │      진단 에이전트         │
         │      GPT-4o-mini          │
         │      숙련도: 상 / 중 / 하  │
         └──────────────────────────┘
                        │
                   숙련도 수준
                        │
                        ▼
         ┌──────────────────────────┐
         │      추천 에이전트         │
         │  Neo4j GraphRAG (MCP)     │
         │  사전 정의된 Cypher 쿼리   │
         │  GPT-4o-mini              │
         └──────────────────────────┘
                        │
                        ▼
               개인화된 학습 피드백
```

**LangGraph 흐름:** `START → run_bkt → diagnose → recommend → END`

**숙련도별 Cypher 쿼리 선택:**
| 수준 | 쿼리 | 목적 |
|---|---|---|
| 하 (Low) | `get_prerequisites` | 먼저 학습해야 할 선행 개념 조회 |
| 중 (Mid) | `get_current_concept` | 현재 개념 심화 학습 |
| 상 (High) | `get_advanced_concepts` | 학습 가능한 다음 연계 개념 조회 |

---

## 핵심 설계 결정

**1. 원시 로그가 아닌 BKT 파라미터를 LLM에 전달**
`prior=0.82, slip=0.31`과 같은 파라미터를 LLM에 전달하는 것은 0과 1의 나열보다 훨씬 정보 밀도가 높고 의미 있습니다. LLM은 단순한 데이터 덤프가 아닌 인지적 지문(cognitive fingerprint)을 받습니다. 이것이 진단을 설명 가능하고 이론적으로 근거 있게 만드는 핵심입니다.

**2. 결정론적 도구 선택**
추천 에이전트는 어떤 Cypher 쿼리를 실행할지 스스로 결정하지 않습니다. 진단 노드의 숙련도 수준이 쿼리를 결정론적으로 선택합니다. 이는 그래프 탐색을 예측 가능하고 감사 가능하게 유지하며, 교육 맥락에서 중요한 특성입니다.

**3. MCP를 통한 그래프 접근**
Neo4j는 직접 드라이버 대신 Google GenAI Toolbox(MCP 패턴)를 통해 접근합니다. 이는 그래프 인터페이스를 에이전트 코드로부터 분리하여 도구 레이어를 독립적으로 테스트하고 교체 가능하게 합니다.

**4. Langfuse를 통한 완전한 가시성**
모든 파이프라인 실행은 엔드투엔드로 추적됩니다: BKT 출력, LLM 프롬프트/응답, 그래프 컨텍스트, 최종 피드백. 이는 전문가 평가와 반복적인 프롬프트 개선을 지원합니다.

---

## 기술 스택

| 레이어 | 기술 |
|---|---|
| 오케스트레이션 | LangGraph |
| 진단 모델 | PyTorch (커스텀 BKTransformer) |
| LLM | OpenAI GPT-4o-mini |
| 지식 그래프 | Neo4j Aura |
| 그래프 접근 | Google GenAI Toolbox (MCP) |
| 관찰 가능성 | Langfuse |

---

## 프로젝트 구조

```
src/
├── agents/
│   ├── graph.py              # LangGraph StateGraph 정의
│   ├── state.py              # AgentState TypedDict
│   ├── diagnosis_node.py     # BKT 추론 + LLM 숙련도 진단
│   └── recommendation_node.py# GraphRAG + LLM 피드백 생성
├── bkt/
│   └── model.py              # BKTransformer (RoPE, SwiGLU, 인과적 어텐션)
├── tools/
│   └── neo4j_tool.py         # MCP 툴박스 클라이언트 래퍼
└── run.py                    # CLI 진입점

tools.yaml                    # 사전 정의된 Cypher 쿼리 (MCP 도구 정의)
```

---

## 관찰 가능성

모든 파이프라인 실행은 다음 스팬 계층으로 Langfuse에 추적됩니다:

```
tutor_pipeline  [trace]
  ├── run_bkt      — 타임스텝별 BKT 파라미터, 디바이스 메타데이터
  ├── diagnose     — LLM 생성: 프롬프트 + 숙련도 출력
  └── recommend    — 지식 요소별 LLM 생성: 그래프 컨텍스트 + 피드백
```

---

## 관련 자료

- **논문:** *(논문 링크)*
- **데이터셋:** ASSISTments 2009 (skill_builder), 한국 수학 교육과정 (아이스크림 중2)
