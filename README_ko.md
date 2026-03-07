# Agentic GraphRAG 튜터

> Neural Bayesian Knowledge Tracing 모델과 Agentic GraphRAG 파이프라인을 결합하여, 설명 가능한 개인별 학습 추천을 제공하는 AI 튜터링 시스템입니다.
>
> 석사학위논문 기반: **TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성** — 손준호, 서울시립대학교 일반대학원, 2026. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

---

## 개발 배경

LLM은 강력한 피드백 생성 도구이지만, 세 가지 근본적인 문제가 있습니다:

1. **환각(Hallucination)** — 비일관적이거나 부정확한 피드백을 생성할 수 있음
2. **관계에 대한 데이터 부재** — 교육과정에 따른 지식 개념 간의 연관 관계를 반영하기 어려움 
3. **불투명성** — 특정 응답이 왜 생성되었는지 설명할 수 없음

 
이 프로젝트는 다음과 같은 단계적 접근으로 위 세 가지 문제를 모두 해결합니다:
- 투명한 파라미터 추출: 기존 딥러닝 KT 모델의 블랙박스 연산과 달리, BKTransformer는 베이즈 정리 기반의 BKT 파라미터(P(know), P(learn), P(guess), P(slip))를 산출하여 학습 상태를 투명하게 모델링합니다.
- 해석 가능한 진단: 진단 에이전트는 산출된 BKT 파라미터를 기반으로 학생의 지식 개념 숙련도(상/중/하)를 논리적으로 진단합니다.
- 환각 없는 맞춤형 추천: 추천 에이전트는 진단 결과를 바탕으로 교육 과정의 연관 관계가 정의된 지식 그래프(Neo4j)를 검색하여, LLM의 환각 없이 정확하고 근거 있는 맞춤형 피드백을 생성합니다.

---

## 아키텍처

```
    학생 상호작용 기록 (skill_id, correct) × T 타임스텝
                        │
                        ▼
         ┌──────────────────────────┐
         │   BKTransformer (PyTorch)│
         │       RoPE · SwiGLU      │
         └──────────────────────────┘
                        │
          지식 요소별 BKT 파라미터:
          P(know), P(learn), P(guess), P(slip)
                        │
                        ▼
         ┌──────────────────────────┐
         │       진단 에이전트         │
         │       GPT-4o-mini        │
         └──────────────────────────┘
                        │
              Proficiency Level(상 / 중 / 하)
                        │
                        ▼
         ┌──────────────────────────┐
         │       추천 에이전트         │
         │   Neo4j GraphRAG (MCP)   │
         │   사전 정의된 Cypher 쿼리    │
         │   GPT-4o-mini            │
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

**1. BKT 파라미터를 LLM에 전달**
기존의 딥러닝 KT(Knowledge Tracing) 모델이 해석 불가능한 신경망 계산을 통해 곧바로 숙련도를 산출하는 것과 달리 산출된 BKT 파라미터를 기반으로 베이즈 정리를 활용하여 숙련도를 산출합니다. 이로써 정답 확률이 어떤 과정과 의미를 거쳐 산출되었는지 분석하고 해석할 수 있는 투명성을 제공합니다.

**2. 사전정의된 도구 선택**
추천 에이전트는 어떤 Cypher 쿼리를 실행할지 스스로 결정하지 않습니다. 진단 에이전트의 숙련도 수준이 쿼리를 결정합니다. 이는 GraphRAG의 투명성을 보장합니다.

**3. MCP를 통한 그래프 접근**
Neo4j는 직접 드라이버 대신 Google GenAI Toolbox(MCP 패턴)를 통해 접근합니다. 이는 그래프 인터페이스를 에이전트 코드로부터 분리하여 도구 레이어를 독립적으로 테스트하고 교체 가능하게 합니다.

**4. Langfuse를 통한 완전한 가시성**
모든 파이프라인 실행은 추적됩니다. 이는 전문가 평가와 반복적인 프롬프트 개선을 지원합니다.

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

## 🚀 로컬 실행 방법 (How to Run Locally)

1. 저장소를 클론하고 의존성 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

2. `.env` 파일에 OpenAI, Langfuse, Neo4j Aura 크레덴셜을 설정합니다.

3. Google GenAI Toolbox 서버를 실행합니다:
   ```bash
   toolbox --tools-file tools.yaml --port 5001
   ```

4. 파이프라인을 실행합니다:
   ```bash
   python src/run.py
   ```

---

## 관찰 가능성

모든 파이프라인 실행은 다음 스팬 계층으로 Langfuse에 추적됩니다:

```
tutor_pipeline  [trace]
  ├── run_bkt      — 매 시점별 BKT 파라미터
  ├── diagnose     — BKT 파리미터에 대한 자연어 해석과 각 지식개념별 숙련도 출력
  └── recommend    — 지식 개념별 자연어 피드백 생성s
```

**1. LangGraph 오케스트레이션 파이프라인**
- 각 노드의 실행 시간, 통신 계층 구조 및 토큰 사용 비용
![파이프라인 트레이스](./assets/trace-tree.png)

**2. BKT 파라미터 기반 숙련도 진단**
- 단순 정오답이 아닌 BKT 파라미터 수치를 논리적 근거로 삼아 LLM이 숙련도(상/중/하)를 진단하는 과정
**[Input: 추출된 BKT 파라미터]**
![진단 노드 입력](./assets/diagnose-input.png)

**[Output: 진단 에이전트의 숙련도 진단 및 추론]**
![진단 노드 출력](./assets/diagnose-output.png)

**3. Agentic GraphRAG 맞춤형 학습 피드백 생성**
- 진단된 숙련도를 바탕으로 지식 그래프에서 연계 개념을 검색하고, 이를 바탕으로 최종 생성된 맞춤형 피드백
![추천 노드](./assets/recommend.png)



---

## 참고 문헌

**본 논문:**
> 손준호 (2026). TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성. 서울시립대학교 일반대학원 석사학위논문. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

**원본 BKTransformer:**
> Badrinath, A., & Pardos, Z. (2025). Optimizing Bayesian Knowledge Tracing with Neural Network Parameter Generation. *Journal of Educational Data Mining*, 17(1), 41–65.

**GraphRAG 참고 논문:**
> Gao, Y., et al. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.

---

## 데이터셋
- **AI-Hub** — 수학분야 학습자 역량 측정 데이터, https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=133
