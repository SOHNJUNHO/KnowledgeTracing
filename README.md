# Agentic GraphRAG Tutor

LLM 환각을 구조적으로 제거한 맞춤형 AI 수학 튜터 시스템입니다.
신경망 BKT 모델(BKTransformer)로 학습 상태를 진단하고, 지식 그래프(Neo4j)를 기반으로 한 Agentic GraphRAG 파이프라인이 개인화된 학습 피드백을 생성합니다.

석사 학위 논문 기반: **TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성** — 손준호, 서울시립대학교, 2026. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

---

## 왜 만들었는가

LLM은 강력한 피드백 생성기이지만, 교육 환경에서 세 가지 근본적인 문제를 가집니다.

| 문제 | 설명 |
|---|---|
| **환각(Hallucination)** | 사실과 다른 피드백을 자신 있게 생성 |
| **관계 무시** | 교육과정 내 지식 개념 간 선후 관계를 반영하지 못함 |
| **불투명성** | 왜 그 응답을 했는지 근거를 설명하지 못함 |

이 시스템은 세 문제를 구조적으로 제거합니다.

- **투명한 지식 상태 추출**: BKTransformer는 심층 신경망이 블랙박스로 처리하는 것과 달리, 베이즈 정리 기반의 BKT 파라미터(P(know), P(learn), P(guess), P(slip))를 명시적으로 출력합니다. 학생이 개념을 얼마나 알고 있는지, 왜 그렇게 판단하는지가 수치로 드러납니다.
- **해석 가능한 진단**: 진단 에이전트는 BKT 파라미터를 논리적 근거로 사용해 학생의 숙련도(상/중/하)를 결정합니다. 모델의 추론 과정이 자연어로 설명됩니다.
- **환각 없는 개인화 추천**: 추천 에이전트는 Neo4j 지식 그래프에서 교육과정 관계를 조회한 뒤 피드백을 생성합니다. LLM이 임의로 개념을 만들어낼 여지가 없습니다.

---

## 아키텍처

```
학생 학습 이력 (skill_id, correct) × T 타임스텝
                        │
                        ▼
         ┌──────────────────────────┐
         │   BKTransformer (PyTorch)│
         │       RoPE · SwiGLU      │
         └──────────────────────────┘
                        │
          스킬별 BKT 파라미터:
          P(know), P(learn), P(guess), P(slip)
                        │
                        ▼
         ┌──────────────────────────┐
         │      진단 에이전트        │
         │       (GPT-4o-mini)      │
         └──────────────────────────┘
                        │
              숙련도 레벨 (상 / 중 / 하)
                        │
                        ▼
         ┌──────────────────────────┐
         │      추천 에이전트        │
         │       (GPT-4o-mini)      │
         │    Neo4j GraphRAG        │
         │   사전 정의된 Cypher 쿼리 │
         └──────────────────────────┘
                        │
                        ▼
             개인화된 학습 피드백
```

**LangGraph 흐름:** `START → run_bkt → diagnose → recommend → END`

**숙련도별 Cypher 쿼리 선택:**
| 레벨 | 쿼리 | 목적 |
|---|---|---|
| 하 (Low) | `get_prerequisites` | 먼저 학습해야 할 선행 개념 조회 |
| 중 (Mid) | `get_current_concept` | 현재 개념의 심화 학습 유도 |
| 상 (High) | `get_advanced_concepts` | 다음 단계로 도전할 수 있는 심화 개념 조회 |

---

## 기술 스택

| 레이어 | 기술 |
|---|---|
| 오케스트레이션 | LangGraph |
| 진단 모델 | PyTorch (BKTransformer) |
| LLM | OpenAI GPT-4o-mini |
| 지식 그래프 | Neo4j Aura |
| 그래프 접근 | Neo4j 비동기 드라이버 |
| 관찰 가능성 | Langfuse |

---

## 핵심 설계 결정

이 프로젝트를 진행하면서 여러 기술적 선택지를 직접 실험하고 검토했습니다. 아래에 그 판단 과정을 기록합니다.

### 1. ONNX를 사용하지 않은 이유

BKTransformer 추론 파이프라인에서 ONNX 런타임 도입을 검토했습니다.

ONNX는 텐서 연산 그래프를 최적화할 때 효과적입니다. 그러나 BKT의 핵심 연산은 텐서 행렬 계산이 아닌, 타임스텝마다 반복되는 **스칼라 단위의 베이즈 업데이트**입니다. 즉, 병렬화할 텐서 그래프 자체가 없습니다.

ONNX를 적용해도 속도 이점이 없는 반면, 직렬화 포맷 관리와 버전 호환성 문제가 발생했습니다. 따라서 PyTorch 추론을 그대로 유지했습니다.

### 2. MCP(Google GenAI Toolbox)를 사용하지 않은 이유

초기에는 Google GenAI Toolbox를 MCP 서버로 운영하고 `tools.yaml`에 Cypher 쿼리를 정의하는 방식을 사용했습니다(`langgraph_toolbox` 브랜치).

문제는 **버전 관리**였습니다. 애플리케이션 코드와 별도로 동작하는 툴박스 서버의 버전을 동기화하는 것이 개발 과정에서 지속적인 부담이 됐습니다. Cypher 쿼리를 수정할 때마다 `tools.yaml`과 코드, 서버를 모두 일치시켜야 했고, 이는 불필요한 운영 복잡도였습니다.

해결책은 단순했습니다. Cypher 쿼리를 코드 안에 직접 정의하고 Neo4j 비동기 드라이버로 직접 호출합니다. 동일한 기능을 별도 서버 없이 달성할 수 있었고, 코드베이스도 단순해졌습니다.

### 3. LlamaIndex와 LangGraph 비교

`llamaindex-experiment` 브랜치에서 LlamaIndex Workflows로 동일한 파이프라인을 구현했습니다.

**LlamaIndex의 장점**: 각 스텝이 명시적인 타입의 입력/출력 `Event`를 선언합니다. 노드 간 데이터 계약이 코드에 드러나고, 하나의 공유 상태 딕셔너리를 모든 노드가 공유하지 않기 때문에 각 노드를 독립적으로 테스트하기 쉬웠습니다.

**LangGraph를 선택한 이유**: LangGraph는 현재 프로덕션 AI 파이프라인 오케스트레이션의 업계 표준에 가깝습니다. Langfuse를 포함한 주요 관찰 가능성 도구들과의 통합이 잘 되어 있고, 커뮤니티와 레퍼런스가 풍부합니다. 공유 `AgentState`의 디버깅 복잡도는 명확한 타입 정의와 Langfuse 트레이싱으로 관리할 수 있었습니다.

### 4. Langfuse를 선택한 이유

단순한 로깅이 아닌 **LLM 파이프라인 전용 관찰 가능성**이 필요했습니다.

Langfuse는 세 가지 이유로 선택했습니다. 첫째, **프롬프트 버전 관리**입니다. 진단 프롬프트와 피드백 프롬프트를 Langfuse Prompt Registry에서 관리하므로, 코드 배포 없이 프롬프트를 수정하고 이전 버전과 성능을 비교할 수 있습니다. 둘째, **엔드-투-엔드 트레이싱**입니다. BKT 출력, LLM 입력/출력, 그래프 컨텍스트, 최종 피드백까지 하나의 트레이스로 연결되어 파이프라인 어느 단계에서 문제가 발생했는지 즉시 파악할 수 있습니다. 셋째, **인간 평가 지원**입니다. 교육 전문가가 Langfuse UI에서 생성된 피드백의 품질을 직접 평가하고 점수를 부여할 수 있어, 반복적인 프롬프트 개선이 가능합니다.

### 5. Cypher 쿼리 결정을 에이전트에게 맡기지 않은 이유

추천 에이전트가 어떤 Cypher 쿼리를 실행할지 스스로 결정하지 않습니다. 진단 에이전트의 숙련도 레벨(상/중/하)이 쿼리를 결정합니다.

LLM이 도구 선택까지 담당하면 그래프 조회 단계의 투명성이 사라집니다. 사전 정의된 쿼리 선택은 GraphRAG의 검색 단계를 감사 가능하게(auditable) 만들고, 예측 불가능한 동작을 차단합니다.

---

## 프로젝트 구조

```
src/
├── ai_tutor/
│   ├── agents/
│   │   ├── graph.py              # LangGraph StateGraph 정의
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── diagnosis_node.py     # BKT 추론 + LLM 숙련도 진단
│   │   └── recommendation_node.py# GraphRAG + LLM 피드백 생성
│   ├── bkt/
│   │   └── model.py              # BKTransformer (RoPE, SwiGLU)
│   ├── tools/
│   │   └── neo4j_tool.py         # Neo4j 비동기 드라이버 + Cypher 쿼리
│   └── api.py                    # FastAPI HTTP 서비스
└── bkt_service/
    └── app.py                    # 독립 BKT 추론 서비스
```

---

## 로컬 실행 방법

1. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

2. `.env` 파일 생성:
   ```
   OPENAI_API_KEY=...
   LANGFUSE_HOST=...
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_SECRET_KEY=...
   NEO4J_URI=...
   NEO4J_USERNAME=...
   NEO4J_PASSWORD=...
   BKT_CHECKPOINT=src/ai_tutor/bkt/checkpoints/<checkpoint>.pt
   N_SKILLS=138
   ```

3. Langfuse 프롬프트 등록 (최초 1회):
   ```bash
   python scripts/create_prompts.py
   ```

4. 파이프라인 실행:
   ```bash
   python -m ai_tutor.main
   ```

   또는 HTTP API 서버 시작:
   ```bash
   uvicorn ai_tutor.api:app --port 8000 --reload
   ```

---

## 관찰 가능성

모든 파이프라인 실행은 Langfuse에서 다음 계층 구조로 추적됩니다.

```
tutor_pipeline  [trace]
  ├── run_bkt      — 타임스텝별 BKT 파라미터
  ├── diagnose     — LLM 생성: 자연어 진단, 숙련도 레벨
  └── recommend    — 스킬별 LLM 생성: 그래프 컨텍스트 + 피드백
```

**1. LangGraph 오케스트레이션 파이프라인**
- 노드별 실행 시간, 스팬 계층, 토큰 비용
![Pipeline Trace](./assets/trace-tree.png)

**2. BKT 파라미터 기반 숙련도 진단**
- LLM이 정답/오답 이진값이 아닌 BKT 파라미터 수치를 논리적 근거로 숙련도를 판단

**[진단 입력: BKT 파라미터]**
![Diagnosis Node Input](./assets/diagnose-input.png)

**[진단 출력: 에이전트 분석]**
![Diagnosis Node](./assets/diagnose-output.png)

**3. Agentic GraphRAG 개인화 피드백 생성**
![Recommendation Node](./assets/recommend.png)
*(진단된 숙련도를 기반으로 지식 그래프에서 관련 개념을 조회한 뒤, 최종 개인화 피드백 생성)*

---

## 참고 문헌

**본 연구:**
> 손준호 (2026). TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성. 서울시립대학교 일반대학원 석사학위논문. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

**BKTransformer 원논문:**
> Badrinath, A., & Pardos, Z. (2025). Optimizing Bayesian Knowledge Tracing with Neural Network Parameter Generation. *Journal of Educational Data Mining*, 17(1), 41–65.

**RAG Survey:**
> Gao, Y., et al. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.

---

## 데이터셋
- **AI-Hub** — 수학분야 학습자 역량 측정 데이터, https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=133
