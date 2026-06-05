# Agentic GraphRAG Tutor

LLM 환각을 최소화하는 AI 튜터링 시스템입니다.
신경망 BKT 모델(BKTransformer)로 학습 상태를 진단하고, Neo4j 지식 그래프 기반 Agentic GraphRAG 파이프라인이 개인화된 학습 피드백을 생성합니다.

석사 학위 논문 기반: **TutorAgent: BKTransformer 기반 지식 추적과 Agentic GraphRAG를 통한 맞춤형 피드백 생성** — 손준호, 서울시립대학교, 2026. [[RISS]](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8e965dc55df10271ffe0bdc3ef48d419)

> **로컬 개발 브랜치** (`llamaindex-experiment`): LlamaIndex Workflows + Ollama Qwen 3 8B. 배포 파일 없음. AWS 프로덕션은 `production-readiness` 브랜치를 참조하세요.

---

## 왜 만들었는가

LLM의 고질적 환각 현상은 교육 현장 적용을 저해하는 주 요인입니다.
교육에서는 정확도만큼이나 설명 가능성(explainability)이 중요하기 때문입니다.
이 시스템은 설명 가능성 극대화와 비용·지연 시간 최소화를 목표로 설계됐습니다.

- **투명한 지식 상태 추출**: BKTransformer는 신경망을 이용하지만, 다른 딥러닝 KT 모델과 달리 설명할 수 없는 로짓값만 리턴하지 않습니다. 고전적인 베이즈 정리 기반 BKT 구조를 차용해 BKT 파라미터(P(know), P(learn), P(guess), P(slip))를 명시적으로 출력합니다. 원조 BKTransformer에 RoPE를 적용하여 파라미터를 3.2M → 2.9M으로 줄이고, AUC를 0.78 → 0.81로 향상했습니다.
- **해석 가능한 진단**: 진단 에이전트는 BKT 파라미터를 근거로 사용해 학생의 숙련도(상/중/하)를 결정합니다. 모델의 추론 과정이 자연어로 설명됩니다.
- **환각 없는 개인화 추천**: 추천 에이전트는 Neo4j 지식 그래프에서 사전 정의된 Cypher 쿼리로 개념 간 관계를 조회한 후 피드백을 생성합니다. LLM의 임의적, 불규칙적인 행동을 원천적으로 차단합니다.

---

## 아키텍처

```
학생 학습 이력 (skill_id, correct) × T timesteps
                        │
                        ▼
         ┌────────────────────────────┐
         │   BKTransformer (PyTorch)  │  ← FastAPI 프로세스 내 in-process 실행
         │       RoPE  ·  SwiGLU      │    (torch.compile 적용, 시작 시 워밍업)
         └──────────────┬─────────────┘
                        │
         Per-skill BKT parameters:
         P(know), P(learn), P(guess), P(slip)
                        │
                        ▼
         ┌────────────────────────────┐
         │      Diagnosis Agent       │
         │     (Ollama Qwen 3 8B)     │
         └──────────────┬─────────────┘
                        │
         Proficiency level: 상 / 중 / 하
                        │
                        ▼
         ┌────────────────────────────┐
         │    Recommendation Agent    │
         │     (Ollama Qwen 3 8B)     │
         │      Neo4j  GraphRAG       │
         │   Predefined Cypher Query  │
         └────────────────────────────┘
                        │
                        ▼
           Personalized study feedback
```

**LlamaIndex Workflow 흐름:**
`StartEvent → run_bkt → BKTDoneEvent → diagnose → DiagnosisDoneEvent → recommend → StopEvent`

**숙련도별 Cypher 쿼리 선택:**
| 레벨 | 쿼리 | 목적 |
|---|---|---|
| 하 (Low) | `get_prerequisites` | 먼저 학습해야 할 선행 개념 조회 |
| 중 (Mid) | `get_current_concept` | 현재 개념의 심화 학습 유도 |
| 상 (High) | `get_advanced_concepts` | 다음 단계 심화 개념 조회 |

---

## 기술 스택

| 레이어 | 기술 |
|---|---|
| 오케스트레이션 | LlamaIndex Workflows |
| 진단 모델 | PyTorch · BKTransformer (in-process, torch.compile) |
| LLM | Ollama Qwen 3 8B (Q4_K_M 양자화, 로컬) |
| 지식 그래프 | Neo4j Aura |
| 그래프 접근 | Neo4j 비동기 드라이버 |
| 관찰 가능성 | Langfuse v4 |
| API | FastAPI + uvicorn |
| 의존성 관리 | uv (uv.lock 단일 소스) |

---

## 핵심 설계 결정

### 1. torch.set_num_threads(2)와 추론 시퀀스 길이 고정 (T=30)

**torch.set_num_threads(2)**

FastAPI는 `asyncio.to_thread`로 BKT 추론을 스레드 풀에서 실행합니다. PyTorch는 기본적으로 모든 가용 코어를 intra-op 스레드로 사용합니다. 46명의 학생 요청이 동시에 들어오면, 각 BKT 추론이 전체 코어를 독점하려 하면서 코어 경합이 발생합니다.

T=30, B=1의 소형 텐서에서는 intra-op 스레드 수를 늘려도 실제 연산 속도 향상이 거의 없습니다. 행렬 크기가 너무 작아 병렬화 효과가 스레드 생성 오버헤드를 상쇄하지 못하기 때문입니다. 따라서 스레드 수를 2로 제한해 각 BKT 추론의 코어 독점을 방지하고 동시 요청 처리 성능을 높입니다.

**T=30 고정**

이 프로토타입에서 학생의 최대 상호작용 시퀀스 길이는 30입니다. BKTransformer는 `block_size=818`로 훈련되었으나, 추론 시 `_run_bkt_loop`의 반복 횟수를 실제 입력 길이(`obs.shape[1]`)로 제한합니다. T=30이면 818회 대신 30회만 반복해 약 27배의 루프 계산을 절감합니다. 훈련된 가중치는 변경하지 않습니다.

### 2. Cypher 쿼리 결정을 에이전트에게 맡기지 않은 이유

추천 에이전트가 어떤 Cypher 쿼리를 실행할지 스스로 결정하지 않습니다. 진단 에이전트의 숙련도 레벨(상/중/하)이 쿼리를 결정합니다.

LLM이 도구 선택까지 담당하면 설명 가능성과 일관성이 떨어집니다. 사전 정의된 쿼리 선택은 GraphRAG 검색 단계에서의 투명성을 보장합니다.

---

## 프로젝트 구조

```
src/
└── ai_tutor/
    ├── bkt/
    │   ├── model.py              # BKTransformer (RoPE, SwiGLU)
    │   ├── config.py             # BKTConfig (n_skills, n_embd, block_size 등)
    │   └── checkpoints/          # 학습된 모델 가중치
    ├── workflow/
    │   ├── workflow.py           # LlamaIndex TutorWorkflow 정의
    │   ├── diagnosis.py          # BKT in-process 추론 + LLM 숙련도 진단
    │   ├── recommendation.py     # GraphRAG + LLM 피드백 생성
    │   └── schemas.py            # Pydantic 모델 (BKTTimestep, AnalysisRecord 등)
    ├── tools/
    │   └── neo4j_tool.py         # Neo4j 비동기 드라이버 + Cypher 쿼리
    ├── llm_client.py             # Ollama 클라이언트 (langfuse.openai 래퍼)
    ├── observability.py          # Langfuse 설정
    └── api.py                    # FastAPI 진입점 (BKT 워밍업 포함)

tests/                            # 단위 테스트 (순수 함수 커버리지)
scripts/                          # Langfuse 프롬프트 등록 스크립트
data/                             # 훈련 데이터셋
```

---

## 로컬 실행 방법

### 사전 요구사항

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- [Ollama](https://ollama.com/) 설치 및 실행

### 1. 의존성 설치

```bash
uv sync
```

### 2. Ollama 모델 다운로드

```bash
ollama pull qwen3:8b   # Q4_K_M 기본 양자화 (~5 GB)
```

Ollama 서버가 백그라운드에서 실행 중이어야 합니다 (기본값: `http://localhost:11434`).

### 3. 환경 변수 설정

```bash
cp .env.example .env
# .env에 Langfuse 키와 Neo4j 연결 정보를 입력하세요
```

`.env.example` 항목:
```
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
LLM_BASE_URL=http://localhost:11434/v1   # Ollama 기본값
LLM_MODEL=qwen3:8b
```

### 4. Langfuse 프롬프트 등록 (최초 1회)

```bash
uv run python scripts/create_prompts.py
```

### 5. API 서버 실행

```bash
uv run uvicorn ai_tutor.api:app --port 8000 --reload
```

서버 시작 시 BKTransformer가 로드되고 `torch.compile` 워밍업이 실행됩니다 (최초 시작 시 수십 초 소요).

### 6. API 호출

```bash
curl -X POST localhost:8000/tutor \
  -H "Content-Type: application/json" \
  -d '{"student_id": "s1", "sequence": [[1,1],[2,0],[1,1]], "skill_id_to_name": {"1": "순환소수", "2": "유리수"}}'
```

---

## 관찰 가능성

모든 파이프라인 실행은 Langfuse에서 다음 계층 구조로 추적됩니다.

```
tutor_pipeline  [trace]
  ├── run_bkt      — 타임스텝별 BKT 파라미터 (P(know), P(learn), P(guess), P(slip))
  ├── diagnose     — LLM 생성: 자연어 진단, 숙련도 레벨 (상/중/하)
  └── recommend    — 스킬별 LLM 생성: 그래프 컨텍스트 + 피드백
```

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
