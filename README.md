# Agentic GraphRAG Tutor — Production Deployment

> This branch adds the deployment surface (Triton, FastAPI worker, Prometheus/Grafana) on top of `main`.

프로젝트 소개, BKTransformer 설계, GraphRAG 흐름, 도메인 설계 결정은 [`main` README](https://github.com/SOHNJUNHO/AI-Tutor/blob/main/README.md)를 참조하세요. 이 README는 `main` 대비 추가된 배포 인프라와 그 설계 결정만 다룹니다.

---

## 프로덕션 설계 결정

`main`의 도메인 설계 결정과 별개로, 배포 단계에서 추가된 결정입니다.

### 1. Triton Inference Server로 BKT 분리

Triton으로 모델을 분리하면 FastAPI는 오케스트레이션 역할만 남고, 모델은 단일 컨테이너에서 호스팅됩니다. 큐잉·배칭·동시성 제어가 모델 경계에서 처리되므로, FastAPI 워커 수와 모델 인스턴스 수를 독립적으로 스케일할 수 있습니다.(단, 두 컨테이너를 서로 다른 호스트에 배치할 때 성립합니다.)

현재 compose는 단일 호스트를 가정합니다. 다음 세 가지 방법으로 CPU 경합을 방지합니다.
 - BKT_NUM_THREADS=2 (triton_server/model_repository/bkt_transformer/1/model.py:13) — PyTorch intra-op 스레드를 2개로 제한. BKT 추론이 모든 코어를 독점하지 못함.                   │
 - instance_group { count: 1, kind: KIND_CPU } (config.pbtxt) — Triton 내부에서 동시 추론을 1개로 직렬화. 컨테이너 내부 경합 자체를 차단.                                           │
 - FastAPI 워커는 torch를 import하지 않으므로 CPU를 거의 쓰지 않음. 워크로드 대부분은 비동기 I/O(LLM, Neo4j) 대기. 

### 2. Python backend 선택 (ONNX/TorchScript 대신)

`main` README의 "ONNX 미사용" 결정과 같은 이유입니다. `_run_bkt_loop`는 타임스텝마다 반복되는 스칼라 베이즈 업데이트이고, ONNX/TorchScript는 이 반복문을 정적 그래프로 펼쳐야 합니다.

Python backend는 기존 `BKTransformer.infer()`를 그대로 호출합니다.

### 3. gRPC 클라이언트 (`tritonclient.grpc.aio`)

FastAPI는 비동기입니다. 동기 클라이언트로 추론을 호출하면 이벤트 루프가 차단됩니다. `tritonclient.grpc.aio`로 호출을 `await`하면 워커는 추론이 진행되는 동안 다른 요청을 계속 처리할 수 있습니다. `/readyz`는 Triton에 ping을 보내, 둘 다 ready일 때만 200을 반환합니다.

### 4. `[bkt]` extra로 torch 분리

`torch`/`torchao`/`torchtune`은 `pyproject.toml`의 optional `[bkt]` extra로 이동했습니다. 메인 deps에는 `tritonclient[grpc]`만 추가했고, `uv pip compile`로 의존성 충돌 없이 3개 패키지(grpcio, python-rapidjson, tritonclient)만 신규 추가됨을 확인했습니다.

결과: FastAPI 이미지에서 torch 스택이 제거되어 빌드/콜드스타트가 가벼워집니다. 모델 인프라는 `Dockerfile.triton` 하나에 격리됩니다.

---

## 로컬 실행

### 사전 요구사항

- Docker, Docker Compose
- [Ollama](https://ollama.com/) (호스트에서 실행, `qwen3:8b` 모델 pull 완료)
- Neo4j Aura 또는 자체 호스팅 Neo4j 인스턴스
- Langfuse Cloud 계정 또는 self-hosted

### 핵심 변수:

- `TRITON_URL=triton:8001` — compose 네트워크 안의 Triton 서비스 이름.
- `BKT_CHECKPOINT` — Triton 컨테이너 전용 (FastAPI는 읽지 않음). 체크포인트 경로를 바꾸려면 `.env`가 아니라 `docker-compose.yml → services.triton.environment`에서 설정.

### 포트

| 포트 | 서비스 | 용도 |
|---|---|---|
| 8000 | app | FastAPI (`/tutor`, `/readyz`, `/metrics`) |
| 8001 | triton | gRPC inference |
| 8002 | triton | Prometheus metrics |
| 9090 | prometheus | 메트릭 UI |
| 3000 | grafana | 대시보드 (admin/admin) |

### 스모크 테스트

```bash
curl localhost:8000/readyz       # 200 == FastAPI + Triton 모두 ready

curl -X POST localhost:8000/tutor \
  -H "Content-Type: application/json" \
  -d '{"student_id": "s1", "sequence": [[1,1],[2,0],[1,1]], "skill_id_to_name": {"1": "순환소수", "2": "유리수"}}'
```

---

## 저장소 변경

`main` 대비 추가·변경된 최상위 경로만 표시합니다.

```
├── Dockerfile                # FastAPI 워커 (슬림, torch 없음)
├── Dockerfile.triton         # Triton + torch CPU + torchtune + BKT 모듈 복사
├── docker-compose.yml        # app · triton · prometheus · grafana
├── triton_server/
│   └── model_repository/bkt_transformer/
│       ├── config.pbtxt      # max_batch_size=0, obs/output → corrects/latents/params
│       └── 1/model.py        # TritonPythonModel.initialize/execute
├── ops/
│   ├── prometheus.yml        # 스크레이프 타겟 (app:8000, triton:8002)
│   ├── grafana-datasources.yml
│   └── dashboards/
└── pyproject.toml            # [bkt] extra로 torch 분리, tritonclient[grpc] 추가
```

`src/ai_tutor/` 변경 사항:
- `workflow/diagnosis.py` — in-process `_get_bkt_model`/`_infer_sync` 제거, `_infer_triton`(gRPC) 추가.
- `api.py` — lifespan에서 in-process BKT 워밍업 제거, `close_triton_client()` 호출 추가, `/readyz`가 `await is_bkt_ready()`로 Triton ping.

---

## TODO

- **`app` 이미지 빌드 파이프라인.** `docker-compose.yml`의 `app` 서비스는 `image: ${ECR_IMAGE}`를 참조하지만, FastAPI 워커용 `Dockerfile`을 ECR로 푸시하는 CI 단계가 아직 없습니다. 현재 흐름은 푸시된 이미지가 있다고 가정 — 로컬 빌드/푸시 스크립트를 추가하거나 `app` 서비스에 `build:` 블록을 두는 결정이 필요.
- **`torch.compile` 워밍업 재적용 (선택).** Triton의 `initialize()`에서 `torch.compile`을 다시 적용할지 미결정. T=30 워크로드에서 큰 차이가 없을 것으로 예상 — p99가 SLO를 넘기면 재검토.
- **`instance_group { count: 1 }` 스케일아웃 검증.** 단일 Python 인스턴스로 동작 확인까지만 완료. 다중 인스턴스 환경에서의 `BKT_NUM_THREADS` 튜닝은 부하 테스트 후 결정.
- **AWS 인프라 코드.** 저장소에 AWS IaC는 없습니다. `main` README의 "AWS 프로덕션은 `production-readiness` 브랜치" 안내는 "AWS에 배포 준비된 상태"를 의미하며, 실제 배포 자동화는 별도 작업.
