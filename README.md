# LIVE MINIATURE

실내 도면 기반 경로 안내와 카메라 없는 혼잡도 추정을 연결한 현재 작업공간 설명서입니다.

이 저장소의 실제 구조는 아래 두 폴더입니다.

```text
live-miniature-hybrid-ai-live-guidance/
├─ ai-service/    # FastAPI AI 서버
└─ frontend/      # 프론트 빌드 산출물
```

현재 `frontend/`는 소스 코드가 아니라 빌드된 정적 파일을 포함하고 있습니다.  
AI 서버는 독립적으로 실행하고, 프론트는 별도 앱 또는 정적 호스팅으로 붙는 구조입니다.

## 역할

- `ai-service/`
  - `/route`, `/route/preview`, `/route-alternatives`, `/route-recommendation`, `/route-multi-stop`
  - `/maps`, `/maps/{map_id}`, `/maps/{map_id}/checkpoints`
  - `/telemetry/*`, `/crowd/*`, `/navigation/*`
  - LLM 도면 후처리, 경로 계산, 혼잡도 추정 담당
- `frontend/`
  - 현재는 `dist/index.html`과 `dist/assets/*`를 포함한 정적 프론트 산출물
  - 프론트가 AI 서버의 `/route` 계약을 사용해 경로를 표시하는 기준 자료

## AI 서버 실행

작업 폴더:

```bash
cd ai-service
```

가상환경 생성 및 실행:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --host 127.0.0.1 --port 4000
```

확인:

```text
http://127.0.0.1:4000/health
```

## AI 서버 기본 설정

`ai-service/main.py` 기준 주요 설정은 다음과 같습니다.

- FastAPI 포트: `4000`
- CORS 허용: `http://localhost:4317`, `http://127.0.0.1:4317`
- 정적 자원:
  - `/static`
  - `/generated-assets`

앱 설정 확인:

```http
GET http://127.0.0.1:4000/app-config
```

이 응답에는 기본 지도 ID, 기본 보행 속도, 기능 플래그, 엔드포인트 목록이 포함됩니다.

## 경로 API

프론트가 현재 사용하는 핵심 요청은 다음입니다.

```http
POST http://127.0.0.1:4000/route
```

```json
{
  "map_id": "default",
  "start_id": "GATE_W1",
  "destination_id": "BOOTH_10",
  "walking_speed_mps": 1.2,
  "use_congestion": true,
  "preference": "shortest",
  "blocked_edge_ids": [],
  "crowd_inputs": {
    "lobby_people": 25,
    "booth_people": 55,
    "recent_inflow": 20,
    "hour": 14,
    "event_phase": 2
  }
}
```

프론트 지도 데이터를 직접 넣는 미리보기는 다음입니다.

```http
POST http://127.0.0.1:4000/route/preview
```

이 API는 `venue_map`, `start_id`, `destination_id`를 받습니다.  
프론트와 같은 지도 기준으로 바로 경로를 계산할 때 사용합니다.

## 혼잡도 추정

카메라를 쓰지 않는 대신 다음 신호를 사용합니다.

- 운영자 수동 혼잡도 입력
- QR 체크포인트 스캔
- 최근 목적지 요청
- 현재 활성 내비게이션 세션
- 최근 내비게이션 업데이트

조회 예시:

```http
GET http://127.0.0.1:4000/crowd/default
GET http://127.0.0.1:4000/telemetry/default?limit=20
```

입력 예시:

```http
POST http://127.0.0.1:4000/telemetry/manual-crowd
POST http://127.0.0.1:4000/telemetry/qr-scan
```

## 지도 생성 및 LLM 후처리

도면 입력에서 지도 JSON을 만들고 검증하는 API도 포함되어 있습니다.

- `/map-generation/jobs`
- `/map-generation/jobs/{job_id}/generate-draft`
- `/map-generation/jobs/{job_id}/draft-map`
- `/map-generation/jobs/{job_id}/postprocess-draft`
- `/map-generation/jobs/{job_id}/route-preview`
- `/map-generation/jobs/{job_id}/save-map`

LLM이 만든 지도는 바로 저장하지 않고, 먼저 검증과 후처리를 거칩니다.

## 프론트 연결 확인

프론트가 AI 서버에 붙는 핵심 계약은 다음입니다.

- `GET /app-config`
- `GET /maps`
- `GET /maps/{map_id}`
- `GET /maps/{map_id}/checkpoints`
- `POST /route`
- `POST /route/preview`
- `POST /navigation/sessions`
- `POST /navigation/update-position`

프론트 번들 기준으로 `/route` 요청 필드는 AI 서버와 일치하도록 맞춰져 있습니다.

## 현재 상태 요약

- AI 서버는 실행 가능
- 프론트는 정적 빌드 산출물 형태로 존재
- 경로 계약은 프론트와 AI 서버가 맞도록 고정됨
- 혼잡도는 카메라 없이도 추정 가능

## 테스트

```bash
cd ai-service
python -m unittest discover -s tests -v
```

