# LIVE MINIATURE — AI 혼잡 예측 + 다익스트라

업로드된 행사장 평면도를 배경으로 사용하는 해커톤 MVP 스타터입니다.

## 핵심 구조

```text
로비·성과부스 인원 입력
        ↓
Random Forest 혼잡 가중치 예측
        ↓
통로별 weight = 거리 × 예측 multiplier
        ↓
다익스트라 최적 경로 계산
        ↓
평면도 위 경로 표시
```

AI는 최단 경로 자체를 임의로 생성하지 않습니다. AI는 통로의 혼잡 가중치만
예측하고, 실제 경로의 유효성과 최단 경로 계산은 다익스트라가 담당합니다.

## 현재 도면 반영 내용

- 서측 상부 쌍여닫이문: 입구
- 서측 하부 쌍여닫이문: 출구
- 북측 쌍여닫이문: 보조 출입구
- 케이터링
- 포토존
- 안내데스크
- 성과 홍보부스 10개
- 성과부스 중앙 통로는 왼쪽 개방부에서 진입
- 성과부스 오른쪽 끝은 외부 출구와 연결하지 않음

입구·출구의 운영 역할은 임시 가정이며 담당자 확인 후 이름만 수정하면 됩니다.

## AI 학습 수준

초기 모델은 실제 행사 로그가 아니라 시뮬레이션 데이터 5,000건으로 학습합니다.

입력 특성:

- 구역별 현재 인원 수(서측·중앙 직행·북측 우회·성과부스)
- 통로 폭
- 시간대
- 행사 단계
- 성과부스 구역 여부
- 최근 유입 인원

출력:

- 통로 이동 비용 배수
- 한산 / 보통 / 혼잡 / 매우 혼잡

발표에서는 다음처럼 설명하는 것이 정확합니다.

> 현재 프로토타입은 실데이터 수집 전 단계이므로 시뮬레이션 혼잡 시나리오로
> 초기 학습했습니다. 향후 카메라·센서·체크인 로그를 수집해 실제 행사별 모델로
> 재학습할 수 있도록 설계했습니다.

실데이터 기반 정확도를 주장하면 안 됩니다.

## 실행 순서

### 1. AI 서버 실행

```bash
cd ai-service

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --port 8000
```

확인:

```text
http://localhost:8000/health
```

백엔드 테스트:

```bash
cd ai-service
python -m unittest discover -s tests -v
```

### 앱 연동용 API

앱 부팅용 설정:

```http
GET http://127.0.0.1:8000/app-config
```

응답에는 기본 지도 ID, 기본 보행 속도, 입력 제한값, telemetry 집계 윈도우,
지원 기능 플래그, 주요 API 경로가 포함됩니다. 앱 화면을 따로 만들 때는 이 값을
먼저 받아 `map_id`, 입력 최대값, QR 카운트 제한, endpoint 목록을 하드코딩하지
않는 흐름을 권장합니다.

사용자 앱에서 보통 호출하는 순서:

1. `GET /app-config`
2. `GET /maps`
3. `GET /maps/{map_id}`
4. `GET /maps/{map_id}/checkpoints`
5. `POST /route` 또는 `POST /navigation/sessions`
6. 사용자의 현재 위치가 바뀔 때 좌표 기반이면
   `POST /navigation/sessions/{session_id}/position-coordinate`, QR/NFC/비콘 기반이면
   `POST /navigation/update-position` 또는
   `POST /navigation/sessions/{session_id}/position`
7. 운영자 화면이 있으면 `POST /telemetry/manual-crowd`, 앱 스캔 이벤트가 있으면
   `POST /telemetry/qr-scan`

지도 생성 관리자 화면에서만 쓰는 API:

- `POST /map-generation/jobs`
- `GET /map-generation/jobs`
- `POST /map-generation/jobs/{job_id}/generate-draft`
- `POST /map-generation/jobs/{job_id}/draft-map`
- `POST /map-generation/jobs/{job_id}/postprocess-draft`
- `POST /map-generation/jobs/{job_id}/route-preview`
- `POST /map-generation/jobs/{job_id}/save-map`
- `POST /maps/validate-data`
- `POST /maps`
- `DELETE /maps/{map_id}`

사용자 앱은 지도 생성 API를 직접 호출하지 않는 구조가 좋습니다. 운영자가 검수해
저장한 `map_id`만 사용자 앱에 노출하면 됩니다.

지도 목록 조회:

```http
GET http://127.0.0.1:8000/maps
```

응답에는 지도 ID, 이름, 이미지 경로, 노드/간선 개수, 선택 가능 노드 개수,
지도 데이터 검증 상태(`valid`)가 포함됩니다.

지도 데이터 조회:

```http
GET http://127.0.0.1:8000/maps/default
```

응답에는 도면 이미지 경로(`/static/floorplan.png`), SVG 좌표계 크기, 노드, 간선,
선택 가능한 노드 목록(`selectable_nodes`)이 포함됩니다. 앱 프론트는
`selectable_nodes`로 출발지·목적지 선택지를 만들고, `nodes` 좌표로 도면 위
표시 위치를 계산하면 됩니다.

체크포인트 조회:

```http
GET http://127.0.0.1:8000/maps/default/checkpoints
```

응답에는 QR/NFC/비콘 체크포인트 ID, 표시 이름, 연결된 `node_id`, 혼잡도 집계
구역(`region`)이 포함됩니다. 실제 QR 코드에는 `checkpoint_id`만 넣고, 앱은
스캔 후 이 ID를 백엔드에 보내 현재 위치를 갱신하면 됩니다.

좌표를 노드로 스냅:

```http
POST http://127.0.0.1:8000/maps/default/snap-coordinate
Content-Type: application/json
```

```json
{
  "x": 1088,
  "y": 646,
  "max_distance_px": 120,
  "selectable_only": false
}
```

앱에서 사용자가 도면을 터치하거나 BLE/UWB/Wi-Fi 등으로 현재 좌표를 추정한 경우,
이 API로 가장 가까운 경로 노드를 확인할 수 있습니다. 응답의
`within_threshold`가 `false`이면 현재 좌표가 경로 그래프에서 너무 멀다는 뜻이므로
앱에서는 위치 재확인 UI를 띄우는 흐름이 좋습니다.
`selectable_only`가 `false`이면 가장 가까운 edge ID, edge 위 투영 좌표
(`projected_x`, `projected_y`), edge 진행률(`edge_progress_ratio`)도 함께 반환하므로
앱 화면에서 현재 위치를 통로 선 위에 표시할 수 있습니다.

지도 데이터 검증:

```http
GET http://127.0.0.1:8000/maps/default/validate
```

응답에는 지도 JSON의 필수 필드 누락, 중복 노드/간선 ID, 존재하지 않는 노드로
연결된 간선, 잘못된 거리·폭 값, 선택 가능한 고립 노드 여부, 선택 가능한 지점 사이의
경로 연결성 오류가 포함됩니다.
LLM이나 관리 화면이 새 지도를 생성한 뒤에는 이 API로 먼저 검증하는 흐름을
권장합니다.

LLM 생성 지도 스키마 조회:

```http
GET http://127.0.0.1:8000/maps/schema
```

LLM에는 이 응답의 필드 계약을 기준으로 지도 JSON을 만들도록 지시하면 됩니다.
좌표계는 도면 이미지의 pixel 좌표이며, 노드 ID는 간선과 체크포인트에서 참조되므로
중복 없이 안정적인 영문 ID로 생성해야 합니다.

LLM 생성 지도 데이터 검증:

```http
POST http://127.0.0.1:8000/maps/validate-data
Content-Type: application/json
```

요청 본문에는 저장 전 지도 JSON 전체를 넣습니다. 응답의 `valid`가 `true`일 때만
파일 저장 또는 운영 반영 단계로 넘어가야 합니다. 이 API는 저장하지 않고 검증만
수행합니다. `unreachable_route_pairs`가 비어 있지 않으면 해당 출발지·목적지
조합은 실제 경로 안내가 불가능하다는 뜻입니다.
LLM이 만든 초안에서 흔한 오류를 찾기 위해 `out_of_bounds_node_ids`,
`invalid_node_ids`, `invalid_edge_ids`, `invalid_checkpoint_ids`도 함께 반환합니다.
프론트 검수 화면에서는 이 목록을 표시하면 운영자가 어떤 노드나 edge를 수정해야
하는지 바로 확인할 수 있습니다.
`fix_suggestions`에는 오류 유형별 수정 가이드, 대상 ID, 허용 enum 값, 연결 실패
샘플 경로쌍이 들어갑니다. 검수 UI에서는 원본 `errors`보다 이 배열을 먼저 보여주는
편이 수정 작업에 더 유용합니다.
`quality_summary`는 검수 상태를 점수화합니다. `score`는 0~100, `readiness`는
`ready`, `needs_review`, `blocked` 중 하나입니다. `can_save`가 `false`면 저장 버튼을
비활성화하고, `reason_codes`를 검수 패널에 표시하는 흐름을 권장합니다.

검증된 지도 저장:

```http
POST http://127.0.0.1:8000/maps
Content-Type: application/json
```

```json
{
  "venue_map": {
    "id": "generated_hall_01",
    "name": "LLM 생성 행사장",
    "image": "/static/floorplan.png",
    "width": 1672,
    "height": 941,
    "nodes": [
      { "id": "ENTRY", "name": "입구", "x": 100, "y": 100, "type": "entrance", "selectable": true },
      { "id": "INFO", "name": "안내데스크", "x": 240, "y": 100, "type": "facility", "selectable": true }
    ],
    "checkpoints": [
      { "id": "ENTRY_QR", "name": "입구 QR", "node_id": "ENTRY", "region": "lobby" }
    ],
    "edges": [
      {
        "id": "E_ENTRY_INFO",
        "from": "ENTRY",
        "to": "INFO",
        "distance": 8.0,
        "widthM": 3.0,
        "zone": "lobby",
        "crowdRegion": "central",
        "bidirectional": true
      }
    ]
  },
  "overwrite": false
}
```

저장 API는 먼저 지도 데이터를 검증하고, 통과한 경우에만 `ai-service/maps/{id}.json`에
파일을 씁니다. `id`는 소문자로 시작하고 소문자, 숫자, `_`, `-`만 사용할 수 있습니다.
기본 지도 `default`는 이 API로 덮어쓸 수 없습니다. 같은 ID가 이미 있으면 `409`를
반환하며, 명시적으로 갱신할 때만 `overwrite: true`를 사용합니다.

앱에서 생성 지도 사용:

1. `GET /maps`로 저장된 지도 목록을 조회합니다.
2. 사용자가 선택한 지도 ID로 `GET /maps/{map_id}`를 호출해 노드와 체크포인트를 불러옵니다.
3. `/route`, `/crowd/{map_id}`, `/navigation/update-position`,
   `/navigation/sessions` 요청의 `map_id`에 선택한 지도 ID를 넣습니다.

프론트 API 클라이언트에서는 `fetchVenueMaps()`로 목록을 받고,
`calculateRoute(startId, destinationId, inputs, options, mapId)`처럼 마지막 인자로
선택한 지도 ID를 넘기면 됩니다.

도면 입력 기반 지도 생성 작업:

현재 단계에서는 서버가 LLM을 직접 호출하지 않고, 도면 이미지와 LLM이 만든 지도 JSON을
안전하게 받는 파이프라인을 제공합니다. 이후 OpenAI Vision 또는 다른 LLM 호출부를
붙일 때도 같은 검증/저장 API를 그대로 사용할 수 있습니다.

1. 도면 이미지를 base64로 보내 생성 작업을 만듭니다.

```http
POST http://127.0.0.1:8000/map-generation/jobs
Content-Type: application/json
```

```json
{
  "source_image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "filename": "floorplan.png",
  "mime_type": "image/png",
  "target_map_id": "generated_hall_01",
  "notes": "사용자가 업로드한 행사장 도면"
}
```

응답의 `status`가 `needs_llm`이면 `source_image_url`, `GET /maps/schema`,
`POST /maps/validate-data` 규격을 LLM 지도 생성 프롬프트에 사용합니다.

생성 작업 목록 조회:

```http
GET http://127.0.0.1:8000/map-generation/jobs?limit=20
```

프론트 검수 화면에서는 이 API로 최근 업로드/생성 작업을 다시 열 수 있습니다.

2. 서버에서 OpenAI Vision 모델로 지도 JSON 초안을 자동 생성할 수 있습니다.

백엔드 실행 전에 `OPENAI_API_KEY`를 설정합니다. API 키는 프론트에 넣지 않고
`ai-service` 서버 환경변수로만 관리합니다.

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MAP_MODEL="gpt-5.6-terra"
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

```http
POST http://127.0.0.1:8000/map-generation/jobs/{job_id}/generate-draft
Content-Type: application/json
```

```json
{
  "model": "gpt-5.6-terra",
  "image_detail": "auto",
  "extra_instructions": "관람객이 실제로 걸을 수 있는 통로만 edge로 연결"
}
```

서버는 OpenAI Responses API에 도면 이미지를 보내고, 받은 JSON을 즉시 기존 지도
검증 로직에 통과시킵니다. `OPENAI_API_KEY`가 없으면 이 엔드포인트는 `503`을
반환하며, 아래 수동 draft 등록 API를 계속 사용할 수 있습니다.

3. 또는 LLM이 만든 지도 JSON 초안을 작업에 직접 붙입니다.

```http
POST http://127.0.0.1:8000/map-generation/jobs/{job_id}/draft-map
Content-Type: application/json
```

```json
{
  "venue_map": {
    "id": "generated_hall_01",
    "name": "LLM 생성 행사장",
    "image": "/generated-assets/{job_id}.png",
    "width": 1672,
    "height": 941,
    "nodes": [],
    "checkpoints": [],
    "edges": []
  }
}
```

서버는 작업의 `target_map_id`를 지도 `id`로 강제하고, 기존 지도 검증 로직을 실행합니다.
`status`가 `draft_valid`가 될 때만 저장 단계로 넘어가야 합니다.

초안 JSON과 검증 결과 조회:

```http
GET http://127.0.0.1:8000/map-generation/jobs/{job_id}/draft-map
```

이 응답의 `draft_map`은 편집기에 표시하고, `validation.fix_suggestions`는 수정 안내
패널에 표시하면 됩니다. 수정 후에는 같은 `POST /draft-map`에 전체 JSON을 다시 보내
재검증합니다.

LLM draft 자동 후처리:

```http
POST http://127.0.0.1:8000/map-generation/jobs/{job_id}/postprocess-draft
Content-Type: application/json
```

```json
{
  "apply": true,
  "recalculate_edge_distance": true,
  "clamp_coordinates": true,
  "close_node_threshold_px": 18,
  "suggest_connection_edges": true,
  "apply_connection_suggestions": false,
  "max_connection_suggestions": 5,
  "max_connection_distance_px": 260
}
```

후처리는 LLM이 만든 초안에서 확실하게 고칠 수 있는 항목만 자동 보정합니다.

- 노드/edge/checkpoint ID를 안정적인 대문자 ID로 정규화
- ID 변경에 맞춰 edge `from`/`to`, checkpoint `node_id` 참조 갱신
- 도면 밖 좌표를 이미지 범위 안으로 보정
- edge 거리 값을 노드 좌표 기반으로 재계산
- 잘못된 `zone`, `crowdRegion`, node `type`, checkpoint `region`을 보수적으로 보정
- 너무 가까운 노드쌍은 자동 병합하지 않고 `suggestions`로만 반환
- 연결이 끊긴 그래프 컴포넌트 사이에서 가장 가까운 edge 후보를 `suggestions`로 반환

`apply: false`로 보내면 보정 결과를 미리보기만 하고 draft 파일은 바꾸지 않습니다.
응답의 `changes`는 실제 자동 수정 목록이고, `suggestions`는 사람이 검수해야 하는
항목입니다. 자동 후처리 후에도 반드시 `validation.valid`와 route preview를 확인한 뒤
저장해야 합니다.
`apply_connection_suggestions: true`를 사용하면 후보 edge를 draft에 추가할 수 있지만,
직선 연결이 실제 보행 가능한 통로인지 검수한 뒤에만 사용해야 합니다. 벽, 부스, 운영
통제 구역을 통과하는 edge는 사람이 제거해야 합니다.
후처리 응답의 `validation.quality_summary`는 연결 후보가 남아 있거나 가까운 노드쌍이
있을 때 `needs_review` 또는 `blocked`로 내려옵니다.

저장 전 경로 미리보기:

```http
POST http://127.0.0.1:8000/map-generation/jobs/{job_id}/route-preview
Content-Type: application/json
```

```json
{
  "start_id": "GATE_W1",
  "destination_id": "BOOTH_10",
  "walking_speed_mps": 1.2,
  "use_congestion": true
}
```

QR/체크포인트 기준으로도 미리볼 수 있습니다.

```json
{
  "checkpoint_id": "INFO_DESK_QR",
  "destination_id": "BOOTH_10",
  "walking_speed_mps": 1.2,
  "use_congestion": true
}
```

이 API는 저장된 지도 대신 draft 지도 JSON으로 `/route`와 같은 알고리즘을 실행합니다.
검수 화면에서는 사용자가 출발지와 목적지를 바꿔가며 `route_points`, `segments`,
`instructions`를 미리 표시할 수 있습니다. draft가 유효하지 않으면 `400`과 함께
기존 검증 결과가 반환됩니다.

4. 검증된 초안을 실제 지도 목록에 저장합니다.

```http
POST http://127.0.0.1:8000/map-generation/jobs/{job_id}/save-map
Content-Type: application/json
```

```json
{
  "overwrite": false
}
```

저장 후에는 `GET /maps`에 생성 지도가 나타나고, 프론트에서 해당 map ID를 선택해
`/route`, `/navigation/update-position`, `/navigation/sessions`에 사용할 수 있습니다.

생성 지도 삭제:

```http
DELETE http://127.0.0.1:8000/maps/generated_hall_01
```

저장 API로 만든 생성 지도만 삭제하는 용도입니다. 기본 지도 `default`는 삭제할 수
없습니다.

카메라 없는 혼잡도 추정:

```http
POST http://127.0.0.1:8000/telemetry/manual-crowd
Content-Type: application/json
```

```json
{
  "map_id": "default",
  "lobby_people": 30,
  "booth_people": 40
}
```

```http
POST http://127.0.0.1:8000/telemetry/qr-scan
Content-Type: application/json
```

```json
{
  "map_id": "default",
  "checkpoint_id": "BOOTH_QR",
  "region": "booth",
  "count": 1
}
```

현재 추정 혼잡도 조회:

```http
GET http://127.0.0.1:8000/crowd/default
```

최근 혼잡도 신호 조회:

```http
GET http://127.0.0.1:8000/telemetry/default?limit=20
```

응답에는 현재 추정 혼잡도, 최근 QR 스캔 목록, 최근 목적지 요청 목록이 포함됩니다.
운영자 화면이나 디버깅 화면에서 “혼잡도 추정이 어떤 신호로 계산됐는지” 확인할 때
사용합니다.

카메라를 쓰지 않는 대신 운영자 수동 입력, QR 체크포인트 스캔, 최근 목적지 요청을
최근 10분 윈도우로 합산해 `lobby_people`, `booth_people`, `recent_inflow`를
추정합니다. 혼잡도 신호는 `ai-service/crowd_signals.sqlite3`에 저장되므로 서버를
재시작해도 최근 윈도우 안의 신호와 운영자 수동 입력은 유지됩니다.

경로 계산:

```http
POST http://127.0.0.1:8000/route
Content-Type: application/json
```

```json
{
  "map_id": "default",
  "start_id": "GATE_W1",
  "destination_id": "BOOTH_10",
  "walking_speed_mps": 1.2,
  "use_congestion": true,
  "algorithm": "astar",
  "log_route_intent": true,
  "crowd_inputs": {
    "lobby_people": 25,
    "booth_people": 55,
    "recent_inflow": 20,
    "hour": 14,
    "event_phase": 2
  }
}
```

응답에는 최단 경로 노드 ID, 간선 ID, 선을 바로 그릴 수 있는 좌표 배열,
구간별 안내 정보(`segments`), 안내 문구(`instructions`), 총 거리, 예상 시간,
혼잡도 반영 비용, 통로별 AI 혼잡 예측값이 포함됩니다.

```json
{
  "map_id": "default",
  "use_congestion": true,
  "walking_speed_mps": 1.2,
  "algorithm": "astar",
  "expanded_state_count": 9,
  "path": ["GATE_W1", "WEST_HALL", "LOBBY_CENTER", "BOOTH_GATE", "AISLE_05", "BOOTH_10"],
  "edge_ids": ["E_W1_WEST", "E_WEST_LOBBY", "E_LOBBY_BOOTH_GATE", "E_GATE_A5", "E_A5_B10"],
  "route_points": [
    { "node_id": "GATE_W1", "name": "메인 입구", "x": 570, "y": 830 },
    { "node_id": "WEST_HALL", "name": "입구 홀", "x": 675, "y": 750 }
  ],
  "segments": [
    {
      "edge_id": "E_W1_WEST",
      "from_node": { "node_id": "GATE_W1", "name": "메인 입구", "x": 570, "y": 830 },
      "to_node": { "node_id": "WEST_HALL", "name": "입구 홀", "x": 675, "y": 750 },
      "distance": 4.8,
      "multiplier": 1.64,
      "weighted_cost": 7.9,
      "estimated_seconds": 4,
      "level": "normal",
      "instruction": "메인 입구에서 입구 홀 방향으로 이동하세요."
    }
  ],
  "total_distance": 38.7,
  "weighted_cost": 65.6,
  "estimated_seconds": 32,
  "instructions": ["메인 입구에서 입구 홀 방향으로 이동하세요."],
  "model_type": "simulated",
  "validation_mae": 0.0877524386918509,
  "predictions": []
}
```

`walking_speed_mps`는 선택 필드이며 기본값은 `1.2`입니다. `use_congestion`도 선택
필드이며 기본값은 `true`입니다. `false`로 보내면 AI 혼잡 배수를 적용하지 않고
순수 거리 기준 최단 경로를 계산합니다. `crowd_inputs`를 생략하면
`GET /crowd/{map_id}`와 같은 추정값을 사용합니다. `log_route_intent`는 선택
필드이며 기본값은 `true`입니다. 경로 요청 목적지를 혼잡도 추정 신호로 기록할지
결정합니다.
`algorithm`은 선택 필드이며 기본값은 `astar`입니다. 기존 다익스트라와 비교하거나
작은 지도에서 디버깅할 때는 `dijkstra`로 보낼 수 있습니다. 응답의
`expanded_state_count`는 탐색 중 확장한 상태 수로, 경로 탐색 비용을 비교할 때
사용할 수 있습니다.

대체 경로 후보 계산:

```http
POST http://127.0.0.1:8000/route-alternatives
Content-Type: application/json
```

```json
{
  "map_id": "default",
  "start_id": "GATE_W1",
  "destination_id": "BOOTH_10",
  "walking_speed_mps": 1.2,
  "use_congestion": true,
  "max_routes": 3,
  "overlap_penalty": 1.8
}
```

응답의 `alternatives`에는 rank별 후보 경로, 좌표, 구간 안내, 예상 시간이 들어갑니다.
1번 후보는 `/route`와 같은 최적 경로이고, 이후 후보는 이미 찾은 경로의 edge에
패널티를 주어 계산한 우회 경로입니다. `overlap_with_best_edge_count`가 낮을수록
최적 경로와 덜 겹치는 경로입니다. 사용자가 “덜 혼잡한 우회 경로”를 고르거나,
운영자가 특정 통로를 막았을 때 대체 안내 후보를 보여주는 데 사용할 수 있습니다.

현재 위치 기반 재경로 계산:

```http
POST http://127.0.0.1:8000/navigation/update-position
Content-Type: application/json
```

```json
{
  "map_id": "default",
  "destination_id": "BOOTH_10",
  "checkpoint_id": "BOOTH_GATE_QR",
  "walking_speed_mps": 1.2,
  "use_congestion": true,
  "log_route_intent": true
}
```

카메라를 쓰지 않는 앱에서는 QR 체크포인트, NFC, 비콘, 운영자 입력 중 하나로
사용자의 현재 위치를 갱신하면 됩니다. `checkpoint_id`만 보내면 백엔드가 지도에
등록된 체크포인트 매핑으로 현재 노드를 찾고, 그 노드를 새 출발지로 삼아 `/route`와
같은 형식의 경로를 다시 계산합니다. `current_node_id`를 직접 보내는 방식도 계속
지원합니다.

도면 좌표 기반 위치 갱신:

```http
POST http://127.0.0.1:8000/navigation/sessions/{session_id}/position-coordinate
Content-Type: application/json
```

```json
{
  "x": 1088,
  "y": 646,
  "max_distance_px": 120,
  "walking_speed_mps": 1.2,
  "use_congestion": true,
  "preference": "shortest",
  "blocked_edge_ids": []
}
```

이 API는 세션의 지도에서 좌표와 가장 가까운 노드를 찾고, 그 노드를 현재 위치로
저장한 뒤 목적지까지 다시 경로를 계산합니다. 좌표는 지도 이미지 픽셀 좌표계입니다.
예를 들어 앱 화면에서 도면 이미지를 확대/축소해서 보여준다면, 터치 좌표를 원본 지도
크기 기준 좌표로 변환한 뒤 보내야 합니다.
좌표가 노드가 아니라 통로 edge 중간에 가까운 경우에는 edge 양끝 중 목적지까지
남은 경로가 더 짧은 노드를 현재 위치로 잡아 불필요하게 뒤로 돌아가는 재경로를
줄입니다.

안내 세션 시작:

```http
POST http://127.0.0.1:8000/navigation/sessions
Content-Type: application/json
```

```json
{
  "map_id": "default",
  "start_id": "GATE_W1",
  "destination_id": "BOOTH_10",
  "walking_speed_mps": 1.2,
  "use_congestion": true,
  "log_route_intent": true
}
```

응답의 `session_id`를 앱에 저장하면 이후 현재 위치를 갱신하거나 앱이 다시 켜졌을 때
안내 상태를 이어서 조회할 수 있습니다.

```http
GET http://127.0.0.1:8000/navigation/sessions/{session_id}
```

세션 현재 위치 갱신:

```http
POST http://127.0.0.1:8000/navigation/sessions/{session_id}/position
Content-Type: application/json
```

```json
{
  "checkpoint_id": "BOOTH_GATE_QR",
  "walking_speed_mps": 1.2,
  "use_congestion": true
}
```

`current_node_id`가 목적지와 같아지면 세션 상태는 `arrived`가 됩니다.

오류 응답:

- 없는 `map_id`: `404`, `Map not found: {map_id}`
- 없는 출발지 또는 목적지: `400`, `missing_node_ids`와 `available_node_ids` 반환
- 노드는 있지만 그래프가 연결되지 않음: `404`, 선택 노드 간 경로 없음
- 체크포인트와 현재 노드가 서로 다름: `400`,
  `current_node_id does not match checkpoint node.`
- 지도 범위 밖 좌표: `400`, `Coordinate is outside map bounds.`
- 가장 가까운 노드가 기준 거리 밖인 좌표 위치 갱신: `400`,
  `Coordinate is too far from the nearest route node.`
- draft 지도가 유효하지 않은 상태에서 route preview 호출: `400`, `validation` 반환
- OpenAI API key 없이 `generate-draft` 호출: `503`, 수동 draft 등록 또는
  `OPENAI_API_KEY` 설정 필요

### 2. 프론트엔드 실행

새 터미널:

```bash
cd frontend
npm install
npm run dev
```

접속:

```text
http://localhost:5173
```

AI 서버가 꺼져 있거나 요청에 실패해도 프론트엔드는 로컬 대체 계산으로
다익스트라 경로를 계속 표시합니다.

## 시연 순서

1. 출발지를 `서측 입구`로 선택
2. 목적지를 `성과 부스 10`으로 선택
3. 중앙 직행 통로 인원을 낮게, 북측 우회 통로 인원을 높게 두고 계산
4. 중앙 직행 통로 인원을 180 이상으로 높이고 북측 우회 통로를 5~10으로 낮춤
5. 다익스트라가 북측 우회 경로로 변경되는 것을 확인
6. 성과부스 인원과 최근 유입 인원을 높여 부스 통로 색상이 변하는지 확인
7. 출발지를 북측 보조 출입구로 변경해 경로를 재계산

## 실제 프로젝트에 합칠 때

- `frontend/src/mapData.ts`: 지도 노드·간선 데이터
- `frontend/src/dijkstra.ts`: 경로 탐색 엔진
- `frontend/src/aiClient.ts`: AI API 호출과 fallback
- `ai-service/main.py`: AI 예측 API
- `ai-service/train_model.py`: 모델 학습

팀 프론트 화면이 이미 있다면 `App.tsx` 전체를 가져오기보다 위 네 모듈을
기존 지도 컴포넌트에 연결하는 방식이 안전합니다.

## 제출 전 반드시 수정할 부분

1. 담당자 확인 후 실제 입구·출구 명칭 확정
2. 부스 번호 방향 확정
3. 노드 좌표를 최종 배경 이미지에 맞게 미세 조정
4. 실제 거리 또는 도면 치수로 edge distance 보정
5. CORS 주소를 배포 도메인에 맞게 변경
6. README에 모델이 시뮬레이션 데이터 기반임을 명시

## 실시간 경로 애니메이션

이번 버전에는 다음 기능이 추가되었습니다.

- 출발지 또는 목적지를 변경하면 자동으로 경로 재계산
- 계산된 경로를 따라 지도 위 캐릭터 자동 이동
- 안내 시작, 일시정지, 처음으로 제어
- 이동 진행률, 다음 지점, 남은 거리, 예상 남은 시간 표시
- 경로 변경 시 이전 애니메이션 중단 후 새 경로로 자동 시작

현재 기능은 실제 GPS·비콘 위치 추적이 아니라 계산된 경로를 시연하는
애니메이션입니다. 실제 위치 갱신은 QR 체크포인트나 관리자 입력을 추후 연결할 수
있습니다.
