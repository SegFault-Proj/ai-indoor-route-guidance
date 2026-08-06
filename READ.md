# AI Indoor Route Guidance

AI 기반 실내 경로 안내 프로토타입입니다. 행사장 도면 위에 노드와 edge를 정의하고,
혼잡도 예측과 다익스트라 경로 탐색을 결합해 목적지까지의 경로를 계산합니다.

## 구성

- `ai-service/`: FastAPI 백엔드, 경로 알고리즘, 혼잡도 추정, LLM 지도 생성 API
- `frontend/`: React/Vite 기반 확인용 프론트
- `ai-service/maps/default.json`: 기본 행사장 지도 데이터
- `README.md`: 상세 실행 방법과 API 문서

## 주요 기능

- 최단 경로 및 혼잡도 반영 경로 계산
- 좌표/QR/checkpoint 기반 현재 위치 갱신
- 좌표를 가까운 node/edge에 스냅
- 막힌 통로 회피
- 경로 대안, 다중 목적지, ETA 보정
- 카메라 없는 혼잡도 추정
- LLM 기반 지도 draft 생성 및 후처리

## 빠른 실행

백엔드:

```bash
cd ai-service
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

프론트:

```bash
cd frontend
npm install
npm run dev
```

상세 API 사용법은 `README.md`를 참고하세요.
