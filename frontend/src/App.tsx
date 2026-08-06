import { useEffect, useMemo, useRef, useState } from "react";
import {
  calculateRoute,
  fetchCrowdEstimate,
  fetchVenueMap,
  fetchVenueMaps,
  predictEdgeWeights,
  recordQrScan,
  updateManualCrowd,
} from "./aiClient";
import { calculateShortestRoute } from "./dijkstra";
import {
  edges as fallbackEdges,
  MAP_HEIGHT as FALLBACK_MAP_HEIGHT,
  MAP_WIDTH as FALLBACK_MAP_WIDTH,
  nodes as fallbackNodes,
  selectableNodes as fallbackSelectableNodes,
} from "./mapData";
import type {
  CrowdInputs,
  CrowdEstimate,
  MapSummary,
  MapNode,
  PredictedEdgeWeight,
  RouteOptions,
  RouteResult,
  VenueMap,
} from "./types";
import { useRouteAnimation } from "./useRouteAnimation";
import "./styles.css";

const fallbackMap: VenueMap = {
  id: "fallback",
  name: "로컬 기본 지도",
  image: "/floorplan.png",
  width: FALLBACK_MAP_WIDTH,
  height: FALLBACK_MAP_HEIGHT,
  nodes: fallbackNodes,
  edges: fallbackEdges,
  checkpoints: [],
  selectableNodes: fallbackSelectableNodes,
};

function polylinePoints(
  route: RouteResult | null,
  nodeMap: Map<string, MapNode>
): string {
  if (!route) return "";

  return route.path
    .map((nodeId) => nodeMap.get(nodeId))
    .filter((node) => Boolean(node))
    .map((node) => `${node!.x},${node!.y}`)
    .join(" ");
}

function statusText(status: ReturnType<typeof useRouteAnimation>["status"]): string {
  switch (status) {
    case "running":
      return "안내 중";
    case "paused":
      return "일시정지";
    case "arrived":
      return "목적지 도착";
    default:
      return "경로 준비 중";
  }
}

export default function App() {
  const [venueMap, setVenueMap] = useState<VenueMap>(fallbackMap);
  const [availableMaps, setAvailableMaps] = useState<MapSummary[]>([]);
  const [selectedMapId, setSelectedMapId] = useState("default");
  const [startId, setStartId] = useState("GATE_W1");
  const [destinationId, setDestinationId] = useState("BOOTH_10");

  const [inputs, setInputs] = useState<CrowdInputs>({
    lobbyPeople: 25,
    boothPeople: 55,
    recentInflow: 20,
    hour: 14,
    eventPhase: 2,
  });
  const [routeOptions, setRouteOptions] = useState<RouteOptions>({
    useCongestion: true,
    useEstimatedCrowd: false,
    walkingSpeedMps: 1.2,
  });

  const [crowdEstimate, setCrowdEstimate] = useState<CrowdEstimate | null>(null);
  const [predictions, setPredictions] = useState<PredictedEdgeWeight[]>([]);
  const [route, setRoute] = useState<RouteResult | null>(null);
  const [routeInstructions, setRouteInstructions] = useState<string[]>([]);
  const [routeEstimatedSeconds, setRouteEstimatedSeconds] = useState<number | null>(null);
  const [routeSource, setRouteSource] = useState<"api" | "fallback" | null>(null);
  const [loading, setLoading] = useState(false);
  const requestSequenceRef = useRef(0);
  const nodeMap = useMemo(
    () => new Map(venueMap.nodes.map((node) => [node.id, node])),
    [venueMap.nodes]
  );

  const sourceLabel =
    routeSource === "api"
      ? "백엔드 경로 API"
      : routeSource === "fallback"
        ? "로컬 대체 계산"
        : predictions.length === 0
      ? "예측 전"
      : predictions.every((prediction) => prediction.source === "ai")
        ? "AI 모델"
        : "로컬 대체 계산";

  const routePoints = useMemo(
    () => polylinePoints(route, nodeMap),
    [nodeMap, route]
  );

  const guidance = useRouteAnimation(route, venueMap.nodes, 90);
  const currentTargetName = guidance.currentTargetNodeId
    ? nodeMap.get(guidance.currentTargetNodeId)?.name ?? "다음 지점"
    : "-";

  const remainingDistance = route
    ? Math.max(
        0,
        route.totalDistance * (1 - guidance.progressPercent / 100)
      )
    : 0;

  const estimatedSeconds = route
    ? Math.ceil(
        (routeEstimatedSeconds ?? route.totalDistance / routeOptions.walkingSpeedMps) *
          (1 - guidance.progressPercent / 100)
      )
    : 0;

  async function calculate(autoStart = true): Promise<void> {
    const requestSequence = ++requestSequenceRef.current;
    setLoading(true);

    try {
      let nextPredictions: PredictedEdgeWeight[];
      let nextRoute: RouteResult | null;
      let nextInstructions: string[] = [];
      let nextEstimatedSeconds: number | null = null;
      let nextRouteSource: "api" | "fallback" = "api";
      let nextCrowdInputs: CrowdInputs | null = null;

      try {
        const routeResponse = await calculateRoute(
          startId,
          destinationId,
          inputs,
          routeOptions,
          selectedMapId
        );
        nextPredictions = routeResponse.predictions;
        nextRoute = routeResponse.route;
        nextInstructions = routeResponse.instructions;
        nextEstimatedSeconds = routeResponse.estimatedSeconds;
        nextCrowdInputs = routeResponse.crowdInputs;
      } catch (error) {
        console.warn("경로 API를 사용할 수 없어 로컬 계산으로 전환합니다.", error);
        nextRouteSource = "fallback";
        nextPredictions = await predictEdgeWeights(venueMap.edges, inputs);
        nextRoute = calculateShortestRoute(
          venueMap.nodes,
          venueMap.edges,
          nextPredictions,
          startId,
          destinationId
        );
      }

      // 빠르게 선택을 변경했을 때 이전 API 결과가 최신 경로를 덮지 않도록 합니다.
      if (requestSequence !== requestSequenceRef.current) return;

      setPredictions(nextPredictions);
      setRoute(nextRoute);
      setRouteInstructions(nextInstructions);
      setRouteEstimatedSeconds(nextEstimatedSeconds);
      setRouteSource(nextRouteSource);
      if (nextRouteSource === "api" && nextCrowdInputs) {
        setCrowdEstimate({
          mapId: selectedMapId,
          lobbyPeople: nextCrowdInputs.lobbyPeople,
          boothPeople: nextCrowdInputs.boothPeople,
          recentInflow: nextCrowdInputs.recentInflow,
          hour: nextCrowdInputs.hour,
          eventPhase: nextCrowdInputs.eventPhase,
          signals: crowdEstimate?.signals ?? {
            windowMinutes: 10,
            qrScans: { lobby: 0, booth: 0 },
            routeIntents: { lobby: 0, booth: 0 },
            manualEstimate: {
              lobby_people: nextCrowdInputs.lobbyPeople,
              booth_people: nextCrowdInputs.boothPeople,
            },
          },
        });
      }

      if (!autoStart && nextRoute) {
        window.setTimeout(guidance.pause, 220);
      }
    } finally {
      if (requestSequence === requestSequenceRef.current) {
        setLoading(false);
      }
    }
  }

  async function refreshCrowdEstimate(): Promise<void> {
    try {
      setCrowdEstimate(await fetchCrowdEstimate(selectedMapId));
    } catch (error) {
      console.warn("혼잡도 추정값을 불러오지 못했습니다.", error);
    }
  }

  async function saveManualCrowd(): Promise<void> {
    try {
      setCrowdEstimate(
        await updateManualCrowd(
          {
            lobbyPeople: inputs.lobbyPeople,
            boothPeople: inputs.boothPeople,
          },
          selectedMapId
        )
      );
    } catch (error) {
      console.warn("운영자 혼잡도 입력 저장에 실패했습니다.", error);
    }
  }

  async function addQrScan(region: "lobby" | "booth"): Promise<void> {
    try {
      setCrowdEstimate(await recordQrScan(region, 1, selectedMapId));
    } catch (error) {
      console.warn("QR 스캔 기록에 실패했습니다.", error);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function loadMapList(): Promise<void> {
      try {
        const nextMaps = await fetchVenueMaps();
        if (cancelled) return;

        setAvailableMaps(nextMaps);
      } catch (error) {
        console.warn("지도 목록 API를 사용할 수 없어 현재 지도만 표시합니다.", error);
      }
    }

    void loadMapList();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadMap(): Promise<void> {
      try {
        const nextMap = await fetchVenueMap(selectedMapId);
        if (cancelled) return;

        requestSequenceRef.current += 1;
        setVenueMap(nextMap);
        setRoute(null);
        setPredictions([]);
        setRouteInstructions([]);
        setRouteEstimatedSeconds(null);
        setRouteSource(null);

        const firstSelectableId = nextMap.selectableNodes[0]?.id;
        const secondSelectableId = nextMap.selectableNodes[1]?.id ?? firstSelectableId;
        setStartId((current) =>
          nextMap.selectableNodes.some((node) => node.id === current)
            ? current
            : firstSelectableId ?? current
        );
        setDestinationId((current) =>
          nextMap.selectableNodes.some((node) => node.id === current)
            ? current
            : secondSelectableId ?? current
        );
      } catch (error) {
        console.warn("지도 API를 사용할 수 없어 로컬 지도 데이터로 전환합니다.", error);
        if (selectedMapId === "default") {
          setVenueMap(fallbackMap);
        }
      }
    }

    void loadMap();
    void refreshCrowdEstimate();

    return () => {
      cancelled = true;
    };
    // 지도 변경 시에만 지도와 혼잡도 추정값을 다시 불러옵니다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMapId]);

  // 출발지, 목적지, 경로 옵션을 바꾸면 별도 버튼 없이 경로를 즉시 다시 계산합니다.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void calculate(true);
    }, 220);

    return () => window.clearTimeout(timer);
    // 혼잡도 입력은 버튼을 눌렀을 때만 다시 예측합니다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    startId,
    destinationId,
    selectedMapId,
    routeOptions.useCongestion,
    routeOptions.useEstimatedCrowd,
    routeOptions.walkingSpeedMps,
  ]);

  return (
    <main className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">LIVE MINIATURE</p>
          <h1>AI 혼잡 예측 기반 실내 길찾기</h1>
          <p className="description">
            출발지와 목적지를 선택하면 경로를 즉시 계산하고 캐릭터가 지도
            위에서 이동 경로를 실시간으로 보여줍니다.
          </p>
        </div>

        <div className="status-card">
          <span>현재 가중치 출처</span>
          <strong>{sourceLabel}</strong>
        </div>
      </header>

      <section className="control-card">
        <div className="field-grid">
          <label>
            지도
            <select
              value={selectedMapId}
              onChange={(event) => setSelectedMapId(event.target.value)}
            >
              {(availableMaps.length > 0
                ? availableMaps
                : [
                    {
                      id: venueMap.id === "fallback" ? "default" : venueMap.id,
                      name: venueMap.name,
                    },
                  ]
              ).map((venueMapOption) => (
                <option key={venueMapOption.id} value={venueMapOption.id}>
                  {venueMapOption.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            출발지
            <select
              value={startId}
              onChange={(event) => setStartId(event.target.value)}
            >
              {venueMap.selectableNodes.map((node) => (
                <option key={node.id} value={node.id}>
                  {node.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            목적지
            <select
              value={destinationId}
              onChange={(event) => setDestinationId(event.target.value)}
            >
              {venueMap.selectableNodes.map((node) => (
                <option key={node.id} value={node.id}>
                  {node.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            로비 인원
            <input
              type="number"
              min="0"
              max="300"
              disabled={routeOptions.useEstimatedCrowd}
              value={inputs.lobbyPeople}
              onChange={(event) =>
                setInputs((previous) => ({
                  ...previous,
                  lobbyPeople: Number(event.target.value),
                }))
              }
            />
          </label>

          <label>
            성과부스 인원
            <input
              type="number"
              min="0"
              max="300"
              disabled={routeOptions.useEstimatedCrowd}
              value={inputs.boothPeople}
              onChange={(event) =>
                setInputs((previous) => ({
                  ...previous,
                  boothPeople: Number(event.target.value),
                }))
              }
            />
          </label>

          <label>
            최근 유입 인원
            <input
              type="number"
              min="0"
              max="150"
              disabled={routeOptions.useEstimatedCrowd}
              value={inputs.recentInflow}
              onChange={(event) =>
                setInputs((previous) => ({
                  ...previous,
                  recentInflow: Number(event.target.value),
                }))
              }
            />
          </label>

          <label>
            행사 단계
            <select
              value={inputs.eventPhase}
              onChange={(event) =>
                setInputs((previous) => ({
                  ...previous,
                  eventPhase: Number(event.target.value) as 0 | 1 | 2 | 3,
                }))
              }
            >
              <option value={0}>행사 전</option>
              <option value={1}>입장 시간</option>
              <option value={2}>행사 진행</option>
              <option value={3}>휴식·이동 시간</option>
            </select>
          </label>

          <label>
            보행 속도
            <input
              type="number"
              min="0.5"
              max="3"
              step="0.1"
              value={routeOptions.walkingSpeedMps}
              onChange={(event) =>
                setRouteOptions((previous) => ({
                  ...previous,
                  walkingSpeedMps: Number(event.target.value),
                }))
              }
            />
          </label>

          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={routeOptions.useCongestion}
              onChange={(event) =>
                setRouteOptions((previous) => ({
                  ...previous,
                  useCongestion: event.target.checked,
                }))
              }
            />
            혼잡도 반영
          </label>

          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={routeOptions.useEstimatedCrowd}
              onChange={(event) =>
                setRouteOptions((previous) => ({
                  ...previous,
                  useEstimatedCrowd: event.target.checked,
                }))
              }
            />
            서버 추정 혼잡도 사용
          </label>
        </div>

        <button type="button" onClick={() => void calculate()} disabled={loading}>
          {loading ? "계산 중..." : "경로 계산"}
        </button>
      </section>

      <section className="telemetry-card">
        <div className="telemetry-summary">
          <div>
            <span>추정 로비 인원</span>
            <strong>{crowdEstimate?.lobbyPeople ?? "-"}</strong>
          </div>
          <div>
            <span>추정 부스 인원</span>
            <strong>{crowdEstimate?.boothPeople ?? "-"}</strong>
          </div>
          <div>
            <span>최근 유입 신호</span>
            <strong>{crowdEstimate?.recentInflow ?? "-"}</strong>
          </div>
          <div>
            <span>QR / 목적지 요청</span>
            <strong>
              {crowdEstimate
                ? `${crowdEstimate.signals.qrScans.lobby + crowdEstimate.signals.qrScans.booth} / ${
                    crowdEstimate.signals.routeIntents.lobby +
                    crowdEstimate.signals.routeIntents.booth
                  }`
                : "-"}
            </strong>
          </div>
        </div>

        <div className="telemetry-actions">
          <button type="button" className="secondary-button" onClick={() => void saveManualCrowd()}>
            운영자 입력 저장
          </button>
          <button type="button" className="ghost-button" onClick={() => void addQrScan("lobby")}>
            로비 QR +1
          </button>
          <button type="button" className="ghost-button" onClick={() => void addQrScan("booth")}>
            부스 QR +1
          </button>
          <button type="button" className="ghost-button" onClick={() => void refreshCrowdEstimate()}>
            추정값 새로고침
          </button>
        </div>
      </section>

      <section className="live-guidance-card">
        <div className="guidance-main">
          <div className={`live-indicator live-indicator--${guidance.status}`}>
            <span className="live-dot" />
            {statusText(guidance.status)}
          </div>

          <div>
            <span className="guidance-caption">다음 안내 지점</span>
            <strong className="guidance-target">{currentTargetName}</strong>
          </div>

          <div>
            <span className="guidance-caption">남은 거리</span>
            <strong className="guidance-target">
              {route ? `${remainingDistance.toFixed(1)}m` : "-"}
            </strong>
          </div>

          <div>
            <span className="guidance-caption">예상 남은 시간</span>
            <strong className="guidance-target">
              {route ? `약 ${estimatedSeconds}초` : "-"}
            </strong>
          </div>
        </div>

        <div className="guidance-actions">
          {guidance.status === "running" ? (
            <button type="button" className="secondary-button" onClick={guidance.pause}>
              일시정지
            </button>
          ) : (
            <button type="button" className="secondary-button" onClick={guidance.start}>
              {guidance.status === "arrived" ? "다시 안내" : "안내 시작"}
            </button>
          )}
          <button type="button" className="ghost-button" onClick={guidance.reset}>
            처음으로
          </button>
        </div>

        <div className="progress-track" aria-label="이동 진행률">
          <div
            className="progress-fill"
            style={{ width: `${guidance.progressPercent}%` }}
          />
        </div>
      </section>

      <section className="map-card">
        <svg
          className="map"
          viewBox={`0 0 ${venueMap.width} ${venueMap.height}`}
          aria-label="행사장 평면도와 실시간 이동 안내"
        >
          <image
            href={venueMap.image}
            x="0"
            y="0"
            width={venueMap.width}
            height={venueMap.height}
            preserveAspectRatio="none"
          />

          {routePoints && <polyline points={routePoints} className="route-line" />}

          {guidance.position && (
            <g
              className={`traveler traveler--${guidance.status}`}
              transform={`translate(${guidance.position.x} ${guidance.position.y})`}
            >
              <circle className="traveler-halo" r="19" />
              <circle className="traveler-body" r="13" />
              <text className="traveler-icon" x="0" y="6" textAnchor="middle">
                ●
              </text>
            </g>
          )}
        </svg>
      </section>

      <section className="result-grid">
        <article className="result-card">
          <span>선택된 경로</span>
          <strong>
            {routeInstructions.length > 0
              ? routeInstructions.join(" ")
              : route
                ? route.path
                    .map((nodeId) => nodeMap.get(nodeId)?.name ?? nodeId)
                    .join(" → ")
              : "경로를 찾지 못했습니다."}
          </strong>
        </article>

        <article className="result-card">
          <span>기본 이동 거리</span>
          <strong>{route ? `${route.totalDistance}m` : "-"}</strong>
        </article>

        <article className="result-card">
          <span>혼잡도 반영 비용</span>
          <strong>{route ? route.weightedCost : "-"}</strong>
        </article>
      </section>

      <section className="legend">
        <span><i className="legend-dot legend-dot--route" />선택 경로</span>
      </section>
    </main>
  );
}
