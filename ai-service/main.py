from datetime import datetime
import math
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crowd_store import (
    DEFAULT_EVENT_PHASE,
    EVENT_WINDOW_MINUTES,
    create_navigation_session,
    estimate_crowd_inputs,
    get_navigation_session,
    navigation_updates_chronological,
    parse_timestamp,
    record_qr_scan,
    record_route_intent,
    recent_navigation_updates,
    region_for_destination,
    set_manual_estimate,
    telemetry_snapshot,
    update_navigation_session_position,
)
from generation_store import (
    GENERATED_ASSETS_DIR,
    attach_draft_map,
    create_generation_job,
    get_draft_map,
    get_generation_job,
    init_generation_store,
    list_generation_jobs,
    save_generation_job_map,
)
from llm_map_generator import configured_openai_map_model, request_openai_map_draft
from map_postprocess import postprocess_map_data
from map_store import (
    delete_map_data,
    list_maps,
    load_map,
    save_map_data,
    validate_map,
    validate_map_data,
)
from route_engine import calculate_route_alternatives, calculate_shortest_route

MODEL_PATH = Path(__file__).with_name("congestion_model.joblib")
STATIC_DIR = Path(__file__).with_name("static")
WALKING_SPEED_MPS = 1.2
MAX_PEOPLE_COUNT = 1000
MAX_RECENT_INFLOW = 1000
MAX_QR_COUNT = 100
MAX_WALKING_SPEED_MPS = 3.0
DEFAULT_MAP_ID = "default"
COORDINATE_NODE_LOCK_PX = 32
RoutePreference = Literal["shortest", "less_crowded", "accessible", "fewest_turns"]
SearchAlgorithm = Literal["astar", "dijkstra"]
MultiStopOrderAlgorithm = Literal["optimal", "nearest"]
TEMPORARY_BLOCKED_EDGES: dict[str, dict[str, str | None]] = {}


class EdgeFeatures(BaseModel):
    edge_id: str
    people_count: int = Field(ge=0, le=1000)
    corridor_width_m: float = Field(gt=0, le=30)
    hour: int = Field(ge=0, le=23)
    event_phase: int = Field(ge=0, le=3)
    booth_zone: int = Field(ge=0, le=1)
    recent_inflow: int = Field(ge=0, le=1000)


class PredictionRequest(BaseModel):
    edges: list[EdgeFeatures]


class EdgePrediction(BaseModel):
    edge_id: str
    multiplier: float
    level: Literal["clear", "normal", "busy", "very_busy"]


class PredictionResponse(BaseModel):
    model_type: str
    validation_mae: float
    predictions: list[EdgePrediction]


class RoutePoint(BaseModel):
    node_id: str
    name: str
    x: float
    y: float


class RouteSegment(BaseModel):
    edge_id: str
    from_node: RoutePoint
    to_node: RoutePoint
    distance: float
    multiplier: float
    weighted_cost: float
    estimated_seconds: int
    level: Literal["clear", "normal", "busy", "very_busy"]
    maneuver: Literal["start", "straight", "slight_left", "left", "slight_right", "right", "u_turn", "arrive"]
    heading_degrees: int
    turn_degrees: int
    instruction: str


class CrowdInputs(BaseModel):
    lobby_people: int = Field(ge=0, le=1000)
    booth_people: int = Field(ge=0, le=1000)
    recent_inflow: int = Field(ge=0, le=1000)
    hour: int = Field(ge=0, le=23)
    event_phase: int = Field(ge=0, le=3)


class RouteRequest(BaseModel):
    map_id: str = "default"
    start_id: str
    destination_id: str
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    use_congestion: bool = True
    log_route_intent: bool = True
    preference: RoutePreference = "shortest"
    algorithm: SearchAlgorithm = "astar"
    blocked_edge_ids: list[str] = Field(default_factory=list)


class RouteAlternativesRequest(RouteRequest):
    max_routes: int = Field(default=3, ge=1, le=5)
    overlap_penalty: float = Field(default=1.8, ge=1.1, le=5)


class RouteMultiStopRequest(BaseModel):
    map_id: str = "default"
    start_id: str
    destination_ids: list[str] = Field(min_length=1, max_length=8)
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    use_congestion: bool = True
    log_route_intent: bool = True
    preference: RoutePreference = "shortest"
    algorithm: SearchAlgorithm = "astar"
    order_algorithm: MultiStopOrderAlgorithm = "optimal"
    blocked_edge_ids: list[str] = Field(default_factory=list)


class RouteSimulationScenario(BaseModel):
    id: str
    start_id: str
    destination_id: str
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float | None = Field(default=None, gt=0, le=3)
    use_congestion: bool | None = None
    preference: RoutePreference | None = None
    algorithm: SearchAlgorithm | None = None
    blocked_edge_ids: list[str] = Field(default_factory=list)


class RouteSimulationRequest(BaseModel):
    map_id: str = "default"
    scenarios: list[RouteSimulationScenario] = Field(min_length=1, max_length=50)
    default_crowd_inputs: CrowdInputs | None = None
    default_walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    default_use_congestion: bool = True
    default_preference: RoutePreference = "shortest"
    default_algorithm: SearchAlgorithm = "astar"
    default_blocked_edge_ids: list[str] = Field(default_factory=list)


class BlockedEdgesRequest(BaseModel):
    edge_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class PositionUpdateRequest(BaseModel):
    map_id: str = "default"
    current_node_id: str | None = None
    destination_id: str
    checkpoint_id: str | None = None
    region: Literal["lobby", "booth"] | None = None
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    use_congestion: bool = True
    log_route_intent: bool = True
    preference: RoutePreference = "shortest"
    algorithm: SearchAlgorithm = "astar"
    blocked_edge_ids: list[str] = Field(default_factory=list)


class CoordinateSnapRequest(BaseModel):
    x: float
    y: float
    max_distance_px: float = Field(default=120, ge=0, le=2000)
    selectable_only: bool = False


class NavigationSessionStartRequest(BaseModel):
    map_id: str = "default"
    start_id: str
    destination_id: str
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    use_congestion: bool = True
    log_route_intent: bool = True
    preference: RoutePreference = "shortest"
    algorithm: SearchAlgorithm = "astar"
    blocked_edge_ids: list[str] = Field(default_factory=list)


class NavigationSessionUpdateRequest(BaseModel):
    current_node_id: str | None = None
    checkpoint_id: str | None = None
    region: Literal["lobby", "booth"] | None = None
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    use_congestion: bool = True
    log_route_intent: bool = False
    preference: RoutePreference = "shortest"
    algorithm: SearchAlgorithm = "astar"
    blocked_edge_ids: list[str] = Field(default_factory=list)


class NavigationSessionCoordinateUpdateRequest(BaseModel):
    x: float
    y: float
    max_distance_px: float = Field(default=120, ge=0, le=2000)
    region: Literal["lobby", "booth"] | None = None
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    use_congestion: bool = True
    log_route_intent: bool = False
    preference: RoutePreference = "shortest"
    algorithm: SearchAlgorithm = "astar"
    blocked_edge_ids: list[str] = Field(default_factory=list)


class QrScanRequest(BaseModel):
    map_id: str = "default"
    checkpoint_id: str
    region: Literal["lobby", "booth"]
    count: int = Field(default=1, ge=1, le=100)


class ManualCrowdRequest(BaseModel):
    map_id: str = "default"
    lobby_people: int = Field(ge=0, le=1000)
    booth_people: int = Field(ge=0, le=1000)


class ApiEndpoint(BaseModel):
    method: Literal["GET", "POST", "DELETE"]
    path: str


class AppConfigResponse(BaseModel):
    api_version: str
    default_map_id: str
    default_walking_speed_mps: float
    max_walking_speed_mps: float
    telemetry_window_minutes: int
    limits: dict[str, int]
    event_phase_labels: dict[int, str]
    features: dict[str, bool]
    endpoints: list[ApiEndpoint]


class MapSchemaResponse(BaseModel):
    required_fields: list[str]
    node_fields: list[str]
    edge_fields: list[str]
    checkpoint_fields: list[str]
    node_types: list[str]
    zones: list[str]
    crowd_regions: list[str]
    checkpoint_regions: list[str]
    coordinate_system: str
    notes: list[str]


class MapSaveRequest(BaseModel):
    venue_map: dict[str, Any]
    overwrite: bool = False


class MapSaveResponse(BaseModel):
    map_id: str
    saved: bool
    overwritten: bool
    path: str
    validation: dict[str, Any]


class MapDeleteResponse(BaseModel):
    map_id: str
    deleted: bool
    path: str


class MapGenerationJobCreateRequest(BaseModel):
    source_image_base64: str
    filename: str = "floorplan.png"
    mime_type: Literal["image/png", "image/jpeg"]
    target_map_id: str
    notes: str | None = None


class MapGenerationDraftRequest(BaseModel):
    venue_map: dict[str, Any]


class MapGenerationRoutePreviewRequest(BaseModel):
    start_id: str | None = None
    current_node_id: str | None = None
    checkpoint_id: str | None = None
    destination_id: str
    crowd_inputs: CrowdInputs | None = None
    walking_speed_mps: float = Field(default=WALKING_SPEED_MPS, gt=0, le=3)
    use_congestion: bool = True
    preference: RoutePreference = "shortest"
    algorithm: SearchAlgorithm = "astar"
    blocked_edge_ids: list[str] = Field(default_factory=list)


class MapGenerationPostprocessRequest(BaseModel):
    apply: bool = True
    recalculate_edge_distance: bool = True
    clamp_coordinates: bool = True
    close_node_threshold_px: float = Field(default=18, ge=0, le=100)
    suggest_connection_edges: bool = True
    apply_connection_suggestions: bool = False
    max_connection_suggestions: int = Field(default=5, ge=1, le=20)
    max_connection_distance_px: float = Field(default=260, ge=0, le=2000)


class MapGenerationSaveRequest(BaseModel):
    overwrite: bool = False


class MapGenerationGenerateDraftRequest(BaseModel):
    model: str | None = None
    image_detail: Literal["low", "high", "auto"] = "auto"
    extra_instructions: str | None = None


class MapGenerationJobResponse(BaseModel):
    job_id: str
    status: Literal["needs_llm", "draft_valid", "draft_invalid", "map_saved"]
    target_map_id: str
    source_filename: str
    source_mime_type: str
    source_image_path: str
    source_image_url: str
    notes: str | None
    created_at: str
    updated_at: str
    validation: dict[str, Any] | None
    saved_map_id: str | None
    next_step: str


class MapGenerationJobsResponse(BaseModel):
    jobs: list[MapGenerationJobResponse]


class MapGenerationSaveResponse(BaseModel):
    job: MapGenerationJobResponse
    save_result: MapSaveResponse


class MapGenerationDraftGenerateResponse(BaseModel):
    job: MapGenerationJobResponse
    draft_map: dict[str, Any]


class MapGenerationDraftResponse(BaseModel):
    job: MapGenerationJobResponse
    draft_map: dict[str, Any]
    validation: dict[str, Any]


class MapGenerationPostprocessResponse(BaseModel):
    job: MapGenerationJobResponse
    processed_map: dict[str, Any]
    changes: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    validation: dict[str, Any]
    applied: bool


class RouteResponse(BaseModel):
    map_id: str
    use_congestion: bool
    walking_speed_mps: float
    preference: RoutePreference
    algorithm: SearchAlgorithm
    expanded_state_count: int
    blocked_edge_ids: list[str]
    path: list[str]
    edge_ids: list[str]
    route_points: list[RoutePoint]
    segments: list[RouteSegment]
    total_distance: float
    weighted_cost: float
    estimated_seconds: int
    instructions: list[str]
    crowd_inputs: CrowdInputs
    model_type: str
    validation_mae: float
    predictions: list[EdgePrediction]


class RouteAlternative(BaseModel):
    rank: int
    algorithm: SearchAlgorithm
    expanded_state_count: int
    overlap_ratio: float
    detour_ratio: float
    quality_score: int
    path: list[str]
    edge_ids: list[str]
    route_points: list[RoutePoint]
    segments: list[RouteSegment]
    total_distance: float
    weighted_cost: float
    estimated_seconds: int
    instructions: list[str]
    overlap_with_best_edge_count: int


class RouteAlternativesResponse(BaseModel):
    map_id: str
    start_id: str
    destination_id: str
    use_congestion: bool
    walking_speed_mps: float
    preference: RoutePreference
    algorithm: SearchAlgorithm
    blocked_edge_ids: list[str]
    crowd_inputs: CrowdInputs
    alternatives: list[RouteAlternative]
    model_type: str
    validation_mae: float
    predictions: list[EdgePrediction]


class RouteRecommendation(BaseModel):
    selected_rank: int
    preference: RoutePreference
    recommendation_score: int
    reasons: list[str]
    selection_metrics: dict[str, Any]
    tradeoffs: dict[str, Any]


class RouteRecommendationResponse(BaseModel):
    map_id: str
    start_id: str
    destination_id: str
    recommendation: RouteRecommendation
    selected: RouteAlternative
    alternatives: list[RouteAlternative]
    predictions: list[EdgePrediction]


class MultiStopLeg(BaseModel):
    leg_index: int
    start_id: str
    destination_id: str
    route: RouteResponse


class RouteMultiStopResponse(BaseModel):
    map_id: str
    start_id: str
    order_algorithm: MultiStopOrderAlgorithm
    pairwise_route_count: int
    ordered_destination_ids: list[str]
    total_distance: float
    weighted_cost: float
    estimated_seconds: int
    legs: list[MultiStopLeg]
    crowd_inputs: CrowdInputs


class RouteSimulationResult(BaseModel):
    scenario_id: str
    ok: bool
    start_id: str
    destination_id: str
    preference: RoutePreference
    algorithm: SearchAlgorithm
    expanded_state_count: int | None = None
    blocked_edge_ids: list[str]
    path: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    total_distance: float | None = None
    weighted_cost: float | None = None
    estimated_seconds: int | None = None
    error: dict[str, Any] | None = None


class RouteSimulationResponse(BaseModel):
    map_id: str
    scenario_count: int
    success_count: int
    failure_count: int
    pass_rate: float
    failures: list[str]
    results: list[RouteSimulationResult]


class DemandHeatRegion(BaseModel):
    region: str
    people_estimate: int
    qr_scans: int
    route_intents: int
    demand_score: float


class DemandHeatEdge(BaseModel):
    edge_id: str
    from_node: str
    to_node: str
    zone: str
    crowd_region: str
    multiplier: float
    level: Literal["clear", "normal", "busy", "very_busy"]
    demand_score: float
    heat_level: Literal["low", "medium", "high", "critical"]


class DemandHeatmapResponse(BaseModel):
    map_id: str
    method: str
    crowd_inputs: CrowdInputs
    regions: list[DemandHeatRegion]
    edges: list[DemandHeatEdge]


class BlockedEdgesResponse(BaseModel):
    map_id: str
    blocked_edge_ids: list[str]
    reason: str | None = None


class EdgeRiskItem(BaseModel):
    edge_id: str
    from_node: str
    to_node: str
    zone: str
    crowd_region: str
    affected_pair_count: int
    disconnected_pair_count: int
    average_detour_ratio: float
    max_detour_ratio: float
    risk_score: float
    risk_level: Literal["low", "medium", "high", "critical"]
    sample_pairs: list[dict[str, Any]]


class RouteRiskAnalysisResponse(BaseModel):
    map_id: str
    selectable_node_count: int
    analyzed_pair_count: int
    analyzed_edge_count: int
    critical_edge_count: int
    high_risk_edge_count: int
    edges: list[EdgeRiskItem]


class TopologyIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    target_ids: list[str]
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MapTopologyAnalysisResponse(BaseModel):
    map_id: str
    score: int
    readiness: Literal["ready", "needs_review", "blocked"]
    node_count: int
    edge_count: int
    checkpoint_count: int
    dead_end_node_ids: list[str]
    low_degree_junction_ids: list[str]
    duplicate_edge_groups: list[list[str]]
    long_edge_ids: list[str]
    checkpoint_gap_node_ids: list[str]
    issues: list[TopologyIssue]


class MapImprovementSuggestion(BaseModel):
    code: str
    priority: Literal["low", "medium", "high", "critical"]
    source: Literal["validation", "postprocess", "topology", "risk"]
    target_ids: list[str]
    message: str
    action: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MapImprovementSuggestionsResponse(BaseModel):
    map_id: str
    readiness: Literal["ready", "needs_review", "blocked"]
    score: int
    suggestion_count: int
    critical_count: int
    high_count: int
    suggestions: list[MapImprovementSuggestion]
    validation_summary: dict[str, Any]
    topology_summary: dict[str, Any]
    risk_summary: dict[str, Any]


class PositionUpdateResponse(BaseModel):
    map_id: str
    current_node: RoutePoint
    destination_node: RoutePoint
    location_source: Literal["node", "checkpoint"]
    route: RouteResponse


class CoordinateSnapResponse(BaseModel):
    map_id: str
    input_x: float
    input_y: float
    snapped_node: RoutePoint
    distance_px: float
    within_threshold: bool
    selectable: bool
    snapped_edge_id: str | None = None
    snapped_edge_from_node_id: str | None = None
    snapped_edge_to_node_id: str | None = None
    edge_distance_px: float | None = None
    edge_progress_ratio: float | None = None
    projected_x: float | None = None
    projected_y: float | None = None


class NavigationUpdateEvent(BaseModel):
    node_id: str
    source: str
    checkpoint_id: str | None
    timestamp: str


class NavigationSessionResponse(BaseModel):
    session_id: str
    map_id: str
    start_node: RoutePoint
    current_node: RoutePoint
    destination_node: RoutePoint
    status: Literal["active", "arrived"]
    created_at: str
    updated_at: str
    route: RouteResponse
    recent_updates: list[NavigationUpdateEvent]


class NavigationGuidanceResponse(BaseModel):
    session_id: str
    map_id: str
    status: Literal["active", "arrived"]
    current_node: RoutePoint
    destination_node: RoutePoint
    off_route: bool
    expected_path: list[str]
    expected_edge_ids: list[str]
    progress_ratio: float
    remaining_distance: float
    remaining_seconds: int
    next_node: RoutePoint | None
    next_segment: RouteSegment | None
    instruction: str
    route: RouteResponse


class EtaCalibrationSample(BaseModel):
    from_node_id: str
    to_node_id: str
    distance: float
    elapsed_seconds: int
    observed_speed_mps: float


class EtaCalibrationResponse(BaseModel):
    session_id: str
    map_id: str
    sample_count: int
    confidence: float
    default_walking_speed_mps: float
    observed_walking_speed_mps: float | None
    recommended_walking_speed_mps: float
    samples: list[EtaCalibrationSample]


def level_from_multiplier(value: float):
    if value < 1.2:
        return "clear"
    if value < 1.65:
        return "normal"
    if value < 2.25:
        return "busy"
    return "very_busy"


if not MODEL_PATH.exists():
    from train_model import train_and_save

    train_and_save()

payload = joblib.load(MODEL_PATH)
model = payload["model"]
init_generation_store()

app = FastAPI(
    title="LIVE MINIATURE Congestion AI",
    version="0.1.0",
    description=(
        "시뮬레이션 데이터로 초기 학습된 통로 혼잡 가중치 예측 API입니다."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount(
    "/generated-assets",
    StaticFiles(directory=GENERATED_ASSETS_DIR),
    name="generated-assets",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": payload["training_type"],
        "validation_mae": payload["validation_mae"],
    }


@app.get("/app-config", response_model=AppConfigResponse)
def get_app_config():
    return AppConfigResponse(
        api_version=app.version,
        default_map_id=DEFAULT_MAP_ID,
        default_walking_speed_mps=WALKING_SPEED_MPS,
        max_walking_speed_mps=MAX_WALKING_SPEED_MPS,
        telemetry_window_minutes=EVENT_WINDOW_MINUTES,
        limits={
            "people_count": MAX_PEOPLE_COUNT,
            "recent_inflow": MAX_RECENT_INFLOW,
            "qr_scan_count": MAX_QR_COUNT,
            "telemetry_snapshot_limit": 100,
        },
        event_phase_labels={
            0: "행사 전",
            1: "입장 시간",
            2: "행사 진행",
            3: "휴식·이동 시간",
        },
        features={
            "route_api": True,
            "congestion_prediction": True,
            "estimated_crowd_without_camera": True,
            "manual_crowd_input": True,
            "qr_scan_telemetry": True,
            "route_intent_telemetry": True,
            "checkpoint_position_update": True,
            "coordinate_position_snap": True,
            "llm_map_validation": True,
            "llm_map_storage": True,
            "llm_map_generation_jobs": True,
            "llm_map_generate_draft": True,
            "route_preferences": True,
            "astar_route_search": True,
            "temporary_blocked_edges": True,
            "route_alternatives": True,
            "route_recommendation": True,
            "multi_stop_route": True,
            "crowd_forecast": True,
            "bottleneck_detection": True,
            "eta_calibration": True,
            "route_risk_analysis": True,
            "map_topology_analysis": True,
            "route_simulation": True,
            "demand_heatmap": True,
            "navigation_guidance": True,
            "map_improvement_suggestions": True,
        },
        endpoints=[
            ApiEndpoint(method="GET", path="/health"),
            ApiEndpoint(method="GET", path="/maps"),
            ApiEndpoint(method="GET", path="/maps/{map_id}"),
            ApiEndpoint(method="GET", path="/maps/{map_id}/checkpoints"),
            ApiEndpoint(method="GET", path="/maps/{map_id}/validate"),
            ApiEndpoint(method="POST", path="/maps/{map_id}/snap-coordinate"),
            ApiEndpoint(method="GET", path="/maps/{map_id}/improvement-suggestions"),
            ApiEndpoint(method="GET", path="/maps/{map_id}/topology-analysis"),
            ApiEndpoint(method="GET", path="/maps/{map_id}/route-risk-analysis"),
            ApiEndpoint(method="GET", path="/maps/schema"),
            ApiEndpoint(method="POST", path="/maps/validate-data"),
            ApiEndpoint(method="POST", path="/maps"),
            ApiEndpoint(method="DELETE", path="/maps/{map_id}"),
            ApiEndpoint(method="GET", path="/map-generation/jobs?limit=20"),
            ApiEndpoint(method="POST", path="/map-generation/jobs"),
            ApiEndpoint(method="GET", path="/map-generation/jobs/{job_id}"),
            ApiEndpoint(method="POST", path="/map-generation/jobs/{job_id}/generate-draft"),
            ApiEndpoint(method="GET", path="/map-generation/jobs/{job_id}/draft-map"),
            ApiEndpoint(method="POST", path="/map-generation/jobs/{job_id}/draft-map"),
            ApiEndpoint(method="POST", path="/map-generation/jobs/{job_id}/postprocess-draft"),
            ApiEndpoint(method="POST", path="/map-generation/jobs/{job_id}/route-preview"),
            ApiEndpoint(method="POST", path="/map-generation/jobs/{job_id}/save-map"),
            ApiEndpoint(method="GET", path="/crowd/{map_id}"),
            ApiEndpoint(method="GET", path="/crowd/{map_id}/forecast"),
            ApiEndpoint(method="GET", path="/crowd/{map_id}/bottlenecks"),
            ApiEndpoint(method="GET", path="/crowd/{map_id}/demand-heatmap"),
            ApiEndpoint(method="GET", path="/telemetry/{map_id}?limit=20"),
            ApiEndpoint(method="POST", path="/telemetry/manual-crowd"),
            ApiEndpoint(method="POST", path="/telemetry/qr-scan"),
            ApiEndpoint(method="POST", path="/predict-congestion"),
            ApiEndpoint(method="POST", path="/route"),
            ApiEndpoint(method="POST", path="/route-alternatives"),
            ApiEndpoint(method="POST", path="/route-recommendation"),
            ApiEndpoint(method="POST", path="/route-multi-stop"),
            ApiEndpoint(method="POST", path="/route-simulation"),
            ApiEndpoint(method="GET", path="/maps/{map_id}/blocked-edges"),
            ApiEndpoint(method="POST", path="/maps/{map_id}/blocked-edges"),
            ApiEndpoint(method="POST", path="/navigation/update-position"),
            ApiEndpoint(method="POST", path="/navigation/sessions"),
            ApiEndpoint(method="GET", path="/navigation/sessions/{session_id}"),
            ApiEndpoint(method="GET", path="/navigation/sessions/{session_id}/guidance"),
            ApiEndpoint(method="GET", path="/navigation/sessions/{session_id}/eta-calibration"),
            ApiEndpoint(method="POST", path="/navigation/sessions/{session_id}/position"),
            ApiEndpoint(
                method="POST",
                path="/navigation/sessions/{session_id}/position-coordinate",
            ),
        ],
    )


@app.post("/predict-congestion", response_model=PredictionResponse)
def predict_congestion(request: PredictionRequest):
    if not request.edges:
        return PredictionResponse(
            model_type=payload["training_type"],
            validation_mae=payload["validation_mae"],
            predictions=[],
        )

    matrix = np.array(
        [
            [
                edge.people_count,
                edge.corridor_width_m,
                edge.hour,
                edge.event_phase,
                edge.booth_zone,
                edge.recent_inflow,
            ]
            for edge in request.edges
        ],
        dtype=float,
    )

    raw_predictions = model.predict(matrix)

    predictions = [
        EdgePrediction(
            edge_id=edge.edge_id,
            multiplier=round(float(np.clip(value, 1.0, 3.5)), 2),
            level=level_from_multiplier(float(value)),
        )
        for edge, value in zip(request.edges, raw_predictions)
    ]

    return PredictionResponse(
        model_type=payload["training_type"],
        validation_mae=payload["validation_mae"],
        predictions=predictions,
    )


def people_count_for_edge(edge: dict, crowd_inputs: CrowdInputs) -> int:
    if edge["crowdRegion"] == "booth":
        return crowd_inputs.booth_people
    return crowd_inputs.lobby_people


def route_points_for_path(venue_map: dict, path: list[str]) -> list[RoutePoint]:
    node_map = {node["id"]: node for node in venue_map["nodes"]}
    return [
        RoutePoint(
            node_id=node_id,
            name=node_map[node_id]["name"],
            x=node_map[node_id]["x"],
            y=node_map[node_id]["y"],
        )
        for node_id in path
    ]


def point_for_node(node: dict) -> RoutePoint:
    return RoutePoint(
        node_id=node["id"],
        name=node["name"],
        x=node["x"],
        y=node["y"],
    )


def heading_degrees(from_node: dict[str, Any], to_node: dict[str, Any]) -> int:
    dx = float(to_node["x"]) - float(from_node["x"])
    dy = float(to_node["y"]) - float(from_node["y"])
    # Image coordinates increase downward, so invert dy to keep 0 degrees as north.
    degrees = math.degrees(math.atan2(dx, -dy))
    return round((degrees + 360) % 360)


def signed_turn_degrees(previous_heading: int | None, current_heading: int) -> int:
    if previous_heading is None:
        return 0
    delta = (current_heading - previous_heading + 540) % 360 - 180
    return round(delta)


def maneuver_from_turn(
    turn_degrees: int,
    *,
    is_first: bool,
) -> Literal["start", "straight", "slight_left", "left", "slight_right", "right", "u_turn", "arrive"]:
    if is_first:
        return "start"
    abs_turn = abs(turn_degrees)
    if abs_turn < 30:
        return "straight"
    if abs_turn >= 145:
        return "u_turn"
    if turn_degrees < 0:
        return "left" if abs_turn >= 70 else "slight_left"
    return "right" if abs_turn >= 70 else "slight_right"


def maneuver_phrase(
    maneuver: str,
    from_name: str,
    to_name: str,
    *,
    is_last: bool,
) -> str:
    if maneuver == "start":
        prefix = f"{from_name}에서 {to_name} 방향으로 출발하세요."
    elif maneuver == "straight":
        prefix = f"{from_name}에서 직진해 {to_name} 방향으로 이동하세요."
    elif maneuver == "slight_left":
        prefix = f"{from_name}에서 살짝 왼쪽으로 이동해 {to_name} 방향으로 가세요."
    elif maneuver == "left":
        prefix = f"{from_name}에서 좌회전해 {to_name} 방향으로 이동하세요."
    elif maneuver == "slight_right":
        prefix = f"{from_name}에서 살짝 오른쪽으로 이동해 {to_name} 방향으로 가세요."
    elif maneuver == "right":
        prefix = f"{from_name}에서 우회전해 {to_name} 방향으로 이동하세요."
    elif maneuver == "u_turn":
        prefix = f"{from_name}에서 방향을 크게 전환해 {to_name} 방향으로 이동하세요."
    else:
        prefix = f"{from_name}에서 {to_name} 방향으로 이동하세요."

    if is_last:
        return f"{prefix} 목적지는 {to_name}입니다."
    return prefix


def route_segments_for_edges(
    venue_map: dict,
    path: list[str],
    edge_ids: list[str],
    predictions: list[EdgePrediction],
    walking_speed_mps: float,
) -> list[RouteSegment]:
    node_map = {node["id"]: node for node in venue_map["nodes"]}
    edge_map = {edge["id"]: edge for edge in venue_map["edges"]}
    prediction_map = {prediction.edge_id: prediction for prediction in predictions}
    segments: list[RouteSegment] = []
    previous_heading: int | None = None

    for index, edge_id in enumerate(edge_ids):
        edge = edge_map[edge_id]
        prediction = prediction_map[edge_id]
        distance = float(edge["distance"])
        from_id = path[index]
        to_id = path[index + 1]
        current_heading = heading_degrees(node_map[from_id], node_map[to_id])
        turn_degrees = signed_turn_degrees(previous_heading, current_heading)
        maneuver = maneuver_from_turn(turn_degrees, is_first=index == 0)
        is_last = index == len(edge_ids) - 1
        segments.append(
            RouteSegment(
                edge_id=edge_id,
                from_node=point_for_node(node_map[from_id]),
                to_node=point_for_node(node_map[to_id]),
                distance=distance,
                multiplier=prediction.multiplier,
                weighted_cost=round(distance * prediction.multiplier, 1),
                estimated_seconds=round(distance / walking_speed_mps),
                level=prediction.level,
                maneuver=maneuver,
                heading_degrees=current_heading,
                turn_degrees=turn_degrees,
                instruction=maneuver_phrase(
                    maneuver,
                    node_map[from_id]["name"],
                    node_map[to_id]["name"],
                    is_last=is_last,
                ),
            )
        )
        previous_heading = current_heading

    return segments


def estimated_seconds(distance: float, walking_speed_mps: float) -> int:
    return round(distance / walking_speed_mps)


def validate_route_nodes(venue_map: dict, start_id: str, destination_id: str):
    node_ids = {node["id"] for node in venue_map["nodes"]}
    missing = [
        node_id
        for node_id in [start_id, destination_id]
        if node_id not in node_ids
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown route node id.",
                "missing_node_ids": missing,
                "available_node_ids": sorted(node_ids),
            },
        )


def validate_route_node_ids(venue_map: dict, node_ids: list[str]):
    available_node_ids = {node["id"] for node in venue_map["nodes"]}
    missing = [node_id for node_id in node_ids if node_id not in available_node_ids]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown route node id.",
                "missing_node_ids": missing,
                "available_node_ids": sorted(available_node_ids),
            },
        )


def validate_edge_ids(venue_map: dict, edge_ids: list[str]):
    available_edge_ids = {edge["id"] for edge in venue_map["edges"]}
    missing = [edge_id for edge_id in edge_ids if edge_id not in available_edge_ids]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown edge id.",
                "missing_edge_ids": missing,
                "available_edge_ids": sorted(available_edge_ids),
            },
        )


def active_blocked_edge_ids(venue_map: dict, requested_edge_ids: list[str]) -> set[str]:
    configured_edge_ids = sorted(TEMPORARY_BLOCKED_EDGES.get(venue_map["id"], {}))
    all_blocked_edge_ids = sorted(set(configured_edge_ids) | set(requested_edge_ids))
    validate_edge_ids(venue_map, all_blocked_edge_ids)
    return set(all_blocked_edge_ids)


def preference_turn_penalty(preference: RoutePreference) -> float:
    if preference == "fewest_turns":
        return 4.0
    return 0.0


def preference_multiplier(edge: dict[str, Any], preference: RoutePreference) -> float:
    if preference == "accessible":
        width = float(edge["widthM"])
        if width < 2.2:
            return 3.0
        if width < 2.8:
            return 1.8
        return 1.0

    if preference == "less_crowded":
        width = float(edge["widthM"])
        return 1.0 + max(0.0, 3.5 - width) * 0.18

    return 1.0


def route_cost_multipliers(
    edges: list[dict[str, Any]],
    predictions: list[EdgePrediction],
    preference: RoutePreference,
) -> dict[str, float]:
    prediction_map = {prediction.edge_id: prediction for prediction in predictions}
    multipliers: dict[str, float] = {}
    for edge in edges:
        prediction = prediction_map[edge["id"]]
        multiplier = prediction.multiplier
        if preference == "less_crowded":
            multiplier = multiplier**1.35
        multipliers[edge["id"]] = multiplier * preference_multiplier(edge, preference)
    return multipliers


def node_by_id(venue_map: dict, node_id: str) -> dict:
    for node in venue_map["nodes"]:
        if node["id"] == node_id:
            return node

    raise HTTPException(
        status_code=400,
        detail={
            "message": "Unknown node id.",
            "missing_node_ids": [node_id],
            "available_node_ids": sorted(node["id"] for node in venue_map["nodes"]),
        },
    )


def checkpoint_by_id(venue_map: dict, checkpoint_id: str) -> dict:
    for checkpoint in venue_map.get("checkpoints", []):
        if checkpoint["id"] == checkpoint_id:
            return checkpoint

    raise HTTPException(
        status_code=400,
        detail={
            "message": "Unknown checkpoint id.",
            "checkpoint_id": checkpoint_id,
            "available_checkpoint_ids": sorted(
                checkpoint["id"] for checkpoint in venue_map.get("checkpoints", [])
            ),
        },
    )


def resolve_position_node(
    venue_map: dict,
    current_node_id: str | None,
    checkpoint_id: str | None,
) -> tuple[dict, dict | None]:
    if checkpoint_id:
        checkpoint = checkpoint_by_id(venue_map, checkpoint_id)
        if current_node_id and current_node_id != checkpoint["node_id"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "current_node_id does not match checkpoint node.",
                    "current_node_id": current_node_id,
                    "checkpoint_node_id": checkpoint["node_id"],
                    "checkpoint_id": checkpoint_id,
                },
            )
        return node_by_id(venue_map, checkpoint["node_id"]), checkpoint

    if current_node_id:
        return node_by_id(venue_map, current_node_id), None

    raise HTTPException(
        status_code=400,
        detail="Either current_node_id or checkpoint_id is required.",
    )


def ensure_coordinate_in_map_bounds(venue_map: dict, x: float, y: float) -> None:
    width = float(venue_map.get("width", 0))
    height = float(venue_map.get("height", 0))
    if x < 0 or y < 0 or x > width or y > height:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Coordinate is outside map bounds.",
                "x": x,
                "y": y,
                "map_width": width,
                "map_height": height,
            },
        )


def nearest_node_for_coordinate(
    venue_map: dict,
    x: float,
    y: float,
    selectable_only: bool = False,
) -> tuple[dict, float]:
    nodes = (
        venue_map.get("selectable_nodes", [])
        if selectable_only
        else venue_map.get("nodes", [])
    )
    if not nodes:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Map has no nodes available for coordinate snapping.",
                "selectable_only": selectable_only,
            },
        )

    nearest = min(
        nodes,
        key=lambda node: math.hypot(float(node["x"]) - x, float(node["y"]) - y),
    )
    distance = math.hypot(float(nearest["x"]) - x, float(nearest["y"]) - y)
    return nearest, round(distance, 1)


def project_coordinate_to_segment(
    x: float,
    y: float,
    start_node: dict,
    end_node: dict,
) -> tuple[float, float, float, float]:
    start_x = float(start_node["x"])
    start_y = float(start_node["y"])
    end_x = float(end_node["x"])
    end_y = float(end_node["y"])
    dx = end_x - start_x
    dy = end_y - start_y
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return start_x, start_y, 0.0, round(math.hypot(x - start_x, y - start_y), 1)

    progress = ((x - start_x) * dx + (y - start_y) * dy) / length_squared
    progress = max(0.0, min(1.0, progress))
    projected_x = start_x + progress * dx
    projected_y = start_y + progress * dy
    distance = math.hypot(x - projected_x, y - projected_y)
    return (
        round(projected_x, 1),
        round(projected_y, 1),
        round(progress, 3),
        round(distance, 1),
    )


def nearest_edge_for_coordinate(
    venue_map: dict,
    x: float,
    y: float,
) -> dict[str, Any] | None:
    nodes_by_id = {node["id"]: node for node in venue_map.get("nodes", [])}
    best_snap: dict[str, Any] | None = None
    for edge in venue_map.get("edges", []):
        start_node = nodes_by_id.get(edge.get("from"))
        end_node = nodes_by_id.get(edge.get("to"))
        if start_node is None or end_node is None:
            continue

        projected_x, projected_y, progress, distance = project_coordinate_to_segment(
            x,
            y,
            start_node,
            end_node,
        )
        if best_snap is None or distance < best_snap["edge_distance_px"]:
            best_snap = {
                "edge_id": edge["id"],
                "from_node_id": start_node["id"],
                "to_node_id": end_node["id"],
                "edge_distance_px": distance,
                "edge_progress_ratio": progress,
                "projected_x": projected_x,
                "projected_y": projected_y,
            }

    return best_snap


def snap_coordinate_to_node(
    venue_map: dict,
    x: float,
    y: float,
    max_distance_px: float,
    selectable_only: bool = False,
) -> CoordinateSnapResponse:
    ensure_coordinate_in_map_bounds(venue_map, x, y)
    nearest, distance = nearest_node_for_coordinate(
        venue_map,
        x=x,
        y=y,
        selectable_only=selectable_only,
    )
    edge_snap = None if selectable_only else nearest_edge_for_coordinate(venue_map, x, y)
    return CoordinateSnapResponse(
        map_id=venue_map["id"],
        input_x=x,
        input_y=y,
        snapped_node=point_for_node(nearest),
        distance_px=distance,
        within_threshold=distance <= max_distance_px,
        selectable=bool(nearest.get("selectable", False)),
        snapped_edge_id=edge_snap["edge_id"] if edge_snap else None,
        snapped_edge_from_node_id=edge_snap["from_node_id"] if edge_snap else None,
        snapped_edge_to_node_id=edge_snap["to_node_id"] if edge_snap else None,
        edge_distance_px=edge_snap["edge_distance_px"] if edge_snap else None,
        edge_progress_ratio=edge_snap["edge_progress_ratio"] if edge_snap else None,
        projected_x=edge_snap["projected_x"] if edge_snap else None,
        projected_y=edge_snap["projected_y"] if edge_snap else None,
    )


def require_navigation_session(session_id: str) -> dict:
    session = get_navigation_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Navigation session not found.",
                "session_id": session_id,
            },
        )
    return session


def navigation_session_response(
    session: dict,
    route_response: RouteResponse,
    update_limit: int = 20,
) -> NavigationSessionResponse:
    venue_map = load_map(session["map_id"])
    return NavigationSessionResponse(
        session_id=session["session_id"],
        map_id=session["map_id"],
        start_node=point_for_node(node_by_id(venue_map, session["start_node_id"])),
        current_node=point_for_node(node_by_id(venue_map, session["current_node_id"])),
        destination_node=point_for_node(node_by_id(venue_map, session["destination_id"])),
        status=session["status"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        route=route_response,
        recent_updates=[
            NavigationUpdateEvent(**event)
            for event in recent_navigation_updates(session["session_id"], update_limit)
        ],
    )


def route_distance_between_nodes(
    venue_map: dict[str, Any],
    start_id: str,
    destination_id: str,
) -> float | None:
    route_result = baseline_route_between_nodes(venue_map, start_id, destination_id)
    if route_result is None:
        return None
    return float(route_result["total_distance"])


def navigation_node_for_coordinate_snap(
    venue_map: dict[str, Any],
    snap: CoordinateSnapResponse,
    destination_id: str,
) -> dict[str, Any]:
    if snap.distance_px <= COORDINATE_NODE_LOCK_PX:
        return node_by_id(venue_map, snap.snapped_node.node_id)

    if (
        snap.snapped_edge_id is None
        or snap.snapped_edge_from_node_id is None
        or snap.snapped_edge_to_node_id is None
        or snap.edge_distance_px is None
        or snap.edge_distance_px > snap.distance_px
    ):
        return node_by_id(venue_map, snap.snapped_node.node_id)

    candidate_ids = [snap.snapped_edge_from_node_id, snap.snapped_edge_to_node_id]
    scored_candidates: list[tuple[float, str]] = []
    for candidate_id in candidate_ids:
        distance = route_distance_between_nodes(venue_map, candidate_id, destination_id)
        if distance is not None:
            scored_candidates.append((distance, candidate_id))

    if not scored_candidates:
        return node_by_id(venue_map, snap.snapped_node.node_id)

    _distance, best_node_id = min(scored_candidates)
    return node_by_id(venue_map, best_node_id)


def baseline_route_between_nodes(
    venue_map: dict[str, Any],
    start_id: str,
    destination_id: str,
) -> dict[str, Any] | None:
    return calculate_shortest_route(
        venue_map["nodes"],
        venue_map["edges"],
        {edge["id"]: 1.0 for edge in venue_map["edges"]},
        start_id,
        destination_id,
    )


def route_risk_level(score: float) -> Literal["low", "medium", "high", "critical"]:
    if score >= 80:
        return "critical"
    if score >= 55:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def selectable_route_pairs(venue_map: dict[str, Any], max_pairs: int) -> list[tuple[str, str]]:
    selectable_node_ids = [
        node["id"]
        for node in venue_map["nodes"]
        if node.get("selectable", False)
    ]
    pairs: list[tuple[str, str]] = []
    for start_id in selectable_node_ids:
        for destination_id in selectable_node_ids:
            if start_id == destination_id:
                continue
            pairs.append((start_id, destination_id))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def analyze_route_risk(
    venue_map: dict[str, Any],
    *,
    max_pairs: int,
    max_edges: int,
) -> RouteRiskAnalysisResponse:
    pairs = selectable_route_pairs(venue_map, max_pairs)
    baseline_routes: dict[tuple[str, str], dict[str, Any]] = {}
    for start_id, destination_id in pairs:
        route_result = calculate_shortest_route(
            venue_map["nodes"],
            venue_map["edges"],
            {edge["id"]: 1.0 for edge in venue_map["edges"]},
            start_id,
            destination_id,
        )
        if route_result is not None:
            baseline_routes[(start_id, destination_id)] = route_result

    node_map = {node["id"]: node for node in venue_map["nodes"]}
    risk_items: list[EdgeRiskItem] = []
    for edge in venue_map["edges"][:max_edges]:
        affected_pairs = [
            pair
            for pair, route_result in baseline_routes.items()
            if edge["id"] in route_result["edge_ids"]
        ]
        if not affected_pairs:
            risk_items.append(
                EdgeRiskItem(
                    edge_id=edge["id"],
                    from_node=node_map[edge["from"]]["name"],
                    to_node=node_map[edge["to"]]["name"],
                    zone=edge["zone"],
                    crowd_region=edge["crowdRegion"],
                    affected_pair_count=0,
                    disconnected_pair_count=0,
                    average_detour_ratio=0,
                    max_detour_ratio=0,
                    risk_score=0,
                    risk_level="low",
                    sample_pairs=[],
                )
            )
            continue

        detour_ratios: list[float] = []
        disconnected_count = 0
        sample_pairs: list[dict[str, Any]] = []
        for pair in affected_pairs:
            baseline = baseline_routes[pair]
            rerouted = calculate_shortest_route(
                venue_map["nodes"],
                venue_map["edges"],
                {candidate["id"]: 1.0 for candidate in venue_map["edges"]},
                pair[0],
                pair[1],
                blocked_edge_ids={edge["id"]},
            )
            if rerouted is None:
                disconnected_count += 1
                if len(sample_pairs) < 8:
                    sample_pairs.append(
                        {
                            "start_id": pair[0],
                            "destination_id": pair[1],
                            "impact": "disconnected",
                        }
                    )
                continue

            baseline_cost = max(float(baseline["weighted_cost"]), 0.1)
            ratio = max(0.0, float(rerouted["weighted_cost"]) / baseline_cost - 1)
            detour_ratios.append(ratio)
            if ratio > 0 and len(sample_pairs) < 8:
                sample_pairs.append(
                    {
                        "start_id": pair[0],
                        "destination_id": pair[1],
                        "impact": "detour",
                        "detour_ratio": round(ratio, 2),
                    }
                )

        average_detour_ratio = (
            sum(detour_ratios) / len(detour_ratios)
            if detour_ratios
            else 0.0
        )
        max_detour_ratio = max(detour_ratios) if detour_ratios else 0.0
        affected_ratio = len(affected_pairs) / max(len(baseline_routes), 1)
        disconnected_ratio = disconnected_count / max(len(affected_pairs), 1)
        risk_score = round(
            min(
                100.0,
                affected_ratio * 45
                + disconnected_ratio * 55
                + min(max_detour_ratio, 2.0) * 20
                + min(average_detour_ratio, 1.0) * 15,
            ),
            1,
        )
        risk_items.append(
            EdgeRiskItem(
                edge_id=edge["id"],
                from_node=node_map[edge["from"]]["name"],
                to_node=node_map[edge["to"]]["name"],
                zone=edge["zone"],
                crowd_region=edge["crowdRegion"],
                affected_pair_count=len(affected_pairs),
                disconnected_pair_count=disconnected_count,
                average_detour_ratio=round(average_detour_ratio, 2),
                max_detour_ratio=round(max_detour_ratio, 2),
                risk_score=risk_score,
                risk_level=route_risk_level(risk_score),
                sample_pairs=sample_pairs,
            )
        )

    risk_items.sort(key=lambda item: item.risk_score, reverse=True)
    return RouteRiskAnalysisResponse(
        map_id=venue_map["id"],
        selectable_node_count=len(venue_map.get("selectable_nodes", [])),
        analyzed_pair_count=len(baseline_routes),
        analyzed_edge_count=len(risk_items),
        critical_edge_count=sum(1 for item in risk_items if item.risk_level == "critical"),
        high_risk_edge_count=sum(1 for item in risk_items if item.risk_level == "high"),
        edges=risk_items,
    )


def pixel_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.hypot(
        float(left["x"]) - float(right["x"]),
        float(left["y"]) - float(right["y"]),
    )


def analyze_map_topology(
    venue_map: dict[str, Any],
    *,
    long_edge_threshold: float,
    checkpoint_gap_px: float,
) -> MapTopologyAnalysisResponse:
    node_map = {node["id"]: node for node in venue_map["nodes"]}
    degree = {node_id: 0 for node_id in node_map}
    directed_edge_groups: dict[tuple[str, str], list[str]] = {}
    undirected_edge_groups: dict[tuple[str, str], list[str]] = {}
    long_edge_ids: list[str] = []

    for edge in venue_map["edges"]:
        from_id = edge["from"]
        to_id = edge["to"]
        degree[from_id] += 1
        degree[to_id] += 1
        directed_edge_groups.setdefault((from_id, to_id), []).append(edge["id"])
        undirected_edge_groups.setdefault(tuple(sorted([from_id, to_id])), []).append(
            edge["id"]
        )
        if float(edge["distance"]) > long_edge_threshold:
            long_edge_ids.append(edge["id"])

    duplicate_edge_groups = [
        sorted(edge_ids)
        for edge_ids in undirected_edge_groups.values()
        if len(edge_ids) > 1
    ]
    dead_end_node_ids = sorted(
        node_id
        for node_id, count in degree.items()
        if count <= 1 and node_map[node_id].get("selectable", False)
    )
    low_degree_junction_ids = sorted(
        node_id
        for node_id, count in degree.items()
        if count <= 1 and node_map[node_id].get("type") == "junction"
    )

    checkpoint_nodes = [
        node_map[checkpoint["node_id"]]
        for checkpoint in venue_map.get("checkpoints", [])
        if checkpoint.get("node_id") in node_map
    ]
    checkpoint_gap_node_ids: list[str] = []
    if checkpoint_nodes:
        for node in venue_map.get("selectable_nodes", []):
            nearest_px = min(
                pixel_distance(node, checkpoint_node)
                for checkpoint_node in checkpoint_nodes
            )
            if nearest_px > checkpoint_gap_px:
                checkpoint_gap_node_ids.append(node["id"])
    elif venue_map.get("selectable_nodes"):
        checkpoint_gap_node_ids = [
            node["id"] for node in venue_map.get("selectable_nodes", [])
        ]

    issues: list[TopologyIssue] = []
    if dead_end_node_ids:
        issues.append(
            TopologyIssue(
                code="selectable_dead_ends",
                severity="warning",
                target_ids=dead_end_node_ids,
                message="Selectable destinations with only one connection are fragile during closures.",
            )
        )
    if low_degree_junction_ids:
        issues.append(
            TopologyIssue(
                code="low_degree_junctions",
                severity="warning",
                target_ids=low_degree_junction_ids,
                message="Junction nodes should usually connect multiple walkable segments.",
            )
        )
    if duplicate_edge_groups:
        issues.append(
            TopologyIssue(
                code="duplicate_edges",
                severity="error",
                target_ids=[edge_id for group in duplicate_edge_groups for edge_id in group],
                message="Multiple edges connect the same node pair. Remove duplicates or clarify directionality.",
                metadata={"groups": duplicate_edge_groups},
            )
        )
    if long_edge_ids:
        issues.append(
            TopologyIssue(
                code="long_edges",
                severity="warning",
                target_ids=sorted(long_edge_ids),
                message="Long edges may skip important turns or checkpoints; consider splitting them.",
                metadata={"threshold": long_edge_threshold},
            )
        )
    if checkpoint_gap_node_ids:
        issues.append(
            TopologyIssue(
                code="checkpoint_coverage_gaps",
                severity="info",
                target_ids=sorted(checkpoint_gap_node_ids),
                message="Some selectable nodes are far from the nearest checkpoint.",
                metadata={"threshold_px": checkpoint_gap_px},
            )
        )

    penalty = 0
    for issue in issues:
        if issue.severity == "error":
            penalty += 25
        elif issue.severity == "warning":
            penalty += 10
        else:
            penalty += 4
    score = max(0, 100 - penalty)
    if any(issue.severity == "error" for issue in issues):
        readiness = "blocked"
    elif score < 90:
        readiness = "needs_review"
    else:
        readiness = "ready"

    return MapTopologyAnalysisResponse(
        map_id=venue_map["id"],
        score=score,
        readiness=readiness,
        node_count=len(venue_map["nodes"]),
        edge_count=len(venue_map["edges"]),
        checkpoint_count=len(venue_map.get("checkpoints", [])),
        dead_end_node_ids=dead_end_node_ids,
        low_degree_junction_ids=low_degree_junction_ids,
        duplicate_edge_groups=duplicate_edge_groups,
        long_edge_ids=sorted(long_edge_ids),
        checkpoint_gap_node_ids=sorted(checkpoint_gap_node_ids),
        issues=issues,
    )


def suggestion_priority_value(priority: str) -> int:
    return {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }[priority]


def build_map_improvement_suggestions(
    venue_map: dict[str, Any],
) -> MapImprovementSuggestionsResponse:
    validation = validate_map_data(venue_map)
    postprocess = postprocess_map_data(
        venue_map,
        apply_connection_suggestions=False,
    )
    topology = analyze_map_topology(
        venue_map,
        long_edge_threshold=20,
        checkpoint_gap_px=420,
    )
    risk = analyze_route_risk(
        venue_map,
        max_pairs=120,
        max_edges=min(100, len(venue_map["edges"])),
    )

    suggestions: list[MapImprovementSuggestion] = []
    for item in validation.get("fix_suggestions", []):
        severity = item.get("severity", "warning")
        priority = "high" if severity == "error" else "medium"
        suggestions.append(
            MapImprovementSuggestion(
                code=item.get("code", "validation_issue"),
                priority=priority,
                source="validation",
                target_ids=[str(target_id) for target_id in item.get("target_ids", [])],
                message=item.get("message", "Map validation issue detected."),
                action="Fix the schema or graph references before using this map in the app.",
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"code", "severity", "target_ids", "message"}
                },
            )
        )

    for item in postprocess.get("suggestions", []):
        code = item.get("code", "postprocess_issue")
        priority: Literal["low", "medium", "high", "critical"] = "medium"
        action = "Review this generated-map issue before saving or publishing the map."
        target_ids: list[str] = []
        if code == "connect_disconnected_components":
            priority = "high"
            action = "Review candidate edges and apply only the walkable corridor connections."
            target_ids = [
                edge["id"]
                for edge in item.get("candidate_edges", [])
                if isinstance(edge, dict) and "id" in edge
            ]
        elif code == "weak_node_labels":
            action = "Rename generic or duplicated node labels so route instructions are understandable."
            target_ids = [
                node["node_id"]
                for node in item.get("nodes", [])
                if isinstance(node, dict) and "node_id" in node
            ]
        elif code == "close_nodes_detected":
            action = "Merge close nodes when they represent the same physical location."

        suggestions.append(
            MapImprovementSuggestion(
                code=code,
                priority=priority,
                source="postprocess",
                target_ids=target_ids,
                message=item.get("message", "Postprocess suggestion detected."),
                action=action,
                metadata=item,
            )
        )

    for issue in topology.issues:
        priority = "high" if issue.severity == "error" else "medium"
        if issue.severity == "info":
            priority = "low"
        action = "Review the topology issue and adjust nodes, edges, or checkpoints."
        if issue.code == "selectable_dead_ends":
            action = "Add an alternate connection for important selectable destinations or mark low-value dead ends as expected."
        elif issue.code == "long_edges":
            action = "Split long edges at meaningful turns or checkpoints."
        elif issue.code == "checkpoint_coverage_gaps":
            action = "Add QR checkpoints near far selectable destinations if precise navigation is needed."
        suggestions.append(
            MapImprovementSuggestion(
                code=issue.code,
                priority=priority,
                source="topology",
                target_ids=issue.target_ids,
                message=issue.message,
                action=action,
                metadata=issue.metadata,
            )
        )

    for edge in risk.edges:
        if edge.risk_level not in {"high", "critical"}:
            continue
        suggestions.append(
            MapImprovementSuggestion(
                code="high_route_risk_edge",
                priority="critical" if edge.risk_level == "critical" else "high",
                source="risk",
                target_ids=[edge.edge_id],
                message=(
                    f"{edge.from_node} - {edge.to_node} 구간 차단 시 "
                    f"{edge.affected_pair_count}개 경로 조합에 영향이 있습니다."
                ),
                action="Add alternate walkable edges or prepare an operational fallback for this corridor.",
                metadata={
                    "risk_score": edge.risk_score,
                    "risk_level": edge.risk_level,
                    "disconnected_pair_count": edge.disconnected_pair_count,
                    "average_detour_ratio": edge.average_detour_ratio,
                    "max_detour_ratio": edge.max_detour_ratio,
                    "sample_pairs": edge.sample_pairs,
                },
            )
        )

    suggestions.sort(
        key=lambda item: (
            suggestion_priority_value(item.priority),
            item.source,
            item.code,
        )
    )
    critical_count = sum(1 for item in suggestions if item.priority == "critical")
    high_count = sum(1 for item in suggestions if item.priority == "high")
    penalty = critical_count * 25 + high_count * 15
    penalty += sum(8 for item in suggestions if item.priority == "medium")
    penalty += sum(3 for item in suggestions if item.priority == "low")
    score = max(0, 100 - penalty)
    if critical_count:
        readiness = "blocked"
    elif high_count or score < 90:
        readiness = "needs_review"
    else:
        readiness = "ready"

    return MapImprovementSuggestionsResponse(
        map_id=venue_map["id"],
        readiness=readiness,
        score=score,
        suggestion_count=len(suggestions),
        critical_count=critical_count,
        high_count=high_count,
        suggestions=suggestions,
        validation_summary={
            "valid": validation["valid"],
            "error_count": len(validation["errors"]),
            "warning_count": len(validation["warnings"]),
            "quality_summary": validation["quality_summary"],
        },
        topology_summary={
            "score": topology.score,
            "readiness": topology.readiness,
            "issue_count": len(topology.issues),
        },
        risk_summary={
            "analyzed_edge_count": risk.analyzed_edge_count,
            "critical_edge_count": risk.critical_edge_count,
            "high_risk_edge_count": risk.high_risk_edge_count,
        },
    )


def calibrated_eta_for_session(session: dict[str, Any]) -> EtaCalibrationResponse:
    venue_map = load_map(session["map_id"])
    updates = navigation_updates_chronological(session["session_id"], limit=100)
    samples: list[EtaCalibrationSample] = []
    total_distance = 0.0
    total_elapsed_seconds = 0

    for previous, current in zip(updates, updates[1:]):
        if previous["node_id"] == current["node_id"]:
            continue
        elapsed_seconds = round(
            (
                parse_timestamp(current["timestamp"])
                - parse_timestamp(previous["timestamp"])
            ).total_seconds()
        )
        if elapsed_seconds < 2 or elapsed_seconds > 1800:
            continue

        distance = route_distance_between_nodes(
            venue_map,
            previous["node_id"],
            current["node_id"],
        )
        if distance is None or distance <= 0:
            continue

        observed_speed = distance / elapsed_seconds
        if observed_speed < 0.2 or observed_speed > MAX_WALKING_SPEED_MPS:
            continue

        total_distance += distance
        total_elapsed_seconds += elapsed_seconds
        samples.append(
            EtaCalibrationSample(
                from_node_id=previous["node_id"],
                to_node_id=current["node_id"],
                distance=round(distance, 1),
                elapsed_seconds=elapsed_seconds,
                observed_speed_mps=round(observed_speed, 2),
            )
        )

    observed_speed_mps = None
    if total_elapsed_seconds > 0:
        observed_speed_mps = round(total_distance / total_elapsed_seconds, 2)

    confidence = round(min(1.0, len(samples) / 5), 2)
    recommended_speed = WALKING_SPEED_MPS
    if observed_speed_mps is not None:
        recommended_speed = round(
            min(
                MAX_WALKING_SPEED_MPS,
                max(
                    0.4,
                    observed_speed_mps * confidence
                    + WALKING_SPEED_MPS * (1 - confidence),
                ),
            ),
            2,
        )

    return EtaCalibrationResponse(
        session_id=session["session_id"],
        map_id=session["map_id"],
        sample_count=len(samples),
        confidence=confidence,
        default_walking_speed_mps=WALKING_SPEED_MPS,
        observed_walking_speed_mps=observed_speed_mps,
        recommended_walking_speed_mps=recommended_speed,
        samples=samples,
    )


def navigation_guidance_for_session(session: dict[str, Any]) -> NavigationGuidanceResponse:
    venue_map = load_map(session["map_id"])
    expected_route = baseline_route_between_nodes(
        venue_map,
        session["start_node_id"],
        session["destination_id"],
    )
    expected_path = expected_route["path"] if expected_route else []
    expected_edge_ids = expected_route["edge_ids"] if expected_route else []
    off_route = (
        session["status"] != "arrived"
        and bool(expected_path)
        and session["current_node_id"] not in expected_path
    )
    route_response = route(
        RouteRequest(
            map_id=session["map_id"],
            start_id=session["current_node_id"],
            destination_id=session["destination_id"],
            log_route_intent=False,
        )
    )
    current_node = point_for_node(node_by_id(venue_map, session["current_node_id"]))
    destination_node = point_for_node(node_by_id(venue_map, session["destination_id"]))
    original_distance = float(expected_route["total_distance"]) if expected_route else None
    remaining_distance = route_response.total_distance
    if off_route:
        progress_ratio = 0.0
    elif original_distance is None or original_distance <= 0:
        progress_ratio = 1.0 if session["status"] == "arrived" else 0.0
    else:
        progress_ratio = round(
            min(1.0, max(0.0, 1 - remaining_distance / original_distance)),
            3,
        )

    next_segment = route_response.segments[0] if route_response.segments else None
    next_node = next_segment.to_node if next_segment else None
    if session["status"] == "arrived":
        instruction = f"{destination_node.name}에 도착했습니다."
    elif next_segment is None:
        instruction = "현재 위치에서 목적지까지의 다음 안내를 계산할 수 없습니다."
    else:
        instruction = next_segment.instruction

    return NavigationGuidanceResponse(
        session_id=session["session_id"],
        map_id=session["map_id"],
        status=session["status"],
        current_node=current_node,
        destination_node=destination_node,
        off_route=off_route,
        expected_path=expected_path,
        expected_edge_ids=expected_edge_ids,
        progress_ratio=progress_ratio,
        remaining_distance=remaining_distance,
        remaining_seconds=route_response.estimated_seconds,
        next_node=next_node,
        next_segment=next_segment,
        instruction=instruction,
        route=route_response,
    )


def estimated_crowd_inputs(map_id: str) -> CrowdInputs:
    now_hour = datetime.now().hour
    estimate = estimate_crowd_inputs(
        map_id=map_id,
        hour=now_hour,
        event_phase=DEFAULT_EVENT_PHASE,
    )
    return CrowdInputs(
        lobby_people=estimate["lobby_people"],
        booth_people=estimate["booth_people"],
        recent_inflow=estimate["recent_inflow"],
        hour=estimate["hour"],
        event_phase=estimate["event_phase"],
    )


@app.get("/maps")
def get_maps():
    return {"maps": list_maps()}


@app.get("/maps/schema", response_model=MapSchemaResponse)
def get_map_schema():
    return MapSchemaResponse(
        required_fields=["id", "name", "image", "width", "height", "nodes", "edges"],
        node_fields=["id", "name", "x", "y", "type", "selectable"],
        edge_fields=[
            "id",
            "from",
            "to",
            "distance",
            "widthM",
            "zone",
            "crowdRegion",
            "bidirectional",
        ],
        checkpoint_fields=["id", "name", "node_id", "region"],
        node_types=["entrance", "exit", "junction", "facility", "booth"],
        zones=["lobby", "booth", "gate", "facility"],
        crowd_regions=["west", "central", "north", "booth"],
        checkpoint_regions=["lobby", "booth"],
        coordinate_system="Image pixel coordinates using the map width and height.",
        notes=[
            "Every edge from/to value must reference an existing node id.",
            "Every checkpoint node_id must reference an existing node id.",
            "At least one selectable node is recommended for app start/destination choices.",
            "Run POST /maps/validate-data before saving LLM-generated map data.",
        ],
    )


@app.post("/maps/validate-data")
def validate_map_payload(venue_map: dict[str, Any] = Body(...)):
    return {
        "map_id": venue_map.get("id"),
        **validate_map_data(venue_map),
    }


@app.post("/maps", response_model=MapSaveResponse)
def save_map_payload(request: MapSaveRequest):
    return MapSaveResponse(**save_map_data(request.venue_map, request.overwrite))


@app.get("/maps/{map_id}/blocked-edges", response_model=BlockedEdgesResponse)
def get_blocked_edges(map_id: str):
    load_map(map_id)
    blocked_edges = TEMPORARY_BLOCKED_EDGES.get(map_id, {})
    reasons = sorted({value for value in blocked_edges.values() if value})
    return BlockedEdgesResponse(
        map_id=map_id,
        blocked_edge_ids=sorted(blocked_edges),
        reason=", ".join(reasons) if reasons else None,
    )


@app.post("/maps/{map_id}/blocked-edges", response_model=BlockedEdgesResponse)
def update_blocked_edges(map_id: str, request: BlockedEdgesRequest):
    venue_map = load_map(map_id)
    unique_edge_ids = sorted(set(request.edge_ids))
    validate_edge_ids(venue_map, unique_edge_ids)
    TEMPORARY_BLOCKED_EDGES[map_id] = {
        edge_id: request.reason
        for edge_id in unique_edge_ids
    }
    return BlockedEdgesResponse(
        map_id=map_id,
        blocked_edge_ids=unique_edge_ids,
        reason=request.reason,
    )


@app.get(
    "/maps/{map_id}/route-risk-analysis",
    response_model=RouteRiskAnalysisResponse,
)
def get_route_risk_analysis(
    map_id: str,
    max_pairs: int = 300,
    max_edges: int = 100,
):
    venue_map = load_map(map_id)
    if max_pairs < 1 or max_pairs > 2000:
        raise HTTPException(status_code=400, detail="max_pairs must be between 1 and 2000")
    if max_edges < 1 or max_edges > 500:
        raise HTTPException(status_code=400, detail="max_edges must be between 1 and 500")
    return analyze_route_risk(
        venue_map,
        max_pairs=max_pairs,
        max_edges=max_edges,
    )


@app.get(
    "/maps/{map_id}/improvement-suggestions",
    response_model=MapImprovementSuggestionsResponse,
)
def get_map_improvement_suggestions(map_id: str):
    venue_map = load_map(map_id)
    return build_map_improvement_suggestions(venue_map)


@app.get(
    "/maps/{map_id}/topology-analysis",
    response_model=MapTopologyAnalysisResponse,
)
def get_map_topology_analysis(
    map_id: str,
    long_edge_threshold: float = 20,
    checkpoint_gap_px: float = 420,
):
    venue_map = load_map(map_id)
    if long_edge_threshold <= 0 or long_edge_threshold > 1000:
        raise HTTPException(
            status_code=400,
            detail="long_edge_threshold must be greater than 0 and at most 1000",
        )
    if checkpoint_gap_px < 0 or checkpoint_gap_px > 5000:
        raise HTTPException(
            status_code=400,
            detail="checkpoint_gap_px must be between 0 and 5000",
        )
    return analyze_map_topology(
        venue_map,
        long_edge_threshold=long_edge_threshold,
        checkpoint_gap_px=checkpoint_gap_px,
    )


@app.post("/maps/{map_id}/snap-coordinate", response_model=CoordinateSnapResponse)
def snap_coordinate(map_id: str, request: CoordinateSnapRequest):
    return snap_coordinate_to_node(
        load_map(map_id),
        x=request.x,
        y=request.y,
        max_distance_px=request.max_distance_px,
        selectable_only=request.selectable_only,
    )


@app.get("/maps/{map_id}")
def get_map(map_id: str):
    return load_map(map_id)


@app.delete("/maps/{map_id}", response_model=MapDeleteResponse)
def delete_map(map_id: str):
    return MapDeleteResponse(**delete_map_data(map_id))


@app.get("/maps/{map_id}/checkpoints")
def get_map_checkpoints(map_id: str):
    venue_map = load_map(map_id)
    return {
        "map_id": venue_map["id"],
        "checkpoints": venue_map.get("checkpoints", []),
    }


@app.get("/maps/{map_id}/validate")
def get_map_validation(map_id: str):
    return validate_map(map_id)


@app.post("/map-generation/jobs", response_model=MapGenerationJobResponse)
def create_map_generation_job(request: MapGenerationJobCreateRequest):
    return MapGenerationJobResponse(
        **create_generation_job(
            source_image_base64=request.source_image_base64,
            filename=request.filename,
            mime_type=request.mime_type,
            target_map_id=request.target_map_id,
            notes=request.notes,
        )
    )


@app.get("/map-generation/jobs", response_model=MapGenerationJobsResponse)
def get_map_generation_jobs(limit: int = 20):
    return MapGenerationJobsResponse(
        jobs=[
            MapGenerationJobResponse(**job)
            for job in list_generation_jobs(limit)
        ]
    )


@app.get("/map-generation/jobs/{job_id}", response_model=MapGenerationJobResponse)
def get_map_generation_job(job_id: str):
    return MapGenerationJobResponse(**get_generation_job(job_id))


@app.post(
    "/map-generation/jobs/{job_id}/generate-draft",
    response_model=MapGenerationDraftGenerateResponse,
)
def generate_map_generation_draft(
    job_id: str,
    request: MapGenerationGenerateDraftRequest,
):
    job = get_generation_job(job_id)
    draft_map = request_openai_map_draft(
        job=job,
        model=request.model or configured_openai_map_model(),
        image_detail=request.image_detail,
        extra_instructions=request.extra_instructions,
    )
    updated_job = attach_draft_map(job_id, draft_map)
    return MapGenerationDraftGenerateResponse(
        job=MapGenerationJobResponse(**updated_job),
        draft_map=draft_map,
    )


@app.get("/map-generation/jobs/{job_id}/draft-map", response_model=MapGenerationDraftResponse)
def get_map_generation_draft(job_id: str):
    job = get_generation_job(job_id)
    draft_map = get_draft_map(job_id)
    return MapGenerationDraftResponse(
        job=MapGenerationJobResponse(**job),
        draft_map=draft_map,
        validation={"map_id": draft_map.get("id"), **validate_map_data(draft_map)},
    )


@app.post("/map-generation/jobs/{job_id}/draft-map", response_model=MapGenerationJobResponse)
def attach_map_generation_draft(job_id: str, request: MapGenerationDraftRequest):
    return MapGenerationJobResponse(
        **attach_draft_map(job_id, request.venue_map)
    )


@app.post(
    "/map-generation/jobs/{job_id}/postprocess-draft",
    response_model=MapGenerationPostprocessResponse,
)
def postprocess_map_generation_draft(
    job_id: str,
    request: MapGenerationPostprocessRequest,
):
    draft_map = get_draft_map(job_id)
    result = postprocess_map_data(
        draft_map,
        recalculate_edge_distance=request.recalculate_edge_distance,
        clamp_coordinates=request.clamp_coordinates,
        close_node_threshold_px=request.close_node_threshold_px,
        suggest_connection_edges=request.suggest_connection_edges,
        apply_connection_suggestions=request.apply_connection_suggestions,
        max_connection_suggestions=request.max_connection_suggestions,
        max_connection_distance_px=request.max_connection_distance_px,
    )
    if request.apply:
        updated_job = attach_draft_map(job_id, result["processed_map"])
    else:
        updated_job = get_generation_job(job_id)

    return MapGenerationPostprocessResponse(
        job=MapGenerationJobResponse(**updated_job),
        processed_map=result["processed_map"],
        changes=result["changes"],
        suggestions=result["suggestions"],
        validation=result["validation"],
        applied=request.apply,
    )


@app.post(
    "/map-generation/jobs/{job_id}/route-preview",
    response_model=RouteResponse,
)
def preview_map_generation_route(
    job_id: str,
    request: MapGenerationRoutePreviewRequest,
):
    draft_map = get_draft_map(job_id)
    validation = validate_map_data(draft_map)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Draft map is invalid.",
                "validation": {"map_id": draft_map.get("id"), **validation},
            },
        )

    current_node, _checkpoint = resolve_position_node(
        draft_map,
        request.start_id or request.current_node_id,
        request.checkpoint_id,
    )
    return build_route_response(
        venue_map=draft_map,
        request=RouteRequest(
            map_id=draft_map["id"],
            start_id=current_node["id"],
            destination_id=request.destination_id,
            crowd_inputs=request.crowd_inputs,
            walking_speed_mps=request.walking_speed_mps,
            use_congestion=request.use_congestion,
            log_route_intent=False,
            preference=request.preference,
            algorithm=request.algorithm,
            blocked_edge_ids=request.blocked_edge_ids,
        ),
    )


@app.post(
    "/map-generation/jobs/{job_id}/save-map",
    response_model=MapGenerationSaveResponse,
)
def save_map_generation_draft(job_id: str, request: MapGenerationSaveRequest):
    result = save_generation_job_map(job_id, overwrite=request.overwrite)
    return MapGenerationSaveResponse(
        job=MapGenerationJobResponse(**result["job"]),
        save_result=MapSaveResponse(**result["save_result"]),
    )


@app.get("/crowd/{map_id}")
def get_crowd_estimate(map_id: str):
    return estimate_crowd_inputs(
        map_id=map_id,
        hour=datetime.now().hour,
        event_phase=DEFAULT_EVENT_PHASE,
    )


@app.get("/crowd/{map_id}/forecast")
def get_crowd_forecast(
    map_id: str,
    horizon_minutes: int = 30,
    step_minutes: int = 10,
):
    load_map(map_id)
    if horizon_minutes < 5 or horizon_minutes > 120:
        raise HTTPException(
            status_code=400,
            detail="horizon_minutes must be between 5 and 120",
        )
    if step_minutes < 5 or step_minutes > 60:
        raise HTTPException(
            status_code=400,
            detail="step_minutes must be between 5 and 60",
        )

    baseline = estimate_crowd_inputs(
        map_id=map_id,
        hour=datetime.now().hour,
        event_phase=DEFAULT_EVENT_PHASE,
    )
    signals = baseline["signals"]
    lobby_pressure = (
        signals["qr_scans"]["lobby"] * 0.25
        + signals["route_intents"]["lobby"] * 0.45
    )
    booth_pressure = (
        signals["qr_scans"]["booth"] * 0.3
        + signals["route_intents"]["booth"] * 0.65
    )

    points = []
    for minutes_ahead in range(0, horizon_minutes + 1, step_minutes):
        pressure_decay = max(0.25, 1 - minutes_ahead / max(horizon_minutes, 1))
        phase_boost = 1 + DEFAULT_EVENT_PHASE * 0.08
        lobby_people = min(
            MAX_PEOPLE_COUNT,
            round(baseline["lobby_people"] + lobby_pressure * pressure_decay),
        )
        booth_people = min(
            MAX_PEOPLE_COUNT,
            round(
                baseline["booth_people"]
                + booth_pressure * pressure_decay * phase_boost
            ),
        )
        recent_inflow = min(
            MAX_RECENT_INFLOW,
            round(baseline["recent_inflow"] * pressure_decay),
        )
        points.append(
            {
                "minutes_ahead": minutes_ahead,
                "lobby_people": lobby_people,
                "booth_people": booth_people,
                "recent_inflow": recent_inflow,
                "level": level_from_multiplier(
                    1 + max(lobby_people, booth_people) / 450
                ),
            }
        )

    return {
        "map_id": map_id,
        "horizon_minutes": horizon_minutes,
        "step_minutes": step_minutes,
        "method": "manual_qr_route_intent_decay",
        "points": points,
    }


@app.get("/crowd/{map_id}/bottlenecks")
def get_crowd_bottlenecks(map_id: str, limit: int = 5):
    venue_map = load_map(map_id)
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 20")

    crowd_inputs = estimated_crowd_inputs(map_id)
    _prediction_response, predictions = route_predictions(
        venue_map["edges"],
        crowd_inputs,
        use_congestion=True,
    )
    edge_map = {edge["id"]: edge for edge in venue_map["edges"]}
    node_map = {node["id"]: node for node in venue_map["nodes"]}
    bottlenecks = []
    for prediction in sorted(
        predictions,
        key=lambda item: item.multiplier,
        reverse=True,
    ):
        if prediction.level not in {"busy", "very_busy"}:
            continue
        edge = edge_map[prediction.edge_id]
        bottlenecks.append(
            {
                "edge_id": edge["id"],
                "from_node": node_map[edge["from"]]["name"],
                "to_node": node_map[edge["to"]]["name"],
                "zone": edge["zone"],
                "crowd_region": edge["crowdRegion"],
                "width_m": edge["widthM"],
                "multiplier": prediction.multiplier,
                "level": prediction.level,
            }
        )
        if len(bottlenecks) >= limit:
            break

    return {
        "map_id": map_id,
        "crowd_inputs": crowd_inputs,
        "bottlenecks": bottlenecks,
    }


@app.get("/crowd/{map_id}/demand-heatmap", response_model=DemandHeatmapResponse)
def get_demand_heatmap(map_id: str):
    venue_map = load_map(map_id)
    return build_demand_heatmap(venue_map)


@app.get("/telemetry/{map_id}")
def get_telemetry(map_id: str, limit: int = 20):
    load_map(map_id)
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100",
        )
    return telemetry_snapshot(map_id, limit)


@app.post("/telemetry/qr-scan")
def create_qr_scan(request: QrScanRequest):
    load_map(request.map_id)
    record_qr_scan(
        map_id=request.map_id,
        checkpoint_id=request.checkpoint_id,
        region=request.region,
        count=request.count,
    )
    return get_crowd_estimate(request.map_id)


@app.post("/telemetry/manual-crowd")
def update_manual_crowd(request: ManualCrowdRequest):
    load_map(request.map_id)
    set_manual_estimate(
        map_id=request.map_id,
        lobby_people=request.lobby_people,
        booth_people=request.booth_people,
    )
    return get_crowd_estimate(request.map_id)


@app.post("/navigation/update-position", response_model=PositionUpdateResponse)
def update_position(request: PositionUpdateRequest):
    venue_map = load_map(request.map_id)
    current_node, checkpoint = resolve_position_node(
        venue_map,
        request.current_node_id,
        request.checkpoint_id,
    )
    destination_node = node_by_id(venue_map, request.destination_id)

    telemetry_region = request.region or (checkpoint["region"] if checkpoint else None)
    if request.checkpoint_id and telemetry_region:
        record_qr_scan(
            map_id=request.map_id,
            checkpoint_id=request.checkpoint_id,
            region=telemetry_region,
            count=1,
        )

    route_response = route(
        RouteRequest(
            map_id=request.map_id,
            start_id=current_node["id"],
            destination_id=request.destination_id,
            crowd_inputs=request.crowd_inputs,
            walking_speed_mps=request.walking_speed_mps,
            use_congestion=request.use_congestion,
            log_route_intent=request.log_route_intent,
            preference=request.preference,
            algorithm=request.algorithm,
            blocked_edge_ids=request.blocked_edge_ids,
        )
    )

    return PositionUpdateResponse(
        map_id=venue_map["id"],
        current_node=point_for_node(current_node),
        destination_node=point_for_node(destination_node),
        location_source="checkpoint" if request.checkpoint_id else "node",
        route=route_response,
    )


@app.post("/navigation/sessions", response_model=NavigationSessionResponse)
def start_navigation_session(request: NavigationSessionStartRequest):
    venue_map = load_map(request.map_id)
    validate_route_nodes(venue_map, request.start_id, request.destination_id)
    session = create_navigation_session(
        map_id=request.map_id,
        start_node_id=request.start_id,
        destination_id=request.destination_id,
    )
    route_response = route(
        RouteRequest(
            map_id=request.map_id,
            start_id=request.start_id,
            destination_id=request.destination_id,
            crowd_inputs=request.crowd_inputs,
            walking_speed_mps=request.walking_speed_mps,
            use_congestion=request.use_congestion,
            log_route_intent=request.log_route_intent,
            preference=request.preference,
            algorithm=request.algorithm,
            blocked_edge_ids=request.blocked_edge_ids,
        )
    )
    return navigation_session_response(session, route_response)


@app.get("/navigation/sessions/{session_id}", response_model=NavigationSessionResponse)
def get_navigation_session_status(session_id: str):
    session = require_navigation_session(session_id)
    route_response = route(
        RouteRequest(
            map_id=session["map_id"],
            start_id=session["current_node_id"],
            destination_id=session["destination_id"],
            log_route_intent=False,
        )
    )
    return navigation_session_response(session, route_response)


@app.get(
    "/navigation/sessions/{session_id}/guidance",
    response_model=NavigationGuidanceResponse,
)
def get_navigation_guidance(session_id: str):
    session = require_navigation_session(session_id)
    return navigation_guidance_for_session(session)


@app.get(
    "/navigation/sessions/{session_id}/eta-calibration",
    response_model=EtaCalibrationResponse,
)
def get_navigation_eta_calibration(session_id: str):
    session = require_navigation_session(session_id)
    return calibrated_eta_for_session(session)


@app.post(
    "/navigation/sessions/{session_id}/position",
    response_model=NavigationSessionResponse,
)
def update_navigation_session_position_api(
    session_id: str,
    request: NavigationSessionUpdateRequest,
):
    session = require_navigation_session(session_id)
    venue_map = load_map(session["map_id"])
    current_node, checkpoint = resolve_position_node(
        venue_map,
        request.current_node_id,
        request.checkpoint_id,
    )

    telemetry_region = request.region or (checkpoint["region"] if checkpoint else None)
    if request.checkpoint_id and telemetry_region:
        record_qr_scan(
            map_id=session["map_id"],
            checkpoint_id=request.checkpoint_id,
            region=telemetry_region,
            count=1,
        )

    updated_session = update_navigation_session_position(
        session_id=session_id,
        node_id=current_node["id"],
        source="checkpoint" if request.checkpoint_id else "node",
        checkpoint_id=request.checkpoint_id,
    )
    if updated_session is None:
        raise HTTPException(status_code=404, detail="Navigation session not found.")

    route_response = route(
        RouteRequest(
            map_id=updated_session["map_id"],
            start_id=updated_session["current_node_id"],
            destination_id=updated_session["destination_id"],
            crowd_inputs=request.crowd_inputs,
            walking_speed_mps=request.walking_speed_mps,
            use_congestion=request.use_congestion,
            log_route_intent=request.log_route_intent,
            preference=request.preference,
            algorithm=request.algorithm,
            blocked_edge_ids=request.blocked_edge_ids,
        )
    )
    return navigation_session_response(updated_session, route_response)


@app.post(
    "/navigation/sessions/{session_id}/position-coordinate",
    response_model=NavigationSessionResponse,
)
def update_navigation_session_coordinate_api(
    session_id: str,
    request: NavigationSessionCoordinateUpdateRequest,
):
    session = require_navigation_session(session_id)
    venue_map = load_map(session["map_id"])
    snap = snap_coordinate_to_node(
        venue_map,
        x=request.x,
        y=request.y,
        max_distance_px=request.max_distance_px,
        selectable_only=False,
    )
    if not snap.within_threshold:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Coordinate is too far from the nearest route node.",
                "snap": snap.model_dump(),
            },
        )
    current_node = navigation_node_for_coordinate_snap(
        venue_map,
        snap,
        session["destination_id"],
    )

    updated_session = update_navigation_session_position(
        session_id=session_id,
        node_id=current_node["id"],
        source="coordinate",
        checkpoint_id=None,
    )
    if updated_session is None:
        raise HTTPException(status_code=404, detail="Navigation session not found.")

    route_response = route(
        RouteRequest(
            map_id=updated_session["map_id"],
            start_id=updated_session["current_node_id"],
            destination_id=updated_session["destination_id"],
            crowd_inputs=request.crowd_inputs,
            walking_speed_mps=request.walking_speed_mps,
            use_congestion=request.use_congestion,
            log_route_intent=request.log_route_intent,
            preference=request.preference,
            algorithm=request.algorithm,
            blocked_edge_ids=request.blocked_edge_ids,
        )
    )
    return navigation_session_response(updated_session, route_response)


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest):
    venue_map = load_map(request.map_id)
    if request.log_route_intent:
        record_route_intent(
            map_id=request.map_id,
            destination_id=request.destination_id,
            region=region_for_destination(request.destination_id),
        )

    return build_route_response(venue_map, request)


@app.post("/route-alternatives", response_model=RouteAlternativesResponse)
def route_alternatives(request: RouteAlternativesRequest):
    venue_map = load_map(request.map_id)
    edges = venue_map["edges"]
    validate_route_nodes(venue_map, request.start_id, request.destination_id)
    blocked_edge_ids = active_blocked_edge_ids(venue_map, request.blocked_edge_ids)
    crowd_inputs = request.crowd_inputs or estimated_crowd_inputs(request.map_id)

    if request.log_route_intent:
        record_route_intent(
            map_id=request.map_id,
            destination_id=request.destination_id,
            region=region_for_destination(request.destination_id),
        )

    prediction_response, applied_predictions = route_predictions(
        edges,
        crowd_inputs,
        request.use_congestion,
    )
    multipliers = route_cost_multipliers(
        edges,
        applied_predictions,
        request.preference,
    )
    routes = calculate_route_alternatives(
        venue_map["nodes"],
        edges,
        multipliers,
        request.start_id,
        request.destination_id,
        max_routes=request.max_routes,
        overlap_penalty=request.overlap_penalty,
        blocked_edge_ids=blocked_edge_ids,
        turn_penalty=preference_turn_penalty(request.preference),
        algorithm=request.algorithm,
    )
    if not routes:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Route not found between the selected nodes.",
                "start_id": request.start_id,
                "destination_id": request.destination_id,
                "map_id": request.map_id,
            },
        )

    best_edge_ids = set(routes[0]["edge_ids"])
    alternatives = []
    for index, route_result in enumerate(routes):
        segments = route_segments_for_edges(
            venue_map,
            route_result["path"],
            route_result["edge_ids"],
            applied_predictions,
            request.walking_speed_mps,
        )
        alternatives.append(
            RouteAlternative(
                rank=index + 1,
                algorithm=route_result["algorithm"],
                expanded_state_count=route_result["expanded_state_count"],
                overlap_ratio=route_result["overlap_ratio"],
                detour_ratio=route_result["detour_ratio"],
                quality_score=alternative_quality_score(route_result),
                path=route_result["path"],
                edge_ids=route_result["edge_ids"],
                route_points=route_points_for_path(venue_map, route_result["path"]),
                segments=segments,
                total_distance=route_result["total_distance"],
                weighted_cost=route_result["weighted_cost"],
                estimated_seconds=estimated_seconds(
                    route_result["total_distance"],
                    request.walking_speed_mps,
                ),
                instructions=[segment.instruction for segment in segments],
                overlap_with_best_edge_count=len(
                    set(route_result["edge_ids"]) & best_edge_ids
                ),
            )
        )

    return RouteAlternativesResponse(
        map_id=venue_map["id"],
        start_id=request.start_id,
        destination_id=request.destination_id,
        use_congestion=request.use_congestion,
        walking_speed_mps=request.walking_speed_mps,
        preference=request.preference,
        algorithm=request.algorithm,
        blocked_edge_ids=sorted(blocked_edge_ids),
        crowd_inputs=crowd_inputs,
        alternatives=alternatives,
        model_type=prediction_response.model_type,
        validation_mae=prediction_response.validation_mae,
        predictions=applied_predictions,
    )


def alternative_quality_score(route_result: dict[str, Any]) -> int:
    overlap_ratio = float(route_result.get("overlap_ratio", 1.0))
    detour_ratio = float(route_result.get("detour_ratio", 1.0))
    overlap_penalty = overlap_ratio * 35
    detour_penalty = max(0.0, detour_ratio - 1.0) * 45
    return max(0, min(100, round(100 - overlap_penalty - detour_penalty)))


def recommendation_reasons(preference: RoutePreference) -> list[str]:
    if preference == "less_crowded":
        return [
            "혼잡 예측 multiplier를 더 강하게 반영했습니다.",
            "좁은 통로에는 추가 penalty를 적용했습니다.",
        ]
    if preference == "accessible":
        return [
            "폭이 좁은 edge를 피하도록 penalty를 적용했습니다.",
            "이동 편의성을 위해 넓은 통로를 우선했습니다.",
        ]
    if preference == "fewest_turns":
        return [
            "급격한 방향 전환이 많은 경로에 turn penalty를 적용했습니다.",
            "안내 문장이 단순해지는 경로를 우선했습니다.",
        ]
    return [
        "거리와 현재 적용된 가중치 기준으로 가장 비용이 낮은 경로입니다.",
    ]


def route_average_multiplier(alternative: RouteAlternative) -> float:
    if not alternative.segments:
        return 1.0
    weighted_sum = sum(
        segment.multiplier * segment.distance for segment in alternative.segments
    )
    distance_sum = sum(segment.distance for segment in alternative.segments)
    return round(weighted_sum / max(distance_sum, 0.001), 3)


def route_max_multiplier(alternative: RouteAlternative) -> float:
    if not alternative.segments:
        return 1.0
    return round(max(segment.multiplier for segment in alternative.segments), 3)


def route_turn_count(alternative: RouteAlternative) -> int:
    turn_maneuvers = {"slight_left", "left", "slight_right", "right", "u_turn"}
    return sum(1 for segment in alternative.segments if segment.maneuver in turn_maneuvers)


def recommendation_metrics(
    alternative: RouteAlternative,
    best: RouteAlternative,
) -> dict[str, Any]:
    return {
        "cost_ratio": round(
            alternative.weighted_cost / max(best.weighted_cost, 0.001),
            3,
        ),
        "distance_ratio": round(
            alternative.total_distance / max(best.total_distance, 0.001),
            3,
        ),
        "average_multiplier": route_average_multiplier(alternative),
        "max_multiplier": route_max_multiplier(alternative),
        "turn_count": route_turn_count(alternative),
        "overlap_ratio": alternative.overlap_ratio,
        "detour_ratio": alternative.detour_ratio,
        "quality_score": alternative.quality_score,
    }


def recommendation_score_for_metrics(
    preference: RoutePreference,
    metrics: dict[str, Any],
) -> int:
    cost_penalty = max(0.0, metrics["cost_ratio"] - 1.0)
    distance_penalty = max(0.0, metrics["distance_ratio"] - 1.0)
    diversity_bonus = max(0.0, 1.0 - metrics["overlap_ratio"]) * 8

    if preference == "less_crowded":
        score = (
            100
            - max(0.0, metrics["average_multiplier"] - 1.0) * 30
            - max(0.0, metrics["max_multiplier"] - 1.0) * 12
            - cost_penalty * 35
            + diversity_bonus
        )
    elif preference == "accessible":
        score = 100 - cost_penalty * 45 - distance_penalty * 25 + metrics["quality_score"] * 0.2
    elif preference == "fewest_turns":
        score = 100 - metrics["turn_count"] * 10 - cost_penalty * 35 + diversity_bonus
    else:
        score = 100 - cost_penalty * 70 - distance_penalty * 25

    return max(0, min(100, round(score)))


def select_recommended_alternative(
    preference: RoutePreference,
    alternatives: list[RouteAlternative],
) -> tuple[RouteAlternative, int, dict[str, Any]]:
    best = alternatives[0]
    scored = []
    for alternative in alternatives:
        metrics = recommendation_metrics(alternative, best)
        score = recommendation_score_for_metrics(preference, metrics)
        scored.append((score, -alternative.weighted_cost, alternative.rank, alternative, metrics))

    score, _negative_cost, _rank, selected, metrics = max(scored)
    return selected, score, metrics


def route_tradeoffs(
    alternatives: list[RouteAlternative],
    selected: RouteAlternative,
) -> dict[str, Any]:
    selected_edge_ids = set(selected.edge_ids)
    return {
        "selected_distance": selected.total_distance,
        "selected_weighted_cost": selected.weighted_cost,
        "selected_quality_score": selected.quality_score,
        "candidate_count": len(alternatives),
        "candidates": [
            {
                "rank": alternative.rank,
                "distance_delta": round(
                    alternative.total_distance - selected.total_distance,
                    1,
                ),
                "weighted_cost_delta": round(
                    alternative.weighted_cost - selected.weighted_cost,
                    1,
                ),
                "overlap_with_selected_edge_count": (
                    len(set(alternative.edge_ids) & selected_edge_ids)
                ),
                "overlap_ratio": alternative.overlap_ratio,
                "detour_ratio": alternative.detour_ratio,
                "quality_score": alternative.quality_score,
            }
            for alternative in alternatives
        ],
    }


def heat_level_from_score(score: float) -> Literal["low", "medium", "high", "critical"]:
    if score >= 85:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def demand_score(
    people_estimate: int,
    qr_scans: int,
    route_intents: int,
    multiplier: float = 1.0,
    width_m: float = 3.0,
) -> float:
    width_pressure = max(0.0, 3.5 - width_m) * 8
    score = (
        min(55.0, people_estimate / 12)
        + min(20.0, qr_scans * 2.0)
        + min(20.0, route_intents * 3.0)
        + max(0.0, multiplier - 1) * 18
        + width_pressure
    )
    return round(min(100.0, score), 1)


def build_demand_heatmap(venue_map: dict[str, Any]) -> DemandHeatmapResponse:
    crowd_inputs = estimated_crowd_inputs(venue_map["id"])
    signals = estimate_crowd_inputs(
        map_id=venue_map["id"],
        hour=crowd_inputs.hour,
        event_phase=crowd_inputs.event_phase,
    )["signals"]
    region_people = {
        "west": crowd_inputs.lobby_people,
        "central": crowd_inputs.lobby_people,
        "north": crowd_inputs.lobby_people,
        "booth": crowd_inputs.booth_people,
    }
    region_signals = {
        "west": {
            "qr_scans": signals["qr_scans"]["lobby"],
            "route_intents": signals["route_intents"]["lobby"],
        },
        "central": {
            "qr_scans": signals["qr_scans"]["lobby"],
            "route_intents": signals["route_intents"]["lobby"],
        },
        "north": {
            "qr_scans": signals["qr_scans"]["lobby"],
            "route_intents": signals["route_intents"]["lobby"],
        },
        "booth": {
            "qr_scans": signals["qr_scans"]["booth"],
            "route_intents": signals["route_intents"]["booth"],
        },
    }

    _prediction_response, predictions = route_predictions(
        venue_map["edges"],
        crowd_inputs,
        use_congestion=True,
    )
    prediction_map = {prediction.edge_id: prediction for prediction in predictions}
    node_map = {node["id"]: node for node in venue_map["nodes"]}

    regions = [
        DemandHeatRegion(
            region=region,
            people_estimate=region_people[region],
            qr_scans=region_signals[region]["qr_scans"],
            route_intents=region_signals[region]["route_intents"],
            demand_score=demand_score(
                region_people[region],
                region_signals[region]["qr_scans"],
                region_signals[region]["route_intents"],
            ),
        )
        for region in ["west", "central", "north", "booth"]
    ]
    edges: list[DemandHeatEdge] = []
    for edge in venue_map["edges"]:
        region = edge["crowdRegion"]
        prediction = prediction_map[edge["id"]]
        score = demand_score(
            region_people[region],
            region_signals[region]["qr_scans"],
            region_signals[region]["route_intents"],
            multiplier=prediction.multiplier,
            width_m=float(edge["widthM"]),
        )
        edges.append(
            DemandHeatEdge(
                edge_id=edge["id"],
                from_node=node_map[edge["from"]]["name"],
                to_node=node_map[edge["to"]]["name"],
                zone=edge["zone"],
                crowd_region=region,
                multiplier=prediction.multiplier,
                level=prediction.level,
                demand_score=score,
                heat_level=heat_level_from_score(score),
            )
        )

    edges.sort(key=lambda item: item.demand_score, reverse=True)
    regions.sort(key=lambda item: item.demand_score, reverse=True)
    return DemandHeatmapResponse(
        map_id=venue_map["id"],
        method="manual_qr_route_intent_prediction_width",
        crowd_inputs=crowd_inputs,
        regions=regions,
        edges=edges,
    )


@app.post("/route-recommendation", response_model=RouteRecommendationResponse)
def route_recommendation(request: RouteAlternativesRequest):
    response = route_alternatives(request)
    selected, recommendation_score, selection_metrics = select_recommended_alternative(
        request.preference,
        response.alternatives,
    )
    return RouteRecommendationResponse(
        map_id=response.map_id,
        start_id=response.start_id,
        destination_id=response.destination_id,
        recommendation=RouteRecommendation(
            selected_rank=selected.rank,
            preference=request.preference,
            recommendation_score=recommendation_score,
            reasons=recommendation_reasons(request.preference),
            selection_metrics=selection_metrics,
            tradeoffs=route_tradeoffs(response.alternatives, selected),
        ),
        selected=selected,
        alternatives=response.alternatives,
        predictions=response.predictions,
    )


def multi_stop_pairwise_routes(
    venue_map: dict[str, Any],
    start_id: str,
    destination_ids: list[str],
    multipliers: dict[str, float],
    blocked_edge_ids: set[str],
    turn_penalty: float,
    algorithm: SearchAlgorithm,
) -> dict[tuple[str, str], dict[str, Any]]:
    node_ids = [start_id, *destination_ids]
    pairwise_routes: dict[tuple[str, str], dict[str, Any]] = {}
    for from_id in node_ids:
        for to_id in destination_ids:
            if from_id == to_id:
                continue
            route_result = calculate_shortest_route(
                venue_map["nodes"],
                venue_map["edges"],
                multipliers,
                from_id,
                to_id,
                blocked_edge_ids=blocked_edge_ids,
                turn_penalty=turn_penalty,
                algorithm=algorithm,
            )
            if route_result is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": "Route not found for one of the requested stops.",
                        "start_id": from_id,
                        "destination_id": to_id,
                        "map_id": venue_map["id"],
                    },
                )
            pairwise_routes[(from_id, to_id)] = route_result
    return pairwise_routes


def nearest_multi_stop_order(
    start_id: str,
    destination_ids: list[str],
    pairwise_routes: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    ordered_destination_ids: list[str] = []
    remaining_destination_ids = list(destination_ids)
    current_id = start_id
    while remaining_destination_ids:
        next_destination_id = min(
            remaining_destination_ids,
            key=lambda destination_id: pairwise_routes[
                (current_id, destination_id)
            ]["weighted_cost"],
        )
        ordered_destination_ids.append(next_destination_id)
        remaining_destination_ids.remove(next_destination_id)
        current_id = next_destination_id
    return ordered_destination_ids


def optimal_multi_stop_order(
    start_id: str,
    destination_ids: list[str],
    pairwise_routes: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    destination_count = len(destination_ids)
    full_mask = (1 << destination_count) - 1
    costs: dict[tuple[int, int], float] = {}
    previous: dict[tuple[int, int], int | None] = {}

    for index, destination_id in enumerate(destination_ids):
        mask = 1 << index
        costs[(mask, index)] = float(
            pairwise_routes[(start_id, destination_id)]["weighted_cost"]
        )
        previous[(mask, index)] = None

    for mask in range(1, full_mask + 1):
        for last_index, last_destination_id in enumerate(destination_ids):
            state = (mask, last_index)
            if state not in costs:
                continue
            for next_index, next_destination_id in enumerate(destination_ids):
                next_bit = 1 << next_index
                if mask & next_bit:
                    continue
                next_mask = mask | next_bit
                next_cost = costs[state] + float(
                    pairwise_routes[(last_destination_id, next_destination_id)][
                        "weighted_cost"
                    ]
                )
                next_state = (next_mask, next_index)
                if next_cost < costs.get(next_state, float("inf")):
                    costs[next_state] = next_cost
                    previous[next_state] = last_index

    best_state = min(
        ((full_mask, index) for index in range(destination_count)),
        key=lambda state: costs.get(state, float("inf")),
    )
    if best_state not in costs:
        raise HTTPException(
            status_code=404,
            detail="No valid multi-stop order found.",
        )

    ordered_reversed: list[str] = []
    mask, current_index = best_state
    while current_index is not None:
        ordered_reversed.append(destination_ids[current_index])
        prev_index = previous[(mask, current_index)]
        mask &= ~(1 << current_index)
        current_index = prev_index

    return list(reversed(ordered_reversed))


def multi_stop_order(
    request: RouteMultiStopRequest,
    venue_map: dict[str, Any],
    multipliers: dict[str, float],
    blocked_edge_ids: set[str],
) -> tuple[list[str], int]:
    turn_penalty = preference_turn_penalty(request.preference)
    pairwise_routes = multi_stop_pairwise_routes(
        venue_map,
        request.start_id,
        request.destination_ids,
        multipliers,
        blocked_edge_ids,
        turn_penalty,
        request.algorithm,
    )
    if request.order_algorithm == "nearest":
        return (
            nearest_multi_stop_order(
                request.start_id,
                request.destination_ids,
                pairwise_routes,
            ),
            len(pairwise_routes),
        )
    return (
        optimal_multi_stop_order(
            request.start_id,
            request.destination_ids,
            pairwise_routes,
        ),
        len(pairwise_routes),
    )


@app.post("/route-multi-stop", response_model=RouteMultiStopResponse)
def route_multi_stop(request: RouteMultiStopRequest):
    venue_map = load_map(request.map_id)
    if request.start_id in request.destination_ids:
        raise HTTPException(
            status_code=400,
            detail="destination_ids must not include start_id.",
        )
    if len(set(request.destination_ids)) != len(request.destination_ids):
        raise HTTPException(
            status_code=400,
            detail="destination_ids must not contain duplicates.",
        )
    validate_route_node_ids(
        venue_map,
        [request.start_id, *request.destination_ids],
    )
    blocked_edge_ids = active_blocked_edge_ids(venue_map, request.blocked_edge_ids)
    crowd_inputs = request.crowd_inputs or estimated_crowd_inputs(request.map_id)
    prediction_response, applied_predictions = route_predictions(
        venue_map["edges"],
        crowd_inputs,
        request.use_congestion,
    )
    multipliers = route_cost_multipliers(
        venue_map["edges"],
        applied_predictions,
        request.preference,
    )
    ordered_destination_ids, pairwise_route_count = multi_stop_order(
        request,
        venue_map,
        multipliers,
        blocked_edge_ids,
    )

    if request.log_route_intent:
        for destination_id in ordered_destination_ids:
            record_route_intent(
                map_id=request.map_id,
                destination_id=destination_id,
                region=region_for_destination(destination_id),
            )

    legs = []
    total_distance = 0.0
    weighted_cost = 0.0
    estimated_total_seconds = 0
    current_id = request.start_id
    for index, destination_id in enumerate(ordered_destination_ids):
        leg_route = build_route_response(
            venue_map,
            RouteRequest(
                map_id=request.map_id,
                start_id=current_id,
                destination_id=destination_id,
                crowd_inputs=crowd_inputs,
                walking_speed_mps=request.walking_speed_mps,
                use_congestion=request.use_congestion,
                log_route_intent=False,
                preference=request.preference,
                algorithm=request.algorithm,
                blocked_edge_ids=sorted(blocked_edge_ids),
            ),
        )
        legs.append(
            MultiStopLeg(
                leg_index=index + 1,
                start_id=current_id,
                destination_id=destination_id,
                route=leg_route,
            )
        )
        total_distance += leg_route.total_distance
        weighted_cost += leg_route.weighted_cost
        estimated_total_seconds += leg_route.estimated_seconds
        current_id = destination_id

    return RouteMultiStopResponse(
        map_id=venue_map["id"],
        start_id=request.start_id,
        order_algorithm=request.order_algorithm,
        pairwise_route_count=pairwise_route_count,
        ordered_destination_ids=ordered_destination_ids,
        total_distance=round(total_distance, 1),
        weighted_cost=round(weighted_cost, 1),
        estimated_seconds=estimated_total_seconds,
        legs=legs,
        crowd_inputs=crowd_inputs,
    )


@app.post("/route-simulation", response_model=RouteSimulationResponse)
def route_simulation(request: RouteSimulationRequest):
    load_map(request.map_id)
    results: list[RouteSimulationResult] = []
    default_crowd_inputs = (
        request.default_crowd_inputs or estimated_crowd_inputs(request.map_id)
    )

    for scenario in request.scenarios:
        walking_speed_mps = (
            scenario.walking_speed_mps
            if scenario.walking_speed_mps is not None
            else request.default_walking_speed_mps
        )
        use_congestion = (
            scenario.use_congestion
            if scenario.use_congestion is not None
            else request.default_use_congestion
        )
        preference = scenario.preference or request.default_preference
        algorithm = scenario.algorithm or request.default_algorithm
        blocked_edge_ids = sorted(
            set(request.default_blocked_edge_ids) | set(scenario.blocked_edge_ids)
        )
        try:
            route_response = route(
                RouteRequest(
                    map_id=request.map_id,
                    start_id=scenario.start_id,
                    destination_id=scenario.destination_id,
                    crowd_inputs=scenario.crowd_inputs or default_crowd_inputs,
                    walking_speed_mps=walking_speed_mps,
                    use_congestion=use_congestion,
                    log_route_intent=False,
                    preference=preference,
                    algorithm=algorithm,
                    blocked_edge_ids=blocked_edge_ids,
                )
            )
            results.append(
                RouteSimulationResult(
                    scenario_id=scenario.id,
                    ok=True,
                    start_id=scenario.start_id,
                    destination_id=scenario.destination_id,
                    preference=preference,
                    algorithm=route_response.algorithm,
                    expanded_state_count=route_response.expanded_state_count,
                    blocked_edge_ids=route_response.blocked_edge_ids,
                    path=route_response.path,
                    edge_ids=route_response.edge_ids,
                    total_distance=route_response.total_distance,
                    weighted_cost=route_response.weighted_cost,
                    estimated_seconds=route_response.estimated_seconds,
                )
            )
        except HTTPException as error:
            results.append(
                RouteSimulationResult(
                    scenario_id=scenario.id,
                    ok=False,
                    start_id=scenario.start_id,
                    destination_id=scenario.destination_id,
                    preference=preference,
                    algorithm=algorithm,
                    blocked_edge_ids=blocked_edge_ids,
                    error={
                        "status_code": error.status_code,
                        "detail": error.detail,
                    },
                )
            )

    success_count = sum(1 for result in results if result.ok)
    failure_count = len(results) - success_count
    return RouteSimulationResponse(
        map_id=request.map_id,
        scenario_count=len(results),
        success_count=success_count,
        failure_count=failure_count,
        pass_rate=round(success_count / max(len(results), 1), 3),
        failures=[result.scenario_id for result in results if not result.ok],
        results=results,
    )


def build_route_response(venue_map: dict, request: RouteRequest) -> RouteResponse:
    edges = venue_map["edges"]
    validate_route_nodes(venue_map, request.start_id, request.destination_id)
    blocked_edge_ids = active_blocked_edge_ids(venue_map, request.blocked_edge_ids)
    crowd_inputs = request.crowd_inputs or estimated_crowd_inputs(request.map_id)
    prediction_response, applied_predictions = route_predictions(
        edges,
        crowd_inputs,
        request.use_congestion,
    )
    multipliers = route_cost_multipliers(
        edges,
        applied_predictions,
        request.preference,
    )

    route_result = calculate_shortest_route(
        venue_map["nodes"],
        edges,
        multipliers,
        request.start_id,
        request.destination_id,
        blocked_edge_ids=blocked_edge_ids,
        turn_penalty=preference_turn_penalty(request.preference),
        algorithm=request.algorithm,
    )
    if route_result is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Route not found between the selected nodes.",
                "start_id": request.start_id,
                "destination_id": request.destination_id,
                "map_id": request.map_id,
            },
        )

    segments = route_segments_for_edges(
        venue_map,
        route_result["path"],
        route_result["edge_ids"],
        applied_predictions,
        request.walking_speed_mps,
    )

    return RouteResponse(
        map_id=venue_map["id"],
        use_congestion=request.use_congestion,
        walking_speed_mps=request.walking_speed_mps,
        preference=request.preference,
        algorithm=route_result["algorithm"],
        expanded_state_count=route_result["expanded_state_count"],
        blocked_edge_ids=sorted(blocked_edge_ids),
        path=route_result["path"],
        edge_ids=route_result["edge_ids"],
        route_points=route_points_for_path(venue_map, route_result["path"]),
        segments=segments,
        total_distance=route_result["total_distance"],
        weighted_cost=route_result["weighted_cost"],
        estimated_seconds=estimated_seconds(
            route_result["total_distance"],
            request.walking_speed_mps,
        ),
        instructions=[segment.instruction for segment in segments],
        crowd_inputs=crowd_inputs,
        model_type=prediction_response.model_type,
        validation_mae=prediction_response.validation_mae,
        predictions=applied_predictions,
    )


def route_predictions(
    edges: list[dict[str, Any]],
    crowd_inputs: CrowdInputs,
    use_congestion: bool,
) -> tuple[PredictionResponse, list[EdgePrediction]]:
    prediction_request = PredictionRequest(
        edges=[
            EdgeFeatures(
                edge_id=edge["id"],
                people_count=people_count_for_edge(edge, crowd_inputs),
                corridor_width_m=edge["widthM"],
                hour=crowd_inputs.hour,
                event_phase=crowd_inputs.event_phase,
                booth_zone=1 if edge["zone"] == "booth" else 0,
                recent_inflow=crowd_inputs.recent_inflow,
            )
            for edge in edges
        ]
    )
    prediction_response = predict_congestion(prediction_request)
    if use_congestion:
        return prediction_response, prediction_response.predictions

    return prediction_response, [
        EdgePrediction(edge_id=edge["id"], multiplier=1.0, level="clear")
        for edge in edges
    ]
