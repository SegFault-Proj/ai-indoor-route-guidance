import type {
  AppConfig,
  CoordinateSnapResult,
  CrowdEstimate,
  CrowdInputs,
  DemandHeatmapResult,
  MapCheckpoint,
  MapDeleteResult,
  MapEdge,
  MapGenerationDraftGenerateOptions,
  MapGenerationDraftGenerateResult,
  MapGenerationDraftReview,
  MapGenerationJob,
  MapGenerationJobCreate,
  MapGenerationPostprocessOptions,
  MapGenerationPostprocessResult,
  MapGenerationRoutePreviewRequest,
  MapGenerationSaveResult,
  MapImprovementSuggestionsResult,
  MapSummary,
  MapSaveResult,
  MapSchema,
  MapTopologyAnalysisResult,
  MapValidationResult,
  NavigationSession,
  NavigationGuidanceResult,
  NavigationSessionCoordinateUpdate,
  NavigationSessionPositionUpdate,
  NavigationSessionStart,
  NavigationPositionUpdate,
  NavigationUpdateResult,
  PredictedEdgeWeight,
  RouteApiResult,
  EtaCalibrationResult,
  RouteAlternative,
  RouteAlternativesResult,
  RouteRecommendationResult,
  RouteRiskAnalysisResult,
  RouteSegment,
  RouteSimulationResult,
  RouteSimulationScenario,
  RouteOptions,
  VenueMap,
} from "./types";

const AI_API_URL =
  import.meta.env.VITE_AI_API_URL ?? "http://127.0.0.1:8000";

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function peopleCountForRegion(
  edge: MapEdge,
  inputs: CrowdInputs
): number {
  if (edge.crowdRegion === "booth") {
    return clampNumber(inputs.boothPeople, 0, 1000);
  }

  return clampNumber(inputs.lobbyPeople, 0, 1000);
}

function levelFromMultiplier(
  multiplier: number
): PredictedEdgeWeight["level"] {
  if (multiplier < 1.2) return "clear";
  if (multiplier < 1.65) return "normal";
  if (multiplier < 2.25) return "busy";
  return "very_busy";
}

function normalizeMapEdge(edge: {
  id: string;
  from: string;
  to: string;
  distance: number;
  widthM: number;
  zone: MapEdge["zone"];
  crowdRegion: MapEdge["crowdRegion"];
  bidirectional: boolean;
}): MapEdge {
  return {
    id: edge.id,
    from: edge.from,
    to: edge.to,
    distance: edge.distance,
    widthM: edge.widthM,
    zone: edge.zone,
    crowdRegion: edge.crowdRegion,
    bidirectional: edge.bidirectional,
  };
}

function normalizeMapCheckpoint(checkpoint: {
  id: string;
  name: string;
  node_id: string;
  region: MapCheckpoint["region"];
}): MapCheckpoint {
  return {
    id: checkpoint.id,
    name: checkpoint.name,
    nodeId: checkpoint.node_id,
    region: checkpoint.region,
  };
}

function normalizeMapSummary(summary: {
  id: string;
  name: string;
  image: string;
  width: number;
  height: number;
  node_count: number;
  edge_count: number;
  checkpoint_count: number;
  selectable_node_count: number;
  valid: boolean;
}): MapSummary {
  return {
    id: summary.id,
    name: summary.name,
    image: summary.image.startsWith("http")
      ? summary.image
      : `${AI_API_URL}${summary.image}`,
    width: summary.width,
    height: summary.height,
    nodeCount: summary.node_count,
    edgeCount: summary.edge_count,
    checkpointCount: summary.checkpoint_count,
    selectableNodeCount: summary.selectable_node_count,
    valid: summary.valid,
  };
}

function normalizeMapValidation(data: {
  map_id?: string | null;
  valid: boolean;
  errors: string[];
  warnings: string[];
  node_count: number;
  edge_count: number;
  checkpoint_count: number;
  selectable_node_count: number;
  unreachable_route_pairs?: Array<{
    start_id: string;
    destination_id: string;
  }>;
  out_of_bounds_node_ids?: string[];
  invalid_node_ids?: string[];
  invalid_edge_ids?: string[];
  invalid_checkpoint_ids?: string[];
  fix_suggestions?: Array<{
    code: string;
    severity: "error" | "warning";
    target_ids: string[];
    message: string;
    allowed_values?: Record<string, string[]>;
    sample_pairs?: Array<{
      start_id: string;
      destination_id: string;
    }>;
  }>;
  quality_summary?: {
    score: number;
    readiness: "ready" | "needs_review" | "blocked";
    can_save: boolean;
    reason_codes: string[];
  };
}): MapValidationResult {
  return {
    mapId: data.map_id ?? null,
    valid: data.valid,
    errors: data.errors,
    warnings: data.warnings,
    nodeCount: data.node_count,
    edgeCount: data.edge_count,
    checkpointCount: data.checkpoint_count,
    selectableNodeCount: data.selectable_node_count,
    unreachableRoutePairs: (data.unreachable_route_pairs ?? []).map((pair) => ({
      startId: pair.start_id,
      destinationId: pair.destination_id,
    })),
    outOfBoundsNodeIds: data.out_of_bounds_node_ids ?? [],
    invalidNodeIds: data.invalid_node_ids ?? [],
    invalidEdgeIds: data.invalid_edge_ids ?? [],
    invalidCheckpointIds: data.invalid_checkpoint_ids ?? [],
    fixSuggestions: (data.fix_suggestions ?? []).map((suggestion) => ({
      code: suggestion.code,
      severity: suggestion.severity,
      targetIds: suggestion.target_ids,
      message: suggestion.message,
      allowedValues: suggestion.allowed_values,
      samplePairs: suggestion.sample_pairs?.map((pair) => ({
        startId: pair.start_id,
        destinationId: pair.destination_id,
      })),
    })),
    qualitySummary: {
      score: data.quality_summary?.score ?? (data.valid ? 100 : 0),
      readiness: data.quality_summary?.readiness ?? (data.valid ? "ready" : "blocked"),
      canSave: data.quality_summary?.can_save ?? data.valid,
      reasonCodes: data.quality_summary?.reason_codes ?? [],
    },
  };
}

function normalizeMapGenerationJob(data: {
  job_id: string;
  status: MapGenerationJob["status"];
  target_map_id: string;
  source_filename: string;
  source_mime_type: string;
  source_image_path: string;
  source_image_url: string;
  notes: string | null;
  created_at: string;
  updated_at: string;
  validation: Parameters<typeof normalizeMapValidation>[0] | null;
  saved_map_id: string | null;
  next_step: string;
}): MapGenerationJob {
  return {
    jobId: data.job_id,
    status: data.status,
    targetMapId: data.target_map_id,
    sourceFilename: data.source_filename,
    sourceMimeType: data.source_mime_type,
    sourceImagePath: data.source_image_path,
    sourceImageUrl: data.source_image_url.startsWith("http")
      ? data.source_image_url
      : `${AI_API_URL}${data.source_image_url}`,
    notes: data.notes,
    createdAt: data.created_at,
    updatedAt: data.updated_at,
    validation: data.validation ? normalizeMapValidation(data.validation) : null,
    savedMapId: data.saved_map_id,
    nextStep: data.next_step,
  };
}

function normalizeMapSchema(data: {
  required_fields: string[];
  node_fields: string[];
  edge_fields: string[];
  checkpoint_fields: string[];
  node_types: MapSchema["nodeTypes"];
  zones: MapSchema["zones"];
  crowd_regions: MapSchema["crowdRegions"];
  checkpoint_regions: MapSchema["checkpointRegions"];
  coordinate_system: string;
  notes: string[];
}): MapSchema {
  return {
    requiredFields: data.required_fields,
    nodeFields: data.node_fields,
    edgeFields: data.edge_fields,
    checkpointFields: data.checkpoint_fields,
    nodeTypes: data.node_types,
    zones: data.zones,
    crowdRegions: data.crowd_regions,
    checkpointRegions: data.checkpoint_regions,
    coordinateSystem: data.coordinate_system,
    notes: data.notes,
  };
}

function normalizeCrowdEstimate(data: {
  map_id: string;
  lobby_people: number;
  booth_people: number;
  recent_inflow: number;
  hour: number;
  event_phase: number;
  signals: {
    window_minutes: number;
    qr_scans: {
      lobby: number;
      booth: number;
    };
    route_intents: {
      lobby: number;
      booth: number;
    };
    manual_estimate: {
      lobby_people: number;
      booth_people: number;
    };
  };
}): CrowdEstimate {
  return {
    mapId: data.map_id,
    lobbyPeople: data.lobby_people,
    boothPeople: data.booth_people,
    recentInflow: data.recent_inflow,
    hour: data.hour,
    eventPhase: data.event_phase,
    signals: {
      windowMinutes: data.signals.window_minutes,
      qrScans: data.signals.qr_scans,
      routeIntents: data.signals.route_intents,
      manualEstimate: data.signals.manual_estimate,
    },
  };
}

function normalizeAppConfig(data: {
  api_version: string;
  default_map_id: string;
  default_walking_speed_mps: number;
  max_walking_speed_mps: number;
  telemetry_window_minutes: number;
  limits: {
    people_count: number;
    recent_inflow: number;
    qr_scan_count: number;
    telemetry_snapshot_limit: number;
  };
  event_phase_labels: Record<number, string>;
  features: {
    route_api: boolean;
    congestion_prediction: boolean;
    estimated_crowd_without_camera: boolean;
    manual_crowd_input: boolean;
    qr_scan_telemetry: boolean;
    route_intent_telemetry: boolean;
    checkpoint_position_update: boolean;
    coordinate_position_snap: boolean;
    llm_map_validation: boolean;
    llm_map_storage: boolean;
    llm_map_generation_jobs: boolean;
    llm_map_generate_draft: boolean;
    route_preferences: boolean;
    temporary_blocked_edges: boolean;
    route_alternatives: boolean;
    route_recommendation: boolean;
    multi_stop_route: boolean;
    crowd_forecast: boolean;
    bottleneck_detection: boolean;
    eta_calibration: boolean;
    route_risk_analysis: boolean;
    map_topology_analysis: boolean;
    route_simulation: boolean;
    demand_heatmap: boolean;
    navigation_guidance: boolean;
    map_improvement_suggestions: boolean;
  };
  endpoints: AppConfig["endpoints"];
}): AppConfig {
  return {
    apiVersion: data.api_version,
    defaultMapId: data.default_map_id,
    defaultWalkingSpeedMps: data.default_walking_speed_mps,
    maxWalkingSpeedMps: data.max_walking_speed_mps,
    telemetryWindowMinutes: data.telemetry_window_minutes,
    limits: {
      peopleCount: data.limits.people_count,
      recentInflow: data.limits.recent_inflow,
      qrScanCount: data.limits.qr_scan_count,
      telemetrySnapshotLimit: data.limits.telemetry_snapshot_limit,
    },
    eventPhaseLabels: data.event_phase_labels,
    features: {
      routeApi: data.features.route_api,
      congestionPrediction: data.features.congestion_prediction,
      estimatedCrowdWithoutCamera: data.features.estimated_crowd_without_camera,
      manualCrowdInput: data.features.manual_crowd_input,
      qrScanTelemetry: data.features.qr_scan_telemetry,
      routeIntentTelemetry: data.features.route_intent_telemetry,
      checkpointPositionUpdate: data.features.checkpoint_position_update,
      coordinatePositionSnap: Boolean(data.features.coordinate_position_snap),
      llmMapValidation: data.features.llm_map_validation,
      llmMapStorage: data.features.llm_map_storage,
      llmMapGenerationJobs: Boolean(data.features.llm_map_generation_jobs),
      llmMapGenerateDraft: Boolean(data.features.llm_map_generate_draft),
      routePreferences: Boolean(data.features.route_preferences),
      temporaryBlockedEdges: Boolean(data.features.temporary_blocked_edges),
      routeAlternatives: Boolean(data.features.route_alternatives),
      routeRecommendation: Boolean(data.features.route_recommendation),
      multiStopRoute: Boolean(data.features.multi_stop_route),
      crowdForecast: Boolean(data.features.crowd_forecast),
      bottleneckDetection: Boolean(data.features.bottleneck_detection),
      etaCalibration: Boolean(data.features.eta_calibration),
      routeRiskAnalysis: Boolean(data.features.route_risk_analysis),
      mapTopologyAnalysis: Boolean(data.features.map_topology_analysis),
      routeSimulation: Boolean(data.features.route_simulation),
      demandHeatmap: Boolean(data.features.demand_heatmap),
      navigationGuidance: Boolean(data.features.navigation_guidance),
      mapImprovementSuggestions: Boolean(data.features.map_improvement_suggestions),
    },
    endpoints: data.endpoints,
  };
}

function normalizeRoutePoint(point: {
  node_id: string;
  name: string;
  x: number;
  y: number;
}) {
  return {
    nodeId: point.node_id,
    name: point.name,
    x: point.x,
    y: point.y,
  };
}

function normalizeCoordinateSnap(data: {
  map_id: string;
  input_x: number;
  input_y: number;
  snapped_node: Parameters<typeof normalizeRoutePoint>[0];
  distance_px: number;
  within_threshold: boolean;
  selectable: boolean;
  snapped_edge_id: string | null;
  snapped_edge_from_node_id: string | null;
  snapped_edge_to_node_id: string | null;
  edge_distance_px: number | null;
  edge_progress_ratio: number | null;
  projected_x: number | null;
  projected_y: number | null;
}): CoordinateSnapResult {
  return {
    mapId: data.map_id,
    inputX: data.input_x,
    inputY: data.input_y,
    snappedNode: normalizeRoutePoint(data.snapped_node),
    distancePx: data.distance_px,
    withinThreshold: data.within_threshold,
    selectable: data.selectable,
    snappedEdgeId: data.snapped_edge_id,
    snappedEdgeFromNodeId: data.snapped_edge_from_node_id,
    snappedEdgeToNodeId: data.snapped_edge_to_node_id,
    edgeDistancePx: data.edge_distance_px,
    edgeProgressRatio: data.edge_progress_ratio,
    projectedX: data.projected_x,
    projectedY: data.projected_y,
  };
}

function normalizeRouteResponse(data: {
  preference?: RouteApiResult["preference"];
  blocked_edge_ids?: string[];
  path: string[];
  edge_ids: string[];
  route_points: Array<{
    node_id: string;
    name: string;
    x: number;
    y: number;
  }>;
  segments: Array<{
    edge_id: string;
    from_node: {
      node_id: string;
      name: string;
      x: number;
      y: number;
    };
    to_node: {
      node_id: string;
      name: string;
      x: number;
      y: number;
    };
    distance: number;
    multiplier: number;
    weighted_cost: number;
    estimated_seconds: number;
    level: PredictedEdgeWeight["level"];
    maneuver: RouteSegment["maneuver"];
    heading_degrees: number;
    turn_degrees: number;
    instruction: string;
  }>;
  total_distance: number;
  weighted_cost: number;
  estimated_seconds: number;
  instructions: string[];
  crowd_inputs: {
    lobby_people: number;
    booth_people: number;
    recent_inflow: number;
    hour: number;
    event_phase: 0 | 1 | 2 | 3;
  };
  predictions: Array<{
    edge_id: string;
    multiplier: number;
    level: PredictedEdgeWeight["level"];
  }>;
}): RouteApiResult {
  return {
    route: {
      path: data.path,
      edgeIds: data.edge_ids,
      totalDistance: data.total_distance,
      weightedCost: data.weighted_cost,
    },
    routePoints: data.route_points.map(normalizeRoutePoint),
    segments: data.segments.map(normalizeRouteSegment),
    predictions: data.predictions.map((prediction) => ({
      edgeId: prediction.edge_id,
      multiplier: prediction.multiplier,
      level: prediction.level,
      source: "ai",
    })),
    instructions: data.instructions,
    estimatedSeconds: data.estimated_seconds,
    preference: data.preference,
    blockedEdgeIds: data.blocked_edge_ids ?? [],
    crowdInputs: {
      lobbyPeople: data.crowd_inputs.lobby_people,
      boothPeople: data.crowd_inputs.booth_people,
      recentInflow: data.crowd_inputs.recent_inflow,
      hour: data.crowd_inputs.hour,
      eventPhase: data.crowd_inputs.event_phase,
    },
  };
}

type RouteAlternativeResponse = {
  rank: number;
  path: string[];
  edge_ids: string[];
  route_points: Parameters<typeof normalizeRoutePoint>[0][];
  segments: Parameters<typeof normalizeRouteResponse>[0]["segments"];
  total_distance: number;
  weighted_cost: number;
  estimated_seconds: number;
  instructions: string[];
  overlap_with_best_edge_count: number;
};

function normalizeRouteAlternative(
  alternative: RouteAlternativeResponse
): RouteAlternative {
  return {
    rank: alternative.rank,
    route: {
      path: alternative.path,
      edgeIds: alternative.edge_ids,
      totalDistance: alternative.total_distance,
      weightedCost: alternative.weighted_cost,
    },
    routePoints: alternative.route_points.map(normalizeRoutePoint),
    segments: alternative.segments.map(normalizeRouteSegment),
    instructions: alternative.instructions,
    estimatedSeconds: alternative.estimated_seconds,
    overlapWithBestEdgeCount: alternative.overlap_with_best_edge_count,
  };
}

function normalizeRouteSegment(
  segment: Parameters<typeof normalizeRouteResponse>[0]["segments"][number]
): RouteSegment {
  return {
    edgeId: segment.edge_id,
    fromNode: normalizeRoutePoint(segment.from_node),
    toNode: normalizeRoutePoint(segment.to_node),
    distance: segment.distance,
    multiplier: segment.multiplier,
    weightedCost: segment.weighted_cost,
    estimatedSeconds: segment.estimated_seconds,
    level: segment.level,
    maneuver: segment.maneuver,
    headingDegrees: segment.heading_degrees,
    turnDegrees: segment.turn_degrees,
    instruction: segment.instruction,
  };
}

function crowdInputsPayload(inputs: CrowdInputs | undefined) {
  if (!inputs) return undefined;

  return {
    lobby_people: clampNumber(inputs.lobbyPeople, 0, 1000),
    booth_people: clampNumber(inputs.boothPeople, 0, 1000),
    recent_inflow: clampNumber(inputs.recentInflow, 0, 1000),
    hour: clampNumber(inputs.hour, 0, 23),
    event_phase: clampNumber(inputs.eventPhase, 0, 3),
  };
}

function normalizeNavigationSession(data: {
  session_id: string;
  map_id: string;
  start_node: Parameters<typeof normalizeRoutePoint>[0];
  current_node: Parameters<typeof normalizeRoutePoint>[0];
  destination_node: Parameters<typeof normalizeRoutePoint>[0];
  status: "active" | "arrived";
  created_at: string;
  updated_at: string;
  route: Parameters<typeof normalizeRouteResponse>[0];
  recent_updates: Array<{
    node_id: string;
    source: string;
    checkpoint_id: string | null;
    timestamp: string;
  }>;
}): NavigationSession {
  return {
    sessionId: data.session_id,
    mapId: data.map_id,
    startNode: normalizeRoutePoint(data.start_node),
    currentNode: normalizeRoutePoint(data.current_node),
    destinationNode: normalizeRoutePoint(data.destination_node),
    status: data.status,
    createdAt: data.created_at,
    updatedAt: data.updated_at,
    route: normalizeRouteResponse(data.route),
    recentUpdates: data.recent_updates.map((event) => ({
      nodeId: event.node_id,
      source: event.source,
      checkpointId: event.checkpoint_id,
      timestamp: event.timestamp,
    })),
  };
}

/**
 * AI 서버가 꺼져 있어도 시연이 멈추지 않도록 하는 로컬 대체 계산입니다.
 * 발표 시에는 AI 서버 응답 여부가 화면에 표시됩니다.
 */
function fallbackPrediction(
  edge: MapEdge,
  inputs: CrowdInputs
): PredictedEdgeWeight {
  const people = peopleCountForRegion(edge, inputs);

  const density = people / Math.max(edge.widthM, 1);
  const phasePenalty =
    inputs.eventPhase === 2 ? 0.25 : inputs.eventPhase === 3 ? 0.55 : 0;

  const multiplier = Math.min(
    3.5,
    Math.max(
      1,
      1 +
        density / 45 +
        inputs.recentInflow / 130 +
        phasePenalty +
        (edge.zone === "booth" ? 0.15 : 0)
    )
  );

  return {
    edgeId: edge.id,
    multiplier: Number(multiplier.toFixed(2)),
    level: levelFromMultiplier(multiplier),
    source: "fallback",
  };
}

export async function predictEdgeWeights(
  edges: MapEdge[],
  inputs: CrowdInputs
): Promise<PredictedEdgeWeight[]> {
  try {
    const requestBody = {
      edges: edges.map((edge) => ({
        edge_id: edge.id,
        people_count: peopleCountForRegion(edge, inputs),
        corridor_width_m: clampNumber(edge.widthM, 0.1, 30),
        hour: clampNumber(inputs.hour, 0, 23),
        event_phase: clampNumber(inputs.eventPhase, 0, 3),
        booth_zone: edge.zone === "booth" ? 1 : 0,
        recent_inflow: clampNumber(inputs.recentInflow, 0, 1000),
      })),
    };

    const response = await fetch(`${AI_API_URL}/predict-congestion`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`AI API 오류: ${response.status} ${errorText}`);
    }

    const data = (await response.json()) as {
      predictions: Array<{
        edge_id: string;
        multiplier: number;
        level: PredictedEdgeWeight["level"];
      }>;
    };

    return data.predictions.map((prediction) => ({
      edgeId: prediction.edge_id,
      multiplier: prediction.multiplier,
      level: prediction.level,
      source: "ai",
    }));
  } catch (error) {
    console.warn("AI 서버를 사용할 수 없어 로컬 계산으로 전환합니다.", error);
    return edges.map((edge) => fallbackPrediction(edge, inputs));
  }
}

export async function fetchAppConfig(): Promise<AppConfig> {
  const response = await fetch(`${AI_API_URL}/app-config`);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`App config API 오류: ${response.status} ${errorText}`);
  }

  return normalizeAppConfig(await response.json());
}

export async function calculateRoute(
  startId: string,
  destinationId: string,
  inputs: CrowdInputs,
  options: RouteOptions,
  mapId = "default"
): Promise<RouteApiResult> {
  const response = await fetch(`${AI_API_URL}/route`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: mapId,
      start_id: startId,
      destination_id: destinationId,
      walking_speed_mps: clampNumber(options.walkingSpeedMps, 0.1, 3),
      use_congestion: options.useCongestion,
      preference: options.preference,
      blocked_edge_ids: options.blockedEdgeIds,
      crowd_inputs: options.useEstimatedCrowd ? undefined : crowdInputsPayload(inputs),
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Route API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as Parameters<typeof normalizeRouteResponse>[0];

  return normalizeRouteResponse(data);
}

export async function calculateRouteAlternatives(
  startId: string,
  destinationId: string,
  inputs: CrowdInputs,
  options: RouteOptions,
  mapId = "default",
  maxRoutes = 3
): Promise<RouteAlternativesResult> {
  const response = await fetch(`${AI_API_URL}/route-alternatives`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: mapId,
      start_id: startId,
      destination_id: destinationId,
      walking_speed_mps: clampNumber(options.walkingSpeedMps, 0.1, 3),
      use_congestion: options.useCongestion,
      preference: options.preference,
      blocked_edge_ids: options.blockedEdgeIds,
      crowd_inputs: options.useEstimatedCrowd ? undefined : crowdInputsPayload(inputs),
      max_routes: clampNumber(maxRoutes, 1, 5),
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Route alternatives API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    start_id: string;
    destination_id: string;
    use_congestion: boolean;
    walking_speed_mps: number;
    preference?: RouteAlternativesResult["preference"];
    blocked_edge_ids?: string[];
    crowd_inputs: Parameters<typeof normalizeRouteResponse>[0]["crowd_inputs"];
    alternatives: RouteAlternativeResponse[];
    predictions: Array<{
      edge_id: string;
      multiplier: number;
      level: PredictedEdgeWeight["level"];
    }>;
  };

  return {
    mapId: data.map_id,
    startId: data.start_id,
    destinationId: data.destination_id,
    useCongestion: data.use_congestion,
    walkingSpeedMps: data.walking_speed_mps,
    preference: data.preference,
    blockedEdgeIds: data.blocked_edge_ids ?? [],
    crowdInputs: {
      lobbyPeople: data.crowd_inputs.lobby_people,
      boothPeople: data.crowd_inputs.booth_people,
      recentInflow: data.crowd_inputs.recent_inflow,
      hour: data.crowd_inputs.hour,
      eventPhase: data.crowd_inputs.event_phase,
    },
    alternatives: data.alternatives.map(normalizeRouteAlternative),
    predictions: data.predictions.map((prediction) => ({
      edgeId: prediction.edge_id,
      multiplier: prediction.multiplier,
      level: prediction.level,
      source: "ai",
    })),
  };
}

export async function calculateRouteRecommendation(
  startId: string,
  destinationId: string,
  inputs: CrowdInputs,
  options: RouteOptions,
  mapId = "default",
  maxRoutes = 3
): Promise<RouteRecommendationResult> {
  const response = await fetch(`${AI_API_URL}/route-recommendation`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: mapId,
      start_id: startId,
      destination_id: destinationId,
      walking_speed_mps: clampNumber(options.walkingSpeedMps, 0.1, 3),
      use_congestion: options.useCongestion,
      preference: options.preference,
      blocked_edge_ids: options.blockedEdgeIds,
      crowd_inputs: options.useEstimatedCrowd ? undefined : crowdInputsPayload(inputs),
      max_routes: clampNumber(maxRoutes, 1, 5),
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Route recommendation API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    start_id: string;
    destination_id: string;
    recommendation: {
      selected_rank: number;
      preference: NonNullable<RouteApiResult["preference"]>;
      reasons: string[];
      tradeoffs: Record<string, unknown>;
    };
    selected: RouteAlternativeResponse;
    alternatives: RouteAlternativeResponse[];
    predictions: Array<{
      edge_id: string;
      multiplier: number;
      level: PredictedEdgeWeight["level"];
    }>;
  };

  return {
    mapId: data.map_id,
    startId: data.start_id,
    destinationId: data.destination_id,
    recommendation: {
      selectedRank: data.recommendation.selected_rank,
      preference: data.recommendation.preference,
      reasons: data.recommendation.reasons,
      tradeoffs: data.recommendation.tradeoffs,
    },
    selected: normalizeRouteAlternative(data.selected),
    alternatives: data.alternatives.map(normalizeRouteAlternative),
    predictions: data.predictions.map((prediction) => ({
      edgeId: prediction.edge_id,
      multiplier: prediction.multiplier,
      level: prediction.level,
      source: "ai",
    })),
  };
}

export async function runRouteSimulation(
  scenarios: RouteSimulationScenario[],
  options: RouteOptions,
  mapId = "default",
  defaultCrowdInputs?: CrowdInputs
): Promise<RouteSimulationResult> {
  const response = await fetch(`${AI_API_URL}/route-simulation`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: mapId,
      scenarios: scenarios.map((scenario) => ({
        id: scenario.id,
        start_id: scenario.startId,
        destination_id: scenario.destinationId,
        crowd_inputs: crowdInputsPayload(scenario.crowdInputs),
        walking_speed_mps: scenario.walkingSpeedMps,
        use_congestion: scenario.useCongestion,
        preference: scenario.preference,
        blocked_edge_ids: scenario.blockedEdgeIds,
      })),
      default_crowd_inputs: crowdInputsPayload(defaultCrowdInputs),
      default_walking_speed_mps: clampNumber(options.walkingSpeedMps, 0.1, 3),
      default_use_congestion: options.useCongestion,
      default_preference: options.preference,
      default_blocked_edge_ids: options.blockedEdgeIds,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Route simulation API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    scenario_count: number;
    success_count: number;
    failure_count: number;
    pass_rate: number;
    failures: string[];
    results: Array<{
      scenario_id: string;
      ok: boolean;
      start_id: string;
      destination_id: string;
      preference: NonNullable<RouteApiResult["preference"]>;
      blocked_edge_ids: string[];
      path: string[];
      edge_ids: string[];
      total_distance: number | null;
      weighted_cost: number | null;
      estimated_seconds: number | null;
      error: Record<string, unknown> | null;
    }>;
  };

  return {
    mapId: data.map_id,
    scenarioCount: data.scenario_count,
    successCount: data.success_count,
    failureCount: data.failure_count,
    passRate: data.pass_rate,
    failures: data.failures,
    results: data.results.map((result) => ({
      scenarioId: result.scenario_id,
      ok: result.ok,
      startId: result.start_id,
      destinationId: result.destination_id,
      preference: result.preference,
      blockedEdgeIds: result.blocked_edge_ids,
      path: result.path,
      edgeIds: result.edge_ids,
      totalDistance: result.total_distance,
      weightedCost: result.weighted_cost,
      estimatedSeconds: result.estimated_seconds,
      error: result.error,
    })),
  };
}

export async function updateNavigationPosition(
  update: NavigationPositionUpdate,
  mapId = "default"
): Promise<NavigationUpdateResult> {
  const targetMapId = update.mapId ?? mapId;
  const response = await fetch(`${AI_API_URL}/navigation/update-position`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: targetMapId,
      current_node_id: update.currentNodeId,
      destination_id: update.destinationId,
      checkpoint_id: update.checkpointId,
      region: update.region,
      walking_speed_mps: clampNumber(update.options.walkingSpeedMps, 0.1, 3),
      use_congestion: update.options.useCongestion,
      preference: update.options.preference,
      blocked_edge_ids: update.options.blockedEdgeIds,
      crowd_inputs: update.options.useEstimatedCrowd
        ? undefined
        : crowdInputsPayload(update.crowdInputs),
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Position update API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    current_node: {
      node_id: string;
      name: string;
      x: number;
      y: number;
    };
    destination_node: {
      node_id: string;
      name: string;
      x: number;
      y: number;
    };
    location_source: "node" | "checkpoint";
    route: Parameters<typeof normalizeRouteResponse>[0];
  };

  return {
    mapId: data.map_id,
    currentNode: normalizeRoutePoint(data.current_node),
    destinationNode: normalizeRoutePoint(data.destination_node),
    locationSource: data.location_source,
    route: normalizeRouteResponse(data.route),
  };
}

export async function snapCoordinate(
  x: number,
  y: number,
  mapId = "default",
  maxDistancePx = 120,
  selectableOnly = false
): Promise<CoordinateSnapResult> {
  const response = await fetch(`${AI_API_URL}/maps/${mapId}/snap-coordinate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      x,
      y,
      max_distance_px: maxDistancePx,
      selectable_only: selectableOnly,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Coordinate snap API 오류: ${response.status} ${errorText}`);
  }

  return normalizeCoordinateSnap(await response.json());
}

export async function startNavigationSession(
  session: NavigationSessionStart,
  mapId = "default"
): Promise<NavigationSession> {
  const targetMapId = session.mapId ?? mapId;
  const response = await fetch(`${AI_API_URL}/navigation/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: targetMapId,
      start_id: session.startId,
      destination_id: session.destinationId,
      walking_speed_mps: clampNumber(session.options.walkingSpeedMps, 0.1, 3),
      use_congestion: session.options.useCongestion,
      preference: session.options.preference,
      blocked_edge_ids: session.options.blockedEdgeIds,
      crowd_inputs: session.options.useEstimatedCrowd
        ? undefined
        : crowdInputsPayload(session.crowdInputs),
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Navigation session API 오류: ${response.status} ${errorText}`);
  }

  return normalizeNavigationSession(await response.json());
}

export async function fetchNavigationSession(
  sessionId: string
): Promise<NavigationSession> {
  const response = await fetch(`${AI_API_URL}/navigation/sessions/${sessionId}`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Navigation session API 오류: ${response.status} ${errorText}`);
  }

  return normalizeNavigationSession(await response.json());
}

export async function fetchNavigationGuidance(
  sessionId: string
): Promise<NavigationGuidanceResult> {
  const response = await fetch(
    `${AI_API_URL}/navigation/sessions/${sessionId}/guidance`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Navigation guidance API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    session_id: string;
    map_id: string;
    status: "active" | "arrived";
    current_node: Parameters<typeof normalizeRoutePoint>[0];
    destination_node: Parameters<typeof normalizeRoutePoint>[0];
    off_route: boolean;
    expected_path: string[];
    expected_edge_ids: string[];
    progress_ratio: number;
    remaining_distance: number;
    remaining_seconds: number;
    next_node: Parameters<typeof normalizeRoutePoint>[0] | null;
    next_segment: Parameters<typeof normalizeRouteResponse>[0]["segments"][number] | null;
    instruction: string;
    route: Parameters<typeof normalizeRouteResponse>[0];
  };

  return {
    sessionId: data.session_id,
    mapId: data.map_id,
    status: data.status,
    currentNode: normalizeRoutePoint(data.current_node),
    destinationNode: normalizeRoutePoint(data.destination_node),
    offRoute: data.off_route,
    expectedPath: data.expected_path,
    expectedEdgeIds: data.expected_edge_ids,
    progressRatio: data.progress_ratio,
    remainingDistance: data.remaining_distance,
    remainingSeconds: data.remaining_seconds,
    nextNode: data.next_node ? normalizeRoutePoint(data.next_node) : null,
    nextSegment: data.next_segment ? normalizeRouteSegment(data.next_segment) : null,
    instruction: data.instruction,
    route: normalizeRouteResponse(data.route),
  };
}

export async function fetchNavigationEtaCalibration(
  sessionId: string
): Promise<EtaCalibrationResult> {
  const response = await fetch(
    `${AI_API_URL}/navigation/sessions/${sessionId}/eta-calibration`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`ETA calibration API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    session_id: string;
    map_id: string;
    sample_count: number;
    confidence: number;
    default_walking_speed_mps: number;
    observed_walking_speed_mps: number | null;
    recommended_walking_speed_mps: number;
    samples: Array<{
      from_node_id: string;
      to_node_id: string;
      distance: number;
      elapsed_seconds: number;
      observed_speed_mps: number;
    }>;
  };

  return {
    sessionId: data.session_id,
    mapId: data.map_id,
    sampleCount: data.sample_count,
    confidence: data.confidence,
    defaultWalkingSpeedMps: data.default_walking_speed_mps,
    observedWalkingSpeedMps: data.observed_walking_speed_mps,
    recommendedWalkingSpeedMps: data.recommended_walking_speed_mps,
    samples: data.samples.map((sample) => ({
      fromNodeId: sample.from_node_id,
      toNodeId: sample.to_node_id,
      distance: sample.distance,
      elapsedSeconds: sample.elapsed_seconds,
      observedSpeedMps: sample.observed_speed_mps,
    })),
  };
}

export async function updateNavigationSessionPosition(
  sessionId: string,
  update: NavigationSessionPositionUpdate
): Promise<NavigationSession> {
  const response = await fetch(
    `${AI_API_URL}/navigation/sessions/${sessionId}/position`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        current_node_id: update.currentNodeId,
        checkpoint_id: update.checkpointId,
        region: update.region,
        walking_speed_mps: clampNumber(update.options.walkingSpeedMps, 0.1, 3),
        use_congestion: update.options.useCongestion,
        preference: update.options.preference,
        blocked_edge_ids: update.options.blockedEdgeIds,
        crowd_inputs: update.options.useEstimatedCrowd
          ? undefined
          : crowdInputsPayload(update.crowdInputs),
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Navigation session API 오류: ${response.status} ${errorText}`);
  }

  return normalizeNavigationSession(await response.json());
}

export async function updateNavigationSessionCoordinatePosition(
  sessionId: string,
  update: NavigationSessionCoordinateUpdate
): Promise<NavigationSession> {
  const response = await fetch(
    `${AI_API_URL}/navigation/sessions/${sessionId}/position-coordinate`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        x: update.x,
        y: update.y,
        max_distance_px: update.maxDistancePx,
        region: update.region,
        walking_speed_mps: clampNumber(update.options.walkingSpeedMps, 0.1, 3),
        use_congestion: update.options.useCongestion,
        preference: update.options.preference,
        blocked_edge_ids: update.options.blockedEdgeIds,
        crowd_inputs: update.options.useEstimatedCrowd
          ? undefined
          : crowdInputsPayload(update.crowdInputs),
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Navigation coordinate update API 오류: ${response.status} ${errorText}`
    );
  }

  return normalizeNavigationSession(await response.json());
}

export async function fetchCrowdEstimate(mapId = "default"): Promise<CrowdEstimate> {
  const response = await fetch(`${AI_API_URL}/crowd/${mapId}`);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Crowd API 오류: ${response.status} ${errorText}`);
  }

  return normalizeCrowdEstimate(await response.json());
}

export async function fetchDemandHeatmap(
  mapId = "default"
): Promise<DemandHeatmapResult> {
  const response = await fetch(`${AI_API_URL}/crowd/${mapId}/demand-heatmap`);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Demand heatmap API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    method: string;
    crowd_inputs: Parameters<typeof normalizeRouteResponse>[0]["crowd_inputs"];
    regions: Array<{
      region: string;
      people_estimate: number;
      qr_scans: number;
      route_intents: number;
      demand_score: number;
    }>;
    edges: Array<{
      edge_id: string;
      from_node: string;
      to_node: string;
      zone: string;
      crowd_region: string;
      multiplier: number;
      level: PredictedEdgeWeight["level"];
      demand_score: number;
      heat_level: "low" | "medium" | "high" | "critical";
    }>;
  };

  return {
    mapId: data.map_id,
    method: data.method,
    crowdInputs: {
      lobbyPeople: data.crowd_inputs.lobby_people,
      boothPeople: data.crowd_inputs.booth_people,
      recentInflow: data.crowd_inputs.recent_inflow,
      hour: data.crowd_inputs.hour,
      eventPhase: data.crowd_inputs.event_phase,
    },
    regions: data.regions.map((region) => ({
      region: region.region,
      peopleEstimate: region.people_estimate,
      qrScans: region.qr_scans,
      routeIntents: region.route_intents,
      demandScore: region.demand_score,
    })),
    edges: data.edges.map((edge) => ({
      edgeId: edge.edge_id,
      fromNode: edge.from_node,
      toNode: edge.to_node,
      zone: edge.zone,
      crowdRegion: edge.crowd_region,
      multiplier: edge.multiplier,
      level: edge.level,
      demandScore: edge.demand_score,
      heatLevel: edge.heat_level,
    })),
  };
}

export async function updateManualCrowd(
  inputs: Pick<CrowdInputs, "lobbyPeople" | "boothPeople">,
  mapId = "default"
): Promise<CrowdEstimate> {
  const response = await fetch(`${AI_API_URL}/telemetry/manual-crowd`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: mapId,
      lobby_people: clampNumber(inputs.lobbyPeople, 0, 1000),
      booth_people: clampNumber(inputs.boothPeople, 0, 1000),
    }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Manual crowd API 오류: ${response.status} ${errorText}`);
  }

  return normalizeCrowdEstimate(await response.json());
}

export async function recordQrScan(
  region: "lobby" | "booth",
  count = 1,
  mapId = "default"
): Promise<CrowdEstimate> {
  const response = await fetch(`${AI_API_URL}/telemetry/qr-scan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      map_id: mapId,
      checkpoint_id: region === "booth" ? "BOOTH_QR" : "LOBBY_QR",
      region,
      count,
    }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`QR telemetry API 오류: ${response.status} ${errorText}`);
  }

  return normalizeCrowdEstimate(await response.json());
}

export async function fetchVenueMap(mapId = "default"): Promise<VenueMap> {
  const response = await fetch(`${AI_API_URL}/maps/${mapId}`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    id: string;
    name: string;
    image: string;
    width: number;
    height: number;
    nodes: VenueMap["nodes"];
    edges: Parameters<typeof normalizeMapEdge>[0][];
    checkpoints?: Parameters<typeof normalizeMapCheckpoint>[0][];
    selectable_nodes: VenueMap["selectableNodes"];
  };

  return {
    id: data.id,
    name: data.name,
    image: data.image.startsWith("http")
      ? data.image
      : `${AI_API_URL}${data.image}`,
    width: data.width,
    height: data.height,
    nodes: data.nodes,
    edges: data.edges.map(normalizeMapEdge),
    checkpoints: (data.checkpoints ?? []).map(normalizeMapCheckpoint),
    selectableNodes: data.selectable_nodes,
  };
}

export async function fetchRouteRiskAnalysis(
  mapId = "default",
  maxPairs = 300,
  maxEdges = 100
): Promise<RouteRiskAnalysisResult> {
  const search = new URLSearchParams({
    max_pairs: String(maxPairs),
    max_edges: String(maxEdges),
  });
  const response = await fetch(
    `${AI_API_URL}/maps/${mapId}/route-risk-analysis?${search.toString()}`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Route risk analysis API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    selectable_node_count: number;
    analyzed_pair_count: number;
    analyzed_edge_count: number;
    critical_edge_count: number;
    high_risk_edge_count: number;
    edges: Array<{
      edge_id: string;
      from_node: string;
      to_node: string;
      zone: string;
      crowd_region: string;
      affected_pair_count: number;
      disconnected_pair_count: number;
      average_detour_ratio: number;
      max_detour_ratio: number;
      risk_score: number;
      risk_level: "low" | "medium" | "high" | "critical";
      sample_pairs: Array<Record<string, unknown>>;
    }>;
  };

  return {
    mapId: data.map_id,
    selectableNodeCount: data.selectable_node_count,
    analyzedPairCount: data.analyzed_pair_count,
    analyzedEdgeCount: data.analyzed_edge_count,
    criticalEdgeCount: data.critical_edge_count,
    highRiskEdgeCount: data.high_risk_edge_count,
    edges: data.edges.map((edge) => ({
      edgeId: edge.edge_id,
      fromNode: edge.from_node,
      toNode: edge.to_node,
      zone: edge.zone,
      crowdRegion: edge.crowd_region,
      affectedPairCount: edge.affected_pair_count,
      disconnectedPairCount: edge.disconnected_pair_count,
      averageDetourRatio: edge.average_detour_ratio,
      maxDetourRatio: edge.max_detour_ratio,
      riskScore: edge.risk_score,
      riskLevel: edge.risk_level,
      samplePairs: edge.sample_pairs,
    })),
  };
}

export async function fetchMapTopologyAnalysis(
  mapId = "default",
  longEdgeThreshold = 20,
  checkpointGapPx = 420
): Promise<MapTopologyAnalysisResult> {
  const search = new URLSearchParams({
    long_edge_threshold: String(longEdgeThreshold),
    checkpoint_gap_px: String(checkpointGapPx),
  });
  const response = await fetch(
    `${AI_API_URL}/maps/${mapId}/topology-analysis?${search.toString()}`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map topology analysis API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    score: number;
    readiness: "ready" | "needs_review" | "blocked";
    node_count: number;
    edge_count: number;
    checkpoint_count: number;
    dead_end_node_ids: string[];
    low_degree_junction_ids: string[];
    duplicate_edge_groups: string[][];
    long_edge_ids: string[];
    checkpoint_gap_node_ids: string[];
    issues: Array<{
      code: string;
      severity: "info" | "warning" | "error";
      target_ids: string[];
      message: string;
      metadata: Record<string, unknown>;
    }>;
  };

  return {
    mapId: data.map_id,
    score: data.score,
    readiness: data.readiness,
    nodeCount: data.node_count,
    edgeCount: data.edge_count,
    checkpointCount: data.checkpoint_count,
    deadEndNodeIds: data.dead_end_node_ids,
    lowDegreeJunctionIds: data.low_degree_junction_ids,
    duplicateEdgeGroups: data.duplicate_edge_groups,
    longEdgeIds: data.long_edge_ids,
    checkpointGapNodeIds: data.checkpoint_gap_node_ids,
    issues: data.issues.map((issue) => ({
      code: issue.code,
      severity: issue.severity,
      targetIds: issue.target_ids,
      message: issue.message,
      metadata: issue.metadata,
    })),
  };
}

export async function fetchMapImprovementSuggestions(
  mapId = "default"
): Promise<MapImprovementSuggestionsResult> {
  const response = await fetch(`${AI_API_URL}/maps/${mapId}/improvement-suggestions`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map improvement suggestions API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    readiness: "ready" | "needs_review" | "blocked";
    score: number;
    suggestion_count: number;
    critical_count: number;
    high_count: number;
    suggestions: Array<{
      code: string;
      priority: "low" | "medium" | "high" | "critical";
      source: "validation" | "postprocess" | "topology" | "risk";
      target_ids: string[];
      message: string;
      action: string;
      metadata: Record<string, unknown>;
    }>;
    validation_summary: Record<string, unknown>;
    topology_summary: Record<string, unknown>;
    risk_summary: Record<string, unknown>;
  };

  return {
    mapId: data.map_id,
    readiness: data.readiness,
    score: data.score,
    suggestionCount: data.suggestion_count,
    criticalCount: data.critical_count,
    highCount: data.high_count,
    suggestions: data.suggestions.map((suggestion) => ({
      code: suggestion.code,
      priority: suggestion.priority,
      source: suggestion.source,
      targetIds: suggestion.target_ids,
      message: suggestion.message,
      action: suggestion.action,
      metadata: suggestion.metadata,
    })),
    validationSummary: data.validation_summary,
    topologySummary: data.topology_summary,
    riskSummary: data.risk_summary,
  };
}

export async function fetchVenueMaps(): Promise<MapSummary[]> {
  const response = await fetch(`${AI_API_URL}/maps`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Maps API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    maps: Parameters<typeof normalizeMapSummary>[0][];
  };

  return data.maps.map(normalizeMapSummary);
}

export async function fetchMapSchema(): Promise<MapSchema> {
  const response = await fetch(`${AI_API_URL}/maps/schema`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map schema API 오류: ${response.status} ${errorText}`);
  }

  return normalizeMapSchema(await response.json());
}

export async function validateVenueMapData(
  venueMap: unknown
): Promise<MapValidationResult> {
  const response = await fetch(`${AI_API_URL}/maps/validate-data`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(venueMap),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map validation API 오류: ${response.status} ${errorText}`);
  }

  return normalizeMapValidation(await response.json());
}

export async function saveVenueMapData(
  venueMap: unknown,
  overwrite = false
): Promise<MapSaveResult> {
  const response = await fetch(`${AI_API_URL}/maps`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      venue_map: venueMap,
      overwrite,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map save API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    saved: boolean;
    overwritten: boolean;
    path: string;
    validation: Parameters<typeof normalizeMapValidation>[0];
  };

  return {
    mapId: data.map_id,
    saved: data.saved,
    overwritten: data.overwritten,
    path: data.path,
    validation: normalizeMapValidation(data.validation),
  };
}

export async function deleteVenueMap(mapId: string): Promise<MapDeleteResult> {
  const response = await fetch(`${AI_API_URL}/maps/${mapId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map delete API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    map_id: string;
    deleted: boolean;
    path: string;
  };

  return {
    mapId: data.map_id,
    deleted: data.deleted,
    path: data.path,
  };
}

export async function createMapGenerationJob(
  job: MapGenerationJobCreate
): Promise<MapGenerationJob> {
  const response = await fetch(`${AI_API_URL}/map-generation/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_image_base64: job.sourceImageBase64,
      filename: job.filename,
      mime_type: job.mimeType,
      target_map_id: job.targetMapId,
      notes: job.notes,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation API 오류: ${response.status} ${errorText}`);
  }

  return normalizeMapGenerationJob(await response.json());
}

export async function fetchMapGenerationJob(
  jobId: string
): Promise<MapGenerationJob> {
  const response = await fetch(`${AI_API_URL}/map-generation/jobs/${jobId}`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation API 오류: ${response.status} ${errorText}`);
  }

  return normalizeMapGenerationJob(await response.json());
}

export async function fetchMapGenerationJobs(limit = 20): Promise<MapGenerationJob[]> {
  const response = await fetch(
    `${AI_API_URL}/map-generation/jobs?limit=${encodeURIComponent(limit)}`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation jobs API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    jobs: Parameters<typeof normalizeMapGenerationJob>[0][];
  };

  return data.jobs.map(normalizeMapGenerationJob);
}

export async function attachMapGenerationDraft(
  jobId: string,
  venueMap: unknown
): Promise<MapGenerationJob> {
  const response = await fetch(
    `${AI_API_URL}/map-generation/jobs/${jobId}/draft-map`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        venue_map: venueMap,
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation draft API 오류: ${response.status} ${errorText}`);
  }

  return normalizeMapGenerationJob(await response.json());
}

export async function fetchMapGenerationDraft(
  jobId: string
): Promise<MapGenerationDraftReview> {
  const response = await fetch(
    `${AI_API_URL}/map-generation/jobs/${jobId}/draft-map`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation draft API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    job: Parameters<typeof normalizeMapGenerationJob>[0];
    draft_map: unknown;
    validation: Parameters<typeof normalizeMapValidation>[0];
  };

  return {
    job: normalizeMapGenerationJob(data.job),
    draftMap: data.draft_map,
    validation: normalizeMapValidation(data.validation),
  };
}

export async function postprocessMapGenerationDraft(
  jobId: string,
  options: MapGenerationPostprocessOptions = {}
): Promise<MapGenerationPostprocessResult> {
  const response = await fetch(
    `${AI_API_URL}/map-generation/jobs/${jobId}/postprocess-draft`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        apply: options.apply ?? true,
        recalculate_edge_distance: options.recalculateEdgeDistance ?? true,
        clamp_coordinates: options.clampCoordinates ?? true,
        close_node_threshold_px: options.closeNodeThresholdPx ?? 18,
        suggest_connection_edges: options.suggestConnectionEdges ?? true,
        apply_connection_suggestions: options.applyConnectionSuggestions ?? false,
        max_connection_suggestions: options.maxConnectionSuggestions ?? 5,
        max_connection_distance_px: options.maxConnectionDistancePx ?? 260,
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation postprocess API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    job: Parameters<typeof normalizeMapGenerationJob>[0];
    processed_map: unknown;
    changes: Array<Record<string, unknown>>;
    suggestions: Array<Record<string, unknown>>;
    validation: Parameters<typeof normalizeMapValidation>[0];
    applied: boolean;
  };

  return {
    job: normalizeMapGenerationJob(data.job),
    processedMap: data.processed_map,
    changes: data.changes,
    suggestions: data.suggestions,
    validation: normalizeMapValidation(data.validation),
    applied: data.applied,
  };
}

export async function previewMapGenerationRoute(
  jobId: string,
  preview: MapGenerationRoutePreviewRequest
): Promise<RouteApiResult> {
  const response = await fetch(
    `${AI_API_URL}/map-generation/jobs/${jobId}/route-preview`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        start_id: preview.startId,
        current_node_id: preview.currentNodeId,
        checkpoint_id: preview.checkpointId,
        destination_id: preview.destinationId,
        crowd_inputs: crowdInputsPayload(preview.crowdInputs),
        walking_speed_mps: clampNumber(preview.options.walkingSpeedMps, 0.1, 3),
        use_congestion: preview.options.useCongestion,
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation route preview API 오류: ${response.status} ${errorText}`);
  }

  return normalizeRouteResponse(await response.json());
}

export async function generateMapGenerationDraft(
  jobId: string,
  options: MapGenerationDraftGenerateOptions = {}
): Promise<MapGenerationDraftGenerateResult> {
  const response = await fetch(
    `${AI_API_URL}/map-generation/jobs/${jobId}/generate-draft`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: options.model,
        image_detail: options.imageDetail,
        extra_instructions: options.extraInstructions,
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation LLM API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    job: Parameters<typeof normalizeMapGenerationJob>[0];
    draft_map: unknown;
  };

  return {
    job: normalizeMapGenerationJob(data.job),
    draftMap: data.draft_map,
  };
}

export async function saveMapGenerationDraft(
  jobId: string,
  overwrite = false
): Promise<MapGenerationSaveResult> {
  const response = await fetch(
    `${AI_API_URL}/map-generation/jobs/${jobId}/save-map`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        overwrite,
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map generation save API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    job: Parameters<typeof normalizeMapGenerationJob>[0];
    save_result: {
      map_id: string;
      saved: boolean;
      overwritten: boolean;
      path: string;
      validation: Parameters<typeof normalizeMapValidation>[0];
    };
  };

  return {
    job: normalizeMapGenerationJob(data.job),
    saveResult: {
      mapId: data.save_result.map_id,
      saved: data.save_result.saved,
      overwritten: data.save_result.overwritten,
      path: data.save_result.path,
      validation: normalizeMapValidation(data.save_result.validation),
    },
  };
}

export async function fetchMapCheckpoints(
  mapId = "default"
): Promise<MapCheckpoint[]> {
  const response = await fetch(`${AI_API_URL}/maps/${mapId}/checkpoints`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Map checkpoints API 오류: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    checkpoints: Parameters<typeof normalizeMapCheckpoint>[0][];
  };

  return data.checkpoints.map(normalizeMapCheckpoint);
}
