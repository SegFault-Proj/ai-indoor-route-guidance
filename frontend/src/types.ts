export type NodeType =
  | "entrance"
  | "exit"
  | "junction"
  | "facility"
  | "booth";

export interface MapNode {
  id: string;
  name: string;
  x: number;
  y: number;
  type: NodeType;
  selectable: boolean;
}

export type ZoneType = "lobby" | "booth" | "gate" | "facility";
export type CrowdRegion = "west" | "central" | "north" | "booth";

export interface MapEdge {
  id: string;
  from: string;
  to: string;
  distance: number;
  widthM: number;
  zone: ZoneType;
  crowdRegion: CrowdRegion;
  bidirectional: boolean;
}

export interface MapCheckpoint {
  id: string;
  name: string;
  nodeId: string;
  region: "lobby" | "booth";
}

export interface VenueMap {
  id: string;
  name: string;
  image: string;
  width: number;
  height: number;
  nodes: MapNode[];
  edges: MapEdge[];
  checkpoints: MapCheckpoint[];
  selectableNodes: MapNode[];
}

export interface MapSummary {
  id: string;
  name: string;
  image: string;
  width: number;
  height: number;
  nodeCount: number;
  edgeCount: number;
  checkpointCount: number;
  selectableNodeCount: number;
  valid: boolean;
}

export interface MapValidationResult {
  mapId: string | null;
  valid: boolean;
  errors: string[];
  warnings: string[];
  nodeCount: number;
  edgeCount: number;
  checkpointCount: number;
  selectableNodeCount: number;
  unreachableRoutePairs: Array<{
    startId: string;
    destinationId: string;
  }>;
  outOfBoundsNodeIds: string[];
  invalidNodeIds: string[];
  invalidEdgeIds: string[];
  invalidCheckpointIds: string[];
  fixSuggestions: MapFixSuggestion[];
  qualitySummary: MapQualitySummary;
}

export interface MapFixSuggestion {
  code: string;
  severity: "error" | "warning";
  targetIds: string[];
  message: string;
  allowedValues?: Record<string, string[]>;
  samplePairs?: Array<{
    startId: string;
    destinationId: string;
  }>;
}

export interface MapQualitySummary {
  score: number;
  readiness: "ready" | "needs_review" | "blocked";
  canSave: boolean;
  reasonCodes: string[];
}

export interface MapSchema {
  requiredFields: string[];
  nodeFields: string[];
  edgeFields: string[];
  checkpointFields: string[];
  nodeTypes: NodeType[];
  zones: ZoneType[];
  crowdRegions: CrowdRegion[];
  checkpointRegions: Array<"lobby" | "booth">;
  coordinateSystem: string;
  notes: string[];
}

export interface MapSaveResult {
  mapId: string;
  saved: boolean;
  overwritten: boolean;
  path: string;
  validation: MapValidationResult;
}

export interface MapDeleteResult {
  mapId: string;
  deleted: boolean;
  path: string;
}

export type MapGenerationJobStatus =
  | "needs_llm"
  | "draft_valid"
  | "draft_invalid"
  | "map_saved";

export interface MapGenerationJobCreate {
  sourceImageBase64: string;
  filename: string;
  mimeType: "image/png" | "image/jpeg";
  targetMapId: string;
  notes?: string;
}

export interface MapGenerationDraftGenerateOptions {
  model?: string;
  imageDetail?: "low" | "high" | "auto";
  extraInstructions?: string;
}

export interface MapGenerationJob {
  jobId: string;
  status: MapGenerationJobStatus;
  targetMapId: string;
  sourceFilename: string;
  sourceMimeType: string;
  sourceImagePath: string;
  sourceImageUrl: string;
  notes: string | null;
  createdAt: string;
  updatedAt: string;
  validation: MapValidationResult | null;
  savedMapId: string | null;
  nextStep: string;
}

export interface MapGenerationSaveResult {
  job: MapGenerationJob;
  saveResult: MapSaveResult;
}

export interface MapGenerationDraftGenerateResult {
  job: MapGenerationJob;
  draftMap: unknown;
}

export interface MapGenerationDraftReview {
  job: MapGenerationJob;
  draftMap: unknown;
  validation: MapValidationResult;
}

export interface MapGenerationRoutePreviewRequest {
  startId?: string;
  currentNodeId?: string;
  checkpointId?: string;
  destinationId: string;
  crowdInputs?: CrowdInputs;
  options: Pick<RouteOptions, "useCongestion" | "walkingSpeedMps">;
}

export interface MapGenerationPostprocessOptions {
  apply?: boolean;
  recalculateEdgeDistance?: boolean;
  clampCoordinates?: boolean;
  closeNodeThresholdPx?: number;
  suggestConnectionEdges?: boolean;
  applyConnectionSuggestions?: boolean;
  maxConnectionSuggestions?: number;
  maxConnectionDistancePx?: number;
}

export interface MapGenerationPostprocessResult {
  job: MapGenerationJob;
  processedMap: unknown;
  changes: Array<Record<string, unknown>>;
  suggestions: Array<Record<string, unknown>>;
  validation: MapValidationResult;
  applied: boolean;
}

export interface ApiEndpoint {
  method: "GET" | "POST" | "DELETE";
  path: string;
}

export interface AppConfig {
  apiVersion: string;
  defaultMapId: string;
  defaultWalkingSpeedMps: number;
  maxWalkingSpeedMps: number;
  telemetryWindowMinutes: number;
  limits: {
    peopleCount: number;
    recentInflow: number;
    qrScanCount: number;
    telemetrySnapshotLimit: number;
  };
  eventPhaseLabels: Record<number, string>;
  features: {
    routeApi: boolean;
    congestionPrediction: boolean;
    estimatedCrowdWithoutCamera: boolean;
    manualCrowdInput: boolean;
    qrScanTelemetry: boolean;
    routeIntentTelemetry: boolean;
    checkpointPositionUpdate: boolean;
    coordinatePositionSnap: boolean;
    llmMapValidation: boolean;
    llmMapStorage: boolean;
    llmMapGenerationJobs: boolean;
    llmMapGenerateDraft: boolean;
    routePreferences: boolean;
    temporaryBlockedEdges: boolean;
    routeAlternatives: boolean;
    routeRecommendation: boolean;
    multiStopRoute: boolean;
    crowdForecast: boolean;
    bottleneckDetection: boolean;
    etaCalibration: boolean;
    routeRiskAnalysis: boolean;
    mapTopologyAnalysis: boolean;
    routeSimulation: boolean;
    demandHeatmap: boolean;
    navigationGuidance: boolean;
    mapImprovementSuggestions: boolean;
  };
  endpoints: ApiEndpoint[];
}

export interface PredictedEdgeWeight {
  edgeId: string;
  multiplier: number;
  level: "clear" | "normal" | "busy" | "very_busy";
  source: "ai" | "fallback";
}

export interface RoutePoint {
  nodeId: string;
  name: string;
  x: number;
  y: number;
}

export interface RouteSegment {
  edgeId: string;
  fromNode: RoutePoint;
  toNode: RoutePoint;
  distance: number;
  multiplier: number;
  weightedCost: number;
  estimatedSeconds: number;
  level: PredictedEdgeWeight["level"];
  maneuver:
    | "start"
    | "straight"
    | "slight_left"
    | "left"
    | "slight_right"
    | "right"
    | "u_turn"
    | "arrive";
  headingDegrees: number;
  turnDegrees: number;
  instruction: string;
}

export interface RouteResult {
  path: string[];
  edgeIds: string[];
  totalDistance: number;
  weightedCost: number;
}

export interface RouteApiResult {
  route: RouteResult;
  routePoints: RoutePoint[];
  segments: RouteSegment[];
  predictions: PredictedEdgeWeight[];
  instructions: string[];
  estimatedSeconds: number;
  crowdInputs: CrowdInputs;
  preference?: "shortest" | "less_crowded" | "accessible" | "fewest_turns";
  blockedEdgeIds?: string[];
}

export interface RouteAlternative {
  rank: number;
  route: RouteResult;
  routePoints: RoutePoint[];
  segments: RouteSegment[];
  instructions: string[];
  estimatedSeconds: number;
  overlapWithBestEdgeCount: number;
}

export interface RouteAlternativesResult {
  mapId: string;
  startId: string;
  destinationId: string;
  useCongestion: boolean;
  walkingSpeedMps: number;
  preference?: "shortest" | "less_crowded" | "accessible" | "fewest_turns";
  blockedEdgeIds?: string[];
  crowdInputs: CrowdInputs;
  alternatives: RouteAlternative[];
  predictions: PredictedEdgeWeight[];
}

export interface RouteRecommendationResult {
  mapId: string;
  startId: string;
  destinationId: string;
  recommendation: {
    selectedRank: number;
    preference: NonNullable<RouteApiResult["preference"]>;
    reasons: string[];
    tradeoffs: Record<string, unknown>;
  };
  selected: RouteAlternative;
  alternatives: RouteAlternative[];
  predictions: PredictedEdgeWeight[];
}

export interface RouteRiskAnalysisResult {
  mapId: string;
  selectableNodeCount: number;
  analyzedPairCount: number;
  analyzedEdgeCount: number;
  criticalEdgeCount: number;
  highRiskEdgeCount: number;
  edges: Array<{
    edgeId: string;
    fromNode: string;
    toNode: string;
    zone: string;
    crowdRegion: string;
    affectedPairCount: number;
    disconnectedPairCount: number;
    averageDetourRatio: number;
    maxDetourRatio: number;
    riskScore: number;
    riskLevel: "low" | "medium" | "high" | "critical";
    samplePairs: Array<Record<string, unknown>>;
  }>;
}

export interface MapTopologyAnalysisResult {
  mapId: string;
  score: number;
  readiness: "ready" | "needs_review" | "blocked";
  nodeCount: number;
  edgeCount: number;
  checkpointCount: number;
  deadEndNodeIds: string[];
  lowDegreeJunctionIds: string[];
  duplicateEdgeGroups: string[][];
  longEdgeIds: string[];
  checkpointGapNodeIds: string[];
  issues: Array<{
    code: string;
    severity: "info" | "warning" | "error";
    targetIds: string[];
    message: string;
    metadata: Record<string, unknown>;
  }>;
}

export interface MapImprovementSuggestionsResult {
  mapId: string;
  readiness: "ready" | "needs_review" | "blocked";
  score: number;
  suggestionCount: number;
  criticalCount: number;
  highCount: number;
  suggestions: Array<{
    code: string;
    priority: "low" | "medium" | "high" | "critical";
    source: "validation" | "postprocess" | "topology" | "risk";
    targetIds: string[];
    message: string;
    action: string;
    metadata: Record<string, unknown>;
  }>;
  validationSummary: Record<string, unknown>;
  topologySummary: Record<string, unknown>;
  riskSummary: Record<string, unknown>;
}

export interface RouteSimulationScenario {
  id: string;
  startId: string;
  destinationId: string;
  crowdInputs?: CrowdInputs;
  walkingSpeedMps?: number;
  useCongestion?: boolean;
  preference?: NonNullable<RouteApiResult["preference"]>;
  blockedEdgeIds?: string[];
}

export interface RouteSimulationResult {
  mapId: string;
  scenarioCount: number;
  successCount: number;
  failureCount: number;
  passRate: number;
  failures: string[];
  results: Array<{
    scenarioId: string;
    ok: boolean;
    startId: string;
    destinationId: string;
    preference: NonNullable<RouteApiResult["preference"]>;
    blockedEdgeIds: string[];
    path: string[];
    edgeIds: string[];
    totalDistance: number | null;
    weightedCost: number | null;
    estimatedSeconds: number | null;
    error: Record<string, unknown> | null;
  }>;
}

export interface DemandHeatmapResult {
  mapId: string;
  method: string;
  crowdInputs: CrowdInputs;
  regions: Array<{
    region: string;
    peopleEstimate: number;
    qrScans: number;
    routeIntents: number;
    demandScore: number;
  }>;
  edges: Array<{
    edgeId: string;
    fromNode: string;
    toNode: string;
    zone: string;
    crowdRegion: string;
    multiplier: number;
    level: PredictedEdgeWeight["level"];
    demandScore: number;
    heatLevel: "low" | "medium" | "high" | "critical";
  }>;
}

export interface NavigationPositionUpdate {
  mapId?: string;
  currentNodeId?: string;
  destinationId: string;
  checkpointId?: string;
  region?: "lobby" | "booth";
  crowdInputs?: CrowdInputs;
  options: RouteOptions;
}

export interface NavigationUpdateResult {
  mapId: string;
  currentNode: RoutePoint;
  destinationNode: RoutePoint;
  locationSource: "node" | "checkpoint";
  route: RouteApiResult;
}

export interface CoordinateSnapResult {
  mapId: string;
  inputX: number;
  inputY: number;
  snappedNode: RoutePoint;
  distancePx: number;
  withinThreshold: boolean;
  selectable: boolean;
  snappedEdgeId: string | null;
  snappedEdgeFromNodeId: string | null;
  snappedEdgeToNodeId: string | null;
  edgeDistancePx: number | null;
  edgeProgressRatio: number | null;
  projectedX: number | null;
  projectedY: number | null;
}

export interface NavigationSessionStart {
  mapId?: string;
  startId: string;
  destinationId: string;
  crowdInputs?: CrowdInputs;
  options: RouteOptions;
}

export interface NavigationSessionPositionUpdate {
  currentNodeId?: string;
  checkpointId?: string;
  region?: "lobby" | "booth";
  crowdInputs?: CrowdInputs;
  options: RouteOptions;
}

export interface NavigationSessionCoordinateUpdate {
  x: number;
  y: number;
  maxDistancePx?: number;
  region?: "lobby" | "booth";
  crowdInputs?: CrowdInputs;
  options: RouteOptions;
}

export interface NavigationUpdateEvent {
  nodeId: string;
  source: string;
  checkpointId: string | null;
  timestamp: string;
}

export interface NavigationSession {
  sessionId: string;
  mapId: string;
  startNode: RoutePoint;
  currentNode: RoutePoint;
  destinationNode: RoutePoint;
  status: "active" | "arrived";
  createdAt: string;
  updatedAt: string;
  route: RouteApiResult;
  recentUpdates: NavigationUpdateEvent[];
}

export interface NavigationGuidanceResult {
  sessionId: string;
  mapId: string;
  status: "active" | "arrived";
  currentNode: RoutePoint;
  destinationNode: RoutePoint;
  offRoute: boolean;
  expectedPath: string[];
  expectedEdgeIds: string[];
  progressRatio: number;
  remainingDistance: number;
  remainingSeconds: number;
  nextNode: RoutePoint | null;
  nextSegment: RouteSegment | null;
  instruction: string;
  route: RouteApiResult;
}

export interface EtaCalibrationResult {
  sessionId: string;
  mapId: string;
  sampleCount: number;
  confidence: number;
  defaultWalkingSpeedMps: number;
  observedWalkingSpeedMps: number | null;
  recommendedWalkingSpeedMps: number;
  samples: Array<{
    fromNodeId: string;
    toNodeId: string;
    distance: number;
    elapsedSeconds: number;
    observedSpeedMps: number;
  }>;
}

export interface RouteOptions {
  useCongestion: boolean;
  useEstimatedCrowd: boolean;
  walkingSpeedMps: number;
  preference?: "shortest" | "less_crowded" | "accessible" | "fewest_turns";
  blockedEdgeIds?: string[];
}

export interface CrowdEstimate {
  mapId: string;
  lobbyPeople: number;
  boothPeople: number;
  recentInflow: number;
  hour: number;
  eventPhase: number;
  signals: {
    windowMinutes: number;
    qrScans: {
      lobby: number;
      booth: number;
    };
    routeIntents: {
      lobby: number;
      booth: number;
    };
    manualEstimate: {
      lobby_people: number;
      booth_people: number;
    };
  };
}

export interface CrowdInputs {
  lobbyPeople: number;
  boothPeople: number;
  recentInflow: number;
  hour: number;
  eventPhase: 0 | 1 | 2 | 3;
}
