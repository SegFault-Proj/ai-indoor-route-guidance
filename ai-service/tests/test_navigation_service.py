import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi import HTTPException

from crowd_store import reset_crowd_store
from generation_store import reset_generation_store
from main import (
    CoordinateSnapRequest,
    CrowdInputs,
    BlockedEdgesRequest,
    ManualCrowdRequest,
    MapGenerationDraftRequest,
    MapGenerationGenerateDraftRequest,
    MapGenerationJobCreateRequest,
    MapGenerationPostprocessRequest,
    MapGenerationRoutePreviewRequest,
    MapGenerationSaveRequest,
    MapSaveRequest,
    NavigationSessionStartRequest,
    NavigationSessionCoordinateUpdateRequest,
    NavigationSessionUpdateRequest,
    PositionUpdateRequest,
    QrScanRequest,
    RouteAlternativesRequest,
    RouteMultiStopRequest,
    RouteRequest,
    RouteSimulationRequest,
    RouteSimulationScenario,
    TEMPORARY_BLOCKED_EDGES,
    create_qr_scan,
    create_map_generation_job,
    delete_map,
    attach_map_generation_draft,
    get_crowd_estimate,
    get_crowd_bottlenecks,
    get_crowd_forecast,
    get_demand_heatmap,
    get_app_config,
    get_blocked_edges,
    get_map,
    get_map_checkpoints,
    get_map_improvement_suggestions,
    get_map_generation_draft,
    get_map_schema,
    get_map_topology_analysis,
    get_map_validation,
    get_maps,
    get_map_generation_job,
    get_map_generation_jobs,
    get_navigation_session_status,
    get_navigation_eta_calibration,
    get_navigation_guidance,
    get_route_risk_analysis,
    get_telemetry,
    generate_map_generation_draft,
    preview_map_generation_route,
    postprocess_map_generation_draft,
    route,
    route_alternatives,
    route_recommendation,
    route_multi_stop,
    route_simulation,
    save_map_generation_draft,
    save_map_payload,
    snap_coordinate,
    start_navigation_session,
    update_manual_crowd,
    update_blocked_edges,
    update_navigation_session_coordinate_api,
    update_navigation_session_position_api,
    update_position,
    validate_map_payload,
)
from map_store import map_file_path, validate_map_data
from map_postprocess import postprocess_map_data

TEST_MAP_ID = "generated_test_map"
TEST_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8"
    "AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class NavigationServiceTest(unittest.TestCase):
    def setUp(self):
        reset_crowd_store()
        reset_generation_store()
        TEMPORARY_BLOCKED_EDGES.clear()
        self.cleanup_test_map()

    def tearDown(self):
        reset_generation_store()
        TEMPORARY_BLOCKED_EDGES.clear()
        self.cleanup_test_map()

    def cleanup_test_map(self):
        map_file_path(TEST_MAP_ID).unlink(missing_ok=True)

    def test_maps_summary_includes_default_map(self):
        response = get_maps()
        default_map = next(
            venue_map for venue_map in response["maps"] if venue_map["id"] == "default"
        )

        self.assertTrue(default_map["valid"])
        self.assertEqual(default_map["node_count"], 25)
        self.assertEqual(default_map["edge_count"], 25)
        self.assertEqual(default_map["checkpoint_count"], 5)
        self.assertEqual(default_map["selectable_node_count"], 16)
        self.assertEqual(default_map["image"], "/static/floorplan.png")

    def test_app_config_exposes_app_integration_defaults(self):
        response = get_app_config()

        self.assertEqual(response.default_map_id, "default")
        self.assertEqual(response.default_walking_speed_mps, 1.2)
        self.assertEqual(response.max_walking_speed_mps, 3.0)
        self.assertEqual(response.telemetry_window_minutes, 10)
        self.assertTrue(response.features["route_api"])
        self.assertTrue(response.features["estimated_crowd_without_camera"])
        self.assertTrue(response.features["llm_map_validation"])
        self.assertTrue(response.features["llm_map_generation_jobs"])
        self.assertTrue(response.features["route_preferences"])
        self.assertTrue(response.features["astar_route_search"])
        self.assertTrue(response.features["temporary_blocked_edges"])
        self.assertTrue(response.features["route_recommendation"])
        self.assertTrue(response.features["multi_stop_route"])
        self.assertTrue(response.features["crowd_forecast"])
        self.assertTrue(response.features["bottleneck_detection"])
        self.assertTrue(response.features["eta_calibration"])
        self.assertTrue(response.features["route_risk_analysis"])
        self.assertTrue(response.features["map_topology_analysis"])
        self.assertTrue(response.features["route_simulation"])
        self.assertTrue(response.features["demand_heatmap"])
        self.assertTrue(response.features["navigation_guidance"])
        self.assertTrue(response.features["map_improvement_suggestions"])
        self.assertEqual(response.limits["people_count"], 1000)
        self.assertIn(
            "/route",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/navigation/update-position",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/navigation/sessions",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/route-multi-stop",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/route-recommendation",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/route-simulation",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/crowd/{map_id}/forecast",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/crowd/{map_id}/demand-heatmap",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/maps/{map_id}/route-risk-analysis",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/maps/{map_id}/topology-analysis",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/maps/{map_id}/improvement-suggestions",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/navigation/sessions/{session_id}/eta-calibration",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/navigation/sessions/{session_id}/guidance",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/maps/{map_id}/checkpoints",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/maps/validate-data",
            [endpoint.path for endpoint in response.endpoints],
        )
        self.assertIn(
            "/maps",
            [
                endpoint.path
                for endpoint in response.endpoints
                if endpoint.method == "POST"
            ],
        )
        self.assertIn(
            "/maps/{map_id}",
            [
                endpoint.path
                for endpoint in response.endpoints
                if endpoint.method == "DELETE"
            ],
        )
        self.assertIn(
            "/map-generation/jobs",
            [
                endpoint.path
                for endpoint in response.endpoints
                if endpoint.method == "POST"
            ],
        )

    def test_map_schema_exposes_llm_generation_contract(self):
        response = get_map_schema()

        self.assertIn("nodes", response.required_fields)
        self.assertIn("edges", response.required_fields)
        self.assertIn("node_id", response.checkpoint_fields)
        self.assertIn("booth", response.node_types)
        self.assertIn("booth", response.checkpoint_regions)

    def test_map_detail_includes_selectable_nodes(self):
        venue_map = get_map("default")

        self.assertEqual(venue_map["id"], "default")
        self.assertEqual(venue_map["width"], 1672)
        self.assertEqual(venue_map["height"], 941)
        self.assertIn("selectable_nodes", venue_map)
        self.assertIn("checkpoints", venue_map)
        self.assertTrue(
            all(node["selectable"] for node in venue_map["selectable_nodes"])
        )

    def test_map_checkpoints_are_exposed(self):
        response = get_map_checkpoints("default")
        checkpoint = next(
            checkpoint
            for checkpoint in response["checkpoints"]
            if checkpoint["id"] == "BOOTH_GATE_QR"
        )

        self.assertEqual(response["map_id"], "default")
        self.assertEqual(checkpoint["node_id"], "BOOTH_GATE")
        self.assertEqual(checkpoint["region"], "booth")

    def test_route_risk_analysis_scores_edges_by_detour_impact(self):
        response = get_route_risk_analysis("default", max_pairs=80, max_edges=25)

        self.assertEqual(response.map_id, "default")
        self.assertEqual(response.selectable_node_count, 16)
        self.assertEqual(response.analyzed_edge_count, 25)
        self.assertGreater(response.analyzed_pair_count, 0)
        self.assertEqual(
            [item.risk_score for item in response.edges],
            sorted([item.risk_score for item in response.edges], reverse=True),
        )
        self.assertGreater(response.edges[0].affected_pair_count, 0)
        self.assertIn(
            response.edges[0].risk_level,
            {"low", "medium", "high", "critical"},
        )

    def test_route_risk_analysis_rejects_invalid_limits(self):
        with self.assertRaises(HTTPException) as context:
            get_route_risk_analysis("default", max_pairs=0)

        self.assertEqual(context.exception.status_code, 400)

    def test_map_topology_analysis_reports_structural_quality(self):
        response = get_map_topology_analysis("default")

        self.assertEqual(response.map_id, "default")
        self.assertEqual(response.node_count, 25)
        self.assertEqual(response.edge_count, 25)
        self.assertEqual(response.checkpoint_count, 5)
        self.assertLess(response.score, 100)
        self.assertIn(response.readiness, {"ready", "needs_review", "blocked"})
        self.assertIn("BOOTH_10", response.dead_end_node_ids)
        self.assertTrue(
            any(issue.code == "selectable_dead_ends" for issue in response.issues)
        )

    def test_map_topology_analysis_rejects_invalid_threshold(self):
        with self.assertRaises(HTTPException) as context:
            get_map_topology_analysis("default", long_edge_threshold=0)

        self.assertEqual(context.exception.status_code, 400)

    def test_map_improvement_suggestions_prioritize_analysis_results(self):
        response = get_map_improvement_suggestions("default")

        self.assertEqual(response.map_id, "default")
        self.assertGreater(response.suggestion_count, 0)
        self.assertEqual(response.suggestion_count, len(response.suggestions))
        self.assertIn(response.readiness, {"ready", "needs_review", "blocked"})
        self.assertIn(
            "valid",
            response.validation_summary,
        )
        self.assertIn(
            "issue_count",
            response.topology_summary,
        )
        self.assertTrue(
            any(suggestion.source == "topology" for suggestion in response.suggestions)
        )
        priorities = [suggestion.priority for suggestion in response.suggestions]
        self.assertEqual(
            priorities,
            sorted(
                priorities,
                key=lambda value: {
                    "critical": 0,
                    "high": 1,
                    "medium": 2,
                    "low": 3,
                }[value],
            ),
        )

    def test_default_map_validation_passes(self):
        validation = get_map_validation("default")

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["errors"], [])
        self.assertEqual(validation["warnings"], [])
        self.assertEqual(validation["unreachable_route_pairs"], [])
        self.assertEqual(validation["quality_summary"]["score"], 100)
        self.assertEqual(validation["quality_summary"]["readiness"], "ready")
        self.assertTrue(validation["quality_summary"]["can_save"])

    def test_validate_map_payload_accepts_generated_map_data(self):
        venue_map = get_map("default")
        validation = validate_map_payload(venue_map)

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["map_id"], "default")
        self.assertEqual(validation["checkpoint_count"], 5)
        self.assertEqual(validation["unreachable_route_pairs"], [])

    def test_validate_map_payload_reports_generated_map_errors(self):
        venue_map = get_map("default")
        broken_map = deepcopy(venue_map)
        broken_map["edges"][0]["from"] = "MISSING_NODE"

        validation = validate_map_payload(broken_map)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("MISSING_NODE" in error for error in validation["errors"])
        )

    def test_validate_map_payload_reports_missing_required_fields(self):
        validation = validate_map_payload({"id": "generated"})

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["map_id"], "generated")
        self.assertEqual(validation["node_count"], 0)
        self.assertTrue(
            any("Missing required map field: nodes" in error for error in validation["errors"])
        )

    def test_save_map_payload_persists_valid_generated_map(self):
        generated_map = deepcopy(get_map("default"))
        generated_map["id"] = TEST_MAP_ID
        generated_map["name"] = "생성 지도 테스트"

        response = save_map_payload(MapSaveRequest(venue_map=generated_map))
        loaded_map = get_map(TEST_MAP_ID)

        self.assertTrue(response.saved)
        self.assertFalse(response.overwritten)
        self.assertEqual(response.map_id, TEST_MAP_ID)
        self.assertEqual(loaded_map["id"], TEST_MAP_ID)
        self.assertEqual(loaded_map["name"], "생성 지도 테스트")

    def test_saved_generated_map_appears_in_map_list(self):
        generated_map = deepcopy(get_map("default"))
        generated_map["id"] = TEST_MAP_ID
        generated_map["name"] = "생성 지도 테스트"
        save_map_payload(MapSaveRequest(venue_map=generated_map))

        response = get_maps()
        saved_summary = next(
            venue_map for venue_map in response["maps"] if venue_map["id"] == TEST_MAP_ID
        )

        self.assertTrue(saved_summary["valid"])
        self.assertEqual(saved_summary["name"], "생성 지도 테스트")
        self.assertEqual(saved_summary["checkpoint_count"], 5)

    def test_save_map_payload_rejects_existing_map_without_overwrite(self):
        generated_map = deepcopy(get_map("default"))
        generated_map["id"] = TEST_MAP_ID
        save_map_payload(MapSaveRequest(venue_map=generated_map))

        with self.assertRaises(HTTPException) as context:
            save_map_payload(MapSaveRequest(venue_map=generated_map))

        self.assertEqual(context.exception.status_code, 409)

    def test_save_map_payload_allows_explicit_overwrite(self):
        generated_map = deepcopy(get_map("default"))
        generated_map["id"] = TEST_MAP_ID
        save_map_payload(MapSaveRequest(venue_map=generated_map))

        generated_map["name"] = "덮어쓴 생성 지도"
        response = save_map_payload(
            MapSaveRequest(venue_map=generated_map, overwrite=True)
        )

        self.assertTrue(response.overwritten)
        self.assertEqual(get_map(TEST_MAP_ID)["name"], "덮어쓴 생성 지도")

    def test_save_map_payload_rejects_protected_default_map(self):
        with self.assertRaises(HTTPException) as context:
            save_map_payload(MapSaveRequest(venue_map=get_map("default")))

        self.assertEqual(context.exception.status_code, 400)

    def test_save_map_payload_rejects_invalid_map_id(self):
        generated_map = deepcopy(get_map("default"))
        generated_map["id"] = "../bad"

        with self.assertRaises(HTTPException) as context:
            save_map_payload(MapSaveRequest(venue_map=generated_map))

        self.assertEqual(context.exception.status_code, 400)

    def test_delete_map_removes_generated_map(self):
        generated_map = deepcopy(get_map("default"))
        generated_map["id"] = TEST_MAP_ID
        save_map_payload(MapSaveRequest(venue_map=generated_map))

        response = delete_map(TEST_MAP_ID)

        self.assertTrue(response.deleted)
        self.assertEqual(response.map_id, TEST_MAP_ID)
        self.assertFalse(map_file_path(TEST_MAP_ID).exists())

    def test_delete_map_rejects_protected_default_map(self):
        with self.assertRaises(HTTPException) as context:
            delete_map("default")

        self.assertEqual(context.exception.status_code, 400)

    def test_delete_map_rejects_missing_map(self):
        with self.assertRaises(HTTPException) as context:
            delete_map(TEST_MAP_ID)

        self.assertEqual(context.exception.status_code, 404)

    def test_map_generation_job_accepts_source_image(self):
        response = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
                notes="테스트 도면",
            )
        )
        loaded = get_map_generation_job(response.job_id)

        self.assertEqual(response.status, "needs_llm")
        self.assertEqual(response.target_map_id, TEST_MAP_ID)
        self.assertEqual(response.source_mime_type, "image/png")
        self.assertTrue(response.source_image_url.startswith("/generated-assets/"))
        self.assertEqual(loaded.job_id, response.job_id)
        self.assertIn("/maps/schema", response.next_step)

    def test_map_generation_job_rejects_invalid_base64(self):
        with self.assertRaises(HTTPException) as context:
            create_map_generation_job(
                MapGenerationJobCreateRequest(
                    source_image_base64="not-base64",
                    filename="floorplan.png",
                    mime_type="image/png",
                    target_map_id=TEST_MAP_ID,
                )
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_map_generation_job_rejects_protected_default_target(self):
        with self.assertRaises(HTTPException) as context:
            create_map_generation_job(
                MapGenerationJobCreateRequest(
                    source_image_base64=TEST_IMAGE_BASE64,
                    filename="floorplan.png",
                    mime_type="image/png",
                    target_map_id="default",
                )
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_map_generation_job_rejects_missing_job(self):
        with self.assertRaises(HTTPException) as context:
            get_map_generation_job("0" * 32)

        self.assertEqual(context.exception.status_code, 404)

    def test_map_generation_jobs_lists_recent_jobs(self):
        first_job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="first.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        second_job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="second.png",
                mime_type="image/png",
                target_map_id="generated_test_map_2",
            )
        )

        response = get_map_generation_jobs(limit=10)

        self.assertEqual(response.jobs[0].job_id, second_job.job_id)
        self.assertEqual(response.jobs[1].job_id, first_job.job_id)
        self.assertEqual(response.jobs[0].source_filename, "second.png")

    def test_map_generation_jobs_rejects_invalid_limit(self):
        with self.assertRaises(HTTPException) as context:
            get_map_generation_jobs(limit=0)

        self.assertEqual(context.exception.status_code, 400)

    def test_map_generation_draft_validates_and_saves_generated_map(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = deepcopy(get_map("default"))
        generated_map["name"] = "LLM 생성 지도 테스트"

        draft = attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )
        save_response = save_map_generation_draft(
            job.job_id,
            MapGenerationSaveRequest(),
        )

        self.assertEqual(draft.status, "draft_valid")
        self.assertTrue(draft.validation["valid"])
        self.assertEqual(draft.validation["map_id"], TEST_MAP_ID)
        self.assertEqual(save_response.job.status, "map_saved")
        self.assertEqual(save_response.save_result.map_id, TEST_MAP_ID)
        self.assertEqual(get_map(TEST_MAP_ID)["name"], "LLM 생성 지도 테스트")

    def test_map_generation_draft_reports_invalid_map(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        broken_map = deepcopy(get_map("default"))
        broken_map["edges"][0]["to"] = "MISSING_NODE"

        draft = attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=broken_map),
        )

        self.assertEqual(draft.status, "draft_invalid")
        self.assertFalse(draft.validation["valid"])
        self.assertTrue(
            any("MISSING_NODE" in error for error in draft.validation["errors"])
        )

    def test_map_generation_postprocess_normalizes_llm_draft(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = {
            "id": "wrong_id",
            "name": "후처리 테스트",
            "image": "/generated-assets/test.png",
            "width": 500,
            "height": 300,
            "nodes": [
                {
                    "id": "main entrance",
                    "name": "메인 입구",
                    "x": -20,
                    "y": "30",
                    "type": "door",
                    "selectable": True,
                },
                {
                    "id": "booth-10",
                    "name": "부스 10",
                    "x": 120,
                    "y": 30,
                    "type": "booth",
                    "selectable": True,
                },
            ],
            "checkpoints": [
                {
                    "id": "info qr",
                    "name": "안내 QR",
                    "node_id": "main entrance",
                    "region": "outside",
                }
            ],
            "edges": [
                {
                    "id": "main entrance to booth-10",
                    "from": "main entrance",
                    "to": "booth-10",
                    "distance": "far",
                    "widthM": "3",
                    "zone": "wall",
                    "crowdRegion": "unknown",
                    "bidirectional": 1,
                }
            ],
        }
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        response = postprocess_map_generation_draft(
            job.job_id,
            MapGenerationPostprocessRequest(),
        )
        draft = get_map_generation_draft(job.job_id)

        self.assertTrue(response.applied)
        self.assertTrue(response.validation["valid"])
        self.assertEqual(response.processed_map["id"], TEST_MAP_ID)
        self.assertEqual(response.processed_map["nodes"][0]["id"], "MAIN_ENTRANCE")
        self.assertEqual(response.processed_map["nodes"][0]["x"], 0)
        self.assertEqual(response.processed_map["edges"][0]["from"], "MAIN_ENTRANCE")
        self.assertEqual(response.processed_map["edges"][0]["to"], "BOOTH_10")
        self.assertGreater(response.processed_map["edges"][0]["distance"], 0)
        self.assertTrue(
            any(change["code"] == "normalize_node_id" for change in response.changes)
        )
        self.assertEqual(draft.draft_map["nodes"][0]["id"], "MAIN_ENTRANCE")

    def test_map_generation_postprocess_can_preview_without_applying(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = deepcopy(get_map("default"))
        generated_map["nodes"][0]["id"] = "main entrance"
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        response = postprocess_map_generation_draft(
            job.job_id,
            MapGenerationPostprocessRequest(apply=False),
        )
        draft = get_map_generation_draft(job.job_id)

        self.assertFalse(response.applied)
        self.assertEqual(response.processed_map["nodes"][0]["id"], "MAIN_ENTRANCE")
        self.assertEqual(draft.draft_map["nodes"][0]["id"], "main entrance")

    def test_map_postprocess_flags_weak_llm_node_labels(self):
        generated_map = {
            "id": TEST_MAP_ID,
            "name": "Weak label map",
            "image": "/generated-assets/test.png",
            "width": 300,
            "height": 200,
            "nodes": [
                {
                    "id": "node 1",
                    "name": "Node 1",
                    "x": 20,
                    "y": 20,
                    "type": "junction",
                    "selectable": True,
                },
                {
                    "id": "node 2",
                    "name": "Node 2",
                    "x": 120,
                    "y": 20,
                    "type": "booth",
                    "selectable": True,
                },
            ],
            "checkpoints": [],
            "edges": [
                {
                    "id": "edge 1",
                    "from": "node 1",
                    "to": "node 2",
                    "distance": 10,
                    "widthM": 3,
                    "zone": "lobby",
                    "crowdRegion": "central",
                    "bidirectional": True,
                }
            ],
        }

        result = postprocess_map_data(generated_map)

        self.assertTrue(
            any(
                suggestion["code"] == "weak_node_labels"
                for suggestion in result["suggestions"]
            )
        )
        self.assertIn(
            "weak_node_labels",
            result["validation"]["quality_summary"]["reason_codes"],
        )
        self.assertEqual(
            result["validation"]["quality_summary"]["readiness"],
            "needs_review",
        )

    def test_map_generation_postprocess_suggests_connection_edges(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = {
            "id": TEST_MAP_ID,
            "name": "연결 후보 테스트",
            "image": "/generated-assets/test.png",
            "width": 500,
            "height": 300,
            "nodes": [
                {
                    "id": "ENTRY",
                    "name": "입구",
                    "x": 10,
                    "y": 20,
                    "type": "entrance",
                    "selectable": True,
                },
                {
                    "id": "LOBBY",
                    "name": "로비",
                    "x": 80,
                    "y": 20,
                    "type": "junction",
                    "selectable": False,
                },
                {
                    "id": "BOOTH_1",
                    "name": "부스 1",
                    "x": 160,
                    "y": 20,
                    "type": "booth",
                    "selectable": True,
                },
            ],
            "checkpoints": [],
            "edges": [
                {
                    "id": "E_ENTRY_LOBBY",
                    "from": "ENTRY",
                    "to": "LOBBY",
                    "distance": 7,
                    "widthM": 3,
                    "zone": "lobby",
                    "crowdRegion": "central",
                    "bidirectional": True,
                }
            ],
        }
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        response = postprocess_map_generation_draft(
            job.job_id,
            MapGenerationPostprocessRequest(apply=False),
        )

        suggestion = next(
            suggestion
            for suggestion in response.suggestions
            if suggestion["code"] == "connect_disconnected_components"
        )
        candidate = suggestion["candidate_edges"][0]
        self.assertEqual({candidate["from"], candidate["to"]}, {"LOBBY", "BOOTH_1"})
        self.assertEqual(len(response.processed_map["edges"]), 1)
        self.assertFalse(response.validation["valid"])
        self.assertEqual(response.validation["quality_summary"]["readiness"], "blocked")
        self.assertIn(
            "connection_edges_need_review",
            response.validation["quality_summary"]["reason_codes"],
        )

    def test_map_generation_postprocess_can_apply_connection_edges(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = {
            "id": TEST_MAP_ID,
            "name": "연결 적용 테스트",
            "image": "/generated-assets/test.png",
            "width": 500,
            "height": 300,
            "nodes": [
                {
                    "id": "ENTRY",
                    "name": "입구",
                    "x": 10,
                    "y": 20,
                    "type": "entrance",
                    "selectable": True,
                },
                {
                    "id": "LOBBY",
                    "name": "로비",
                    "x": 80,
                    "y": 20,
                    "type": "junction",
                    "selectable": False,
                },
                {
                    "id": "BOOTH_1",
                    "name": "부스 1",
                    "x": 160,
                    "y": 20,
                    "type": "booth",
                    "selectable": True,
                },
            ],
            "checkpoints": [],
            "edges": [
                {
                    "id": "E_ENTRY_LOBBY",
                    "from": "ENTRY",
                    "to": "LOBBY",
                    "distance": 7,
                    "widthM": 3,
                    "zone": "lobby",
                    "crowdRegion": "central",
                    "bidirectional": True,
                }
            ],
        }
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        response = postprocess_map_generation_draft(
            job.job_id,
            MapGenerationPostprocessRequest(apply_connection_suggestions=True),
        )
        route_response = preview_map_generation_route(
            job.job_id,
            MapGenerationRoutePreviewRequest(
                start_id="ENTRY",
                destination_id="BOOTH_1",
                crowd_inputs=CrowdInputs(
                    lobby_people=10,
                    booth_people=10,
                    recent_inflow=0,
                    hour=10,
                    event_phase=1,
                ),
            ),
        )

        self.assertTrue(response.validation["valid"])
        self.assertEqual(len(response.processed_map["edges"]), 2)
        self.assertEqual(response.validation["quality_summary"]["readiness"], "ready")
        self.assertTrue(
            any(change["code"] == "add_connection_edge" for change in response.changes)
        )
        self.assertEqual(route_response.path, ["ENTRY", "LOBBY", "BOOTH_1"])

    def test_map_generation_generate_draft_requires_openai_api_key(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(HTTPException) as context:
                generate_map_generation_draft(
                    job.job_id,
                    MapGenerationGenerateDraftRequest(),
                )

        self.assertEqual(context.exception.status_code, 503)

    def test_map_generation_generate_draft_attaches_llm_result(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = deepcopy(get_map("default"))
        generated_map["id"] = "llm_returned_wrong_id"
        generated_map["name"] = "OpenAI 자동 생성 지도"

        with patch("main.request_openai_map_draft", return_value=generated_map):
            response = generate_map_generation_draft(
                job.job_id,
                MapGenerationGenerateDraftRequest(extra_instructions="부스 중심"),
            )

        self.assertEqual(response.job.status, "draft_valid")
        self.assertEqual(response.job.validation["map_id"], TEST_MAP_ID)
        self.assertEqual(response.draft_map["id"], TEST_MAP_ID)
        self.assertEqual(response.draft_map["name"], "OpenAI 자동 생성 지도")

    def test_map_generation_draft_can_be_read_for_review(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = deepcopy(get_map("default"))
        generated_map["name"] = "검수용 draft"
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        response = get_map_generation_draft(job.job_id)

        self.assertEqual(response.job.job_id, job.job_id)
        self.assertEqual(response.draft_map["id"], TEST_MAP_ID)
        self.assertEqual(response.draft_map["name"], "검수용 draft")
        self.assertTrue(response.validation["valid"])

    def test_map_generation_draft_read_rejects_missing_draft(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )

        with self.assertRaises(HTTPException) as context:
            get_map_generation_draft(job.job_id)

        self.assertEqual(context.exception.status_code, 404)

    def test_map_generation_route_preview_uses_draft_map(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = deepcopy(get_map("default"))
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        response = preview_map_generation_route(
            job.job_id,
            MapGenerationRoutePreviewRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            ),
        )

        self.assertEqual(response.map_id, TEST_MAP_ID)
        self.assertEqual(response.path[0], "GATE_W1")
        self.assertEqual(response.path[-1], "BOOTH_10")
        self.assertGreater(response.total_distance, 0)
        self.assertTrue(response.instructions)

    def test_map_generation_route_preview_supports_checkpoint_start(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = deepcopy(get_map("default"))
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        response = preview_map_generation_route(
            job.job_id,
            MapGenerationRoutePreviewRequest(
                checkpoint_id="INFO_DESK_QR",
                destination_id="BOOTH_10",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            ),
        )

        self.assertEqual(response.map_id, TEST_MAP_ID)
        self.assertEqual(response.path[0], "INFO_DESK")
        self.assertEqual(response.path[-1], "BOOTH_10")

    def test_map_generation_route_preview_rejects_mismatched_checkpoint_node(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        generated_map = deepcopy(get_map("default"))
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=generated_map),
        )

        with self.assertRaises(HTTPException) as context:
            preview_map_generation_route(
                job.job_id,
                MapGenerationRoutePreviewRequest(
                    current_node_id="GATE_W1",
                    checkpoint_id="INFO_DESK_QR",
                    destination_id="BOOTH_10",
                ),
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(
            context.exception.detail["message"],
            "current_node_id does not match checkpoint node.",
        )

    def test_map_generation_route_preview_rejects_invalid_draft(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        broken_map = deepcopy(get_map("default"))
        broken_map["edges"][0]["to"] = "MISSING_NODE"
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=broken_map),
        )

        with self.assertRaises(HTTPException) as context:
            preview_map_generation_route(
                job.job_id,
                MapGenerationRoutePreviewRequest(
                    start_id="GATE_W1",
                    destination_id="BOOTH_10",
                ),
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertFalse(context.exception.detail["validation"]["valid"])

    def test_map_generation_save_rejects_invalid_draft(self):
        job = create_map_generation_job(
            MapGenerationJobCreateRequest(
                source_image_base64=TEST_IMAGE_BASE64,
                filename="floorplan.png",
                mime_type="image/png",
                target_map_id=TEST_MAP_ID,
            )
        )
        broken_map = deepcopy(get_map("default"))
        broken_map["edges"][0]["to"] = "MISSING_NODE"
        attach_map_generation_draft(
            job.job_id,
            MapGenerationDraftRequest(venue_map=broken_map),
        )

        with self.assertRaises(HTTPException) as context:
            save_map_generation_draft(job.job_id, MapGenerationSaveRequest())

        self.assertEqual(context.exception.status_code, 400)

    def test_route_returns_path_and_drawable_points(self):
        response = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertEqual(
            response.path,
            ["GATE_W1", "WEST_HALL", "LOBBY_CENTER", "BOOTH_GATE", "AISLE_05", "BOOTH_10"],
        )
        self.assertEqual(len(response.route_points), len(response.path))
        self.assertEqual(len(response.segments), len(response.edge_ids))
        self.assertEqual(response.route_points[0].node_id, "GATE_W1")
        self.assertEqual(response.route_points[-1].node_id, "BOOTH_10")
        self.assertEqual(response.segments[0].from_node.node_id, "GATE_W1")
        self.assertEqual(response.segments[0].to_node.node_id, "WEST_HALL")
        self.assertEqual(response.segments[-1].from_node.node_id, "AISLE_05")
        self.assertEqual(response.segments[-1].to_node.node_id, "BOOTH_10")
        self.assertGreater(response.segments[0].weighted_cost, 0)
        self.assertGreater(response.segments[0].estimated_seconds, 0)
        self.assertIn("메인 입구", response.segments[0].instruction)
        self.assertEqual(response.segments[0].maneuver, "start")
        self.assertIsInstance(response.segments[0].heading_degrees, int)
        self.assertIsInstance(response.segments[0].turn_degrees, int)
        self.assertIn(
            response.segments[1].maneuver,
            {
                "straight",
                "slight_left",
                "left",
                "slight_right",
                "right",
                "u_turn",
            },
        )
        self.assertIn("목적지는", response.segments[-1].instruction)
        self.assertEqual(response.instructions[0], response.segments[0].instruction)
        self.assertTrue(response.use_congestion)
        self.assertEqual(response.walking_speed_mps, 1.2)
        self.assertEqual(response.algorithm, "astar")
        self.assertGreater(response.expanded_state_count, 0)
        self.assertGreater(response.total_distance, 0)
        self.assertGreater(response.weighted_cost, response.total_distance)
        self.assertGreater(response.estimated_seconds, 0)
        self.assertEqual(len(response.predictions), 25)

    def test_route_supports_dijkstra_algorithm_for_comparison(self):
        astar = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                algorithm="astar",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )
        dijkstra = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                algorithm="dijkstra",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertEqual(astar.path, dijkstra.path)
        self.assertEqual(astar.edge_ids, dijkstra.edge_ids)
        self.assertEqual(astar.total_distance, dijkstra.total_distance)
        self.assertEqual(astar.weighted_cost, dijkstra.weighted_cost)
        self.assertEqual(dijkstra.algorithm, "dijkstra")
        self.assertLessEqual(astar.expanded_state_count, dijkstra.expanded_state_count)

    def test_route_supports_custom_walking_speed(self):
        response = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                walking_speed_mps=2.4,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertEqual(response.walking_speed_mps, 2.4)
        self.assertEqual(response.estimated_seconds, 16)

    def test_route_can_ignore_congestion_weights(self):
        response = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertFalse(response.use_congestion)
        self.assertEqual(response.total_distance, response.weighted_cost)
        self.assertTrue(
            all(prediction.multiplier == 1.0 for prediction in response.predictions)
        )
        self.assertTrue(
            all(segment.weighted_cost == segment.distance for segment in response.segments)
        )

    def test_route_preference_can_penalize_narrow_accessible_edges(self):
        shortest = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )
        accessible = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                preference="accessible",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertEqual(accessible.preference, "accessible")
        self.assertGreater(accessible.weighted_cost, shortest.weighted_cost)

    def test_route_avoids_blocked_edges(self):
        response = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                blocked_edge_ids=["E_LOBBY_BOOTH_GATE"],
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertIn("E_LOBBY_BOOTH_GATE", response.blocked_edge_ids)
        self.assertNotIn("E_LOBBY_BOOTH_GATE", response.edge_ids)
        self.assertIn("NORTH_HALL", response.path)

    def test_blocked_edges_can_be_configured_per_map(self):
        updated = update_blocked_edges(
            "default",
            BlockedEdgesRequest(
                edge_ids=["E_LOBBY_BOOTH_GATE"],
                reason="temporary closure",
            ),
        )
        current = get_blocked_edges("default")

        self.assertEqual(updated.blocked_edge_ids, ["E_LOBBY_BOOTH_GATE"])
        self.assertEqual(current.reason, "temporary closure")
        response = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )
        self.assertNotIn("E_LOBBY_BOOTH_GATE", response.edge_ids)

    def test_route_alternatives_returns_ranked_candidates(self):
        shortest = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )
        response = route_alternatives(
            RouteAlternativesRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                use_congestion=False,
                max_routes=3,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertGreaterEqual(len(response.alternatives), 2)
        self.assertEqual(response.alternatives[0].rank, 1)
        self.assertEqual(response.alternatives[0].path, shortest.path)
        self.assertEqual(response.alternatives[0].edge_ids, shortest.edge_ids)
        self.assertEqual(
            response.alternatives[0].overlap_with_best_edge_count,
            len(shortest.edge_ids),
        )
        self.assertEqual(response.alternatives[0].overlap_ratio, 1.0)
        self.assertEqual(response.alternatives[0].detour_ratio, 1.0)
        self.assertGreaterEqual(response.alternatives[0].quality_score, 0)
        self.assertNotEqual(
            response.alternatives[1].path,
            response.alternatives[0].path,
        )
        self.assertEqual(response.alternatives[1].rank, 2)
        self.assertLess(response.alternatives[1].overlap_ratio, 1.0)
        self.assertGreaterEqual(response.alternatives[1].detour_ratio, 1.0)
        self.assertGreaterEqual(response.alternatives[1].quality_score, 0)
        self.assertLessEqual(response.alternatives[1].quality_score, 100)
        self.assertTrue(response.alternatives[1].instructions)

    def test_route_alternatives_rejects_unknown_node_id(self):
        with self.assertRaises(HTTPException) as context:
            route_alternatives(
                RouteAlternativesRequest(
                    start_id="UNKNOWN",
                    destination_id="BOOTH_10",
                    max_routes=2,
                )
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_route_recommendation_explains_selected_candidate(self):
        response = route_recommendation(
            RouteAlternativesRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                preference="less_crowded",
                use_congestion=True,
                max_routes=3,
                crowd_inputs=CrowdInputs(
                    lobby_people=150,
                    booth_people=250,
                    recent_inflow=80,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertEqual(response.recommendation.selected_rank, 1)
        self.assertEqual(response.recommendation.preference, "less_crowded")
        self.assertTrue(response.recommendation.reasons)
        self.assertEqual(response.selected.rank, 1)
        self.assertEqual(
            response.recommendation.tradeoffs["candidate_count"],
            len(response.alternatives),
        )
        self.assertTrue(response.recommendation.tradeoffs["candidates"])

    def test_route_multi_stop_uses_optimal_order_by_default(self):
        response = route_multi_stop(
            RouteMultiStopRequest(
                start_id="GATE_W1",
                destination_ids=["PHOTO_ZONE", "CATERING"],
                use_congestion=False,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertEqual(response.order_algorithm, "optimal")
        self.assertEqual(response.pairwise_route_count, 4)
        self.assertEqual(response.ordered_destination_ids, ["CATERING", "PHOTO_ZONE"])
        self.assertEqual(len(response.legs), 2)
        self.assertEqual(response.legs[0].start_id, "GATE_W1")
        self.assertEqual(response.legs[0].destination_id, "CATERING")
        self.assertGreater(response.total_distance, 0)
        self.assertGreater(response.estimated_seconds, 0)

    def test_route_multi_stop_can_use_nearest_order(self):
        response = route_multi_stop(
            RouteMultiStopRequest(
                start_id="GATE_W1",
                destination_ids=["PHOTO_ZONE", "CATERING"],
                order_algorithm="nearest",
                use_congestion=False,
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        self.assertEqual(response.order_algorithm, "nearest")
        self.assertEqual(response.ordered_destination_ids, ["CATERING", "PHOTO_ZONE"])

    def test_route_multi_stop_rejects_duplicate_destinations(self):
        with self.assertRaises(HTTPException) as context:
            route_multi_stop(
                RouteMultiStopRequest(
                    start_id="GATE_W1",
                    destination_ids=["CATERING", "CATERING"],
                )
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_route_multi_stop_rejects_start_as_destination(self):
        with self.assertRaises(HTTPException) as context:
            route_multi_stop(
                RouteMultiStopRequest(
                    start_id="GATE_W1",
                    destination_ids=["GATE_W1", "CATERING"],
                )
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_route_simulation_runs_mixed_success_and_failure_scenarios(self):
        response = route_simulation(
            RouteSimulationRequest(
                scenarios=[
                    RouteSimulationScenario(
                        id="normal-route",
                        start_id="GATE_W1",
                        destination_id="BOOTH_10",
                    ),
                    RouteSimulationScenario(
                        id="blocked-route",
                        start_id="GATE_W1",
                        destination_id="BOOTH_10",
                        blocked_edge_ids=[
                            "E_W1_WEST",
                            "E_W2_WEST",
                            "E_WEST_LOBBY",
                        ],
                    ),
                    RouteSimulationScenario(
                        id="bad-node",
                        start_id="UNKNOWN",
                        destination_id="BOOTH_10",
                    ),
                ],
                default_crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
                default_use_congestion=False,
            )
        )

        self.assertEqual(response.scenario_count, 3)
        self.assertEqual(response.success_count, 1)
        self.assertEqual(response.failure_count, 2)
        self.assertEqual(response.failures, ["blocked-route", "bad-node"])
        self.assertTrue(response.results[0].ok)
        self.assertFalse(response.results[1].ok)
        self.assertEqual(response.results[2].error["status_code"], 400)

    def test_crowd_estimate_uses_manual_and_qr_signals(self):
        update_manual_crowd(
            ManualCrowdRequest(
                map_id="default",
                lobby_people=30,
                booth_people=40,
            )
        )
        create_qr_scan(
            QrScanRequest(
                map_id="default",
                checkpoint_id="LOBBY_QR",
                region="lobby",
                count=5,
            )
        )
        create_qr_scan(
            QrScanRequest(
                map_id="default",
                checkpoint_id="BOOTH_QR",
                region="booth",
                count=7,
            )
        )

        estimate = get_crowd_estimate("default")

        self.assertEqual(estimate["lobby_people"], 35)
        self.assertEqual(estimate["booth_people"], 47)
        self.assertEqual(estimate["recent_inflow"], 12)

    def test_crowd_forecast_projects_recent_signals(self):
        update_manual_crowd(
            ManualCrowdRequest(
                map_id="default",
                lobby_people=30,
                booth_people=40,
            )
        )
        create_qr_scan(
            QrScanRequest(
                map_id="default",
                checkpoint_id="BOOTH_QR",
                region="booth",
                count=10,
            )
        )
        route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=True,
                crowd_inputs=CrowdInputs(
                    lobby_people=30,
                    booth_people=40,
                    recent_inflow=10,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        forecast = get_crowd_forecast("default", horizon_minutes=20, step_minutes=10)

        self.assertEqual(forecast["method"], "manual_qr_route_intent_decay")
        self.assertEqual(
            [point["minutes_ahead"] for point in forecast["points"]],
            [0, 10, 20],
        )
        self.assertGreaterEqual(
            forecast["points"][0]["booth_people"],
            forecast["points"][-1]["booth_people"],
        )

    def test_crowd_bottlenecks_returns_busy_edges(self):
        update_manual_crowd(
            ManualCrowdRequest(
                map_id="default",
                lobby_people=900,
                booth_people=950,
            )
        )

        response = get_crowd_bottlenecks("default", limit=3)

        self.assertLessEqual(len(response["bottlenecks"]), 3)
        self.assertTrue(response["bottlenecks"])
        self.assertIn(
            response["bottlenecks"][0]["level"],
            {"busy", "very_busy"},
        )

    def test_demand_heatmap_combines_crowd_and_intent_signals(self):
        update_manual_crowd(
            ManualCrowdRequest(
                map_id="default",
                lobby_people=40,
                booth_people=160,
            )
        )
        create_qr_scan(
            QrScanRequest(
                map_id="default",
                checkpoint_id="BOOTH_QR",
                region="booth",
                count=12,
            )
        )
        route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=True,
                crowd_inputs=CrowdInputs(
                    lobby_people=40,
                    booth_people=160,
                    recent_inflow=12,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        heatmap = get_demand_heatmap("default")

        self.assertEqual(heatmap.map_id, "default")
        self.assertEqual(heatmap.method, "manual_qr_route_intent_prediction_width")
        self.assertEqual(heatmap.regions[0].region, "booth")
        self.assertGreater(heatmap.regions[0].demand_score, 0)
        self.assertEqual(len(heatmap.edges), 25)
        self.assertEqual(
            [edge.demand_score for edge in heatmap.edges],
            sorted([edge.demand_score for edge in heatmap.edges], reverse=True),
        )
        self.assertIn(
            heatmap.edges[0].heat_level,
            {"low", "medium", "high", "critical"},
        )

    def test_route_can_use_estimated_crowd_inputs(self):
        update_manual_crowd(
            ManualCrowdRequest(
                map_id="default",
                lobby_people=30,
                booth_people=40,
            )
        )

        response = route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )

        self.assertEqual(response.crowd_inputs.lobby_people, 30)
        self.assertEqual(response.crowd_inputs.booth_people, 40)
        self.assertEqual(response.crowd_inputs.recent_inflow, 0)
        self.assertGreater(response.weighted_cost, 0)

    def test_position_update_reroutes_from_current_node(self):
        response = update_position(
            PositionUpdateRequest(
                current_node_id="LOBBY_CENTER",
                destination_id="BOOTH_10",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
                log_route_intent=False,
            )
        )

        self.assertEqual(response.current_node.node_id, "LOBBY_CENTER")
        self.assertEqual(response.destination_node.node_id, "BOOTH_10")
        self.assertEqual(response.location_source, "node")
        self.assertEqual(response.route.path[0], "LOBBY_CENTER")
        self.assertEqual(response.route.path[-1], "BOOTH_10")

    def test_checkpoint_position_update_records_qr_signal(self):
        response = update_position(
            PositionUpdateRequest(
                destination_id="BOOTH_10",
                checkpoint_id="BOOTH_GATE_QR",
                log_route_intent=False,
            )
        )
        snapshot = get_telemetry("default", limit=10)

        self.assertEqual(response.location_source, "checkpoint")
        self.assertEqual(response.route.path[0], "BOOTH_GATE")
        self.assertEqual(snapshot["crowd_estimate"]["signals"]["qr_scans"]["booth"], 1)
        self.assertEqual(snapshot["recent_qr_scans"][0]["checkpoint_id"], "BOOTH_GATE_QR")

    def test_checkpoint_position_update_rejects_mismatched_node(self):
        with self.assertRaises(HTTPException) as context:
            update_position(
                PositionUpdateRequest(
                    current_node_id="LOBBY_CENTER",
                    destination_id="BOOTH_10",
                    checkpoint_id="BOOTH_GATE_QR",
                )
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_navigation_session_can_start_and_be_read(self):
        response = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
                log_route_intent=False,
            )
        )
        status = get_navigation_session_status(response.session_id)

        self.assertEqual(response.status, "active")
        self.assertEqual(response.current_node.node_id, "GATE_W1")
        self.assertEqual(response.destination_node.node_id, "BOOTH_10")
        self.assertEqual(status.session_id, response.session_id)
        self.assertEqual(status.route.path[0], "GATE_W1")
        self.assertGreaterEqual(len(status.recent_updates), 1)

    def test_navigation_session_position_update_changes_current_route(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )
        response = update_navigation_session_position_api(
            session.session_id,
            NavigationSessionUpdateRequest(
                checkpoint_id="BOOTH_GATE_QR",
            ),
        )
        snapshot = get_telemetry("default", limit=10)

        self.assertEqual(response.current_node.node_id, "BOOTH_GATE")
        self.assertEqual(response.route.path[0], "BOOTH_GATE")
        self.assertEqual(response.recent_updates[0].checkpoint_id, "BOOTH_GATE_QR")
        self.assertEqual(snapshot["crowd_estimate"]["signals"]["qr_scans"]["booth"], 1)

    def test_coordinate_snap_finds_nearest_node(self):
        response = snap_coordinate(
            "default",
            CoordinateSnapRequest(x=572, y=829, max_distance_px=10),
        )

        self.assertEqual(response.snapped_node.node_id, "GATE_W1")
        self.assertTrue(response.within_threshold)
        self.assertTrue(response.selectable)
        self.assertEqual(response.snapped_edge_id, "E_W1_WEST")
        self.assertLessEqual(response.edge_distance_px, 3)

    def test_coordinate_snap_reports_nearest_edge_projection(self):
        response = snap_coordinate(
            "default",
            CoordinateSnapRequest(x=880, y=697, max_distance_px=250),
        )

        self.assertEqual(response.snapped_edge_id, "E_WEST_LOBBY")
        self.assertEqual(response.snapped_edge_from_node_id, "WEST_HALL")
        self.assertEqual(response.snapped_edge_to_node_id, "LOBBY_CENTER")
        self.assertIsNotNone(response.edge_progress_ratio)
        self.assertIsNotNone(response.projected_x)
        self.assertIsNotNone(response.projected_y)
        self.assertLessEqual(response.edge_distance_px, 5)

    def test_coordinate_snap_can_limit_to_selectable_nodes(self):
        response = snap_coordinate(
            "default",
            CoordinateSnapRequest(
                x=1088,
                y=646,
                max_distance_px=500,
                selectable_only=True,
            ),
        )

        self.assertNotEqual(response.snapped_node.node_id, "LOBBY_CENTER")
        self.assertTrue(response.snapped_node.node_id)
        self.assertIsNone(response.snapped_edge_id)

    def test_coordinate_snap_rejects_out_of_bounds_input(self):
        with self.assertRaises(HTTPException) as context:
            snap_coordinate(
                "default",
                CoordinateSnapRequest(x=-1, y=829, max_distance_px=10),
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_navigation_session_coordinate_update_changes_current_route(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )
        response = update_navigation_session_coordinate_api(
            session.session_id,
            NavigationSessionCoordinateUpdateRequest(
                x=1088,
                y=646,
                max_distance_px=20,
                use_congestion=False,
            ),
        )

        self.assertEqual(response.current_node.node_id, "LOBBY_CENTER")
        self.assertEqual(response.route.path[0], "LOBBY_CENTER")
        self.assertEqual(response.recent_updates[0].source, "coordinate")

    def test_navigation_session_coordinate_update_uses_destination_direction_on_edge(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )
        response = update_navigation_session_coordinate_api(
            session.session_id,
            NavigationSessionCoordinateUpdateRequest(
                x=880,
                y=697,
                max_distance_px=250,
                use_congestion=False,
            ),
        )

        self.assertEqual(response.current_node.node_id, "LOBBY_CENTER")
        self.assertEqual(response.route.path[0], "LOBBY_CENTER")

    def test_navigation_session_coordinate_update_uses_reverse_direction_on_edge(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="BOOTH_10",
                destination_id="GATE_W1",
                log_route_intent=False,
            )
        )
        response = update_navigation_session_coordinate_api(
            session.session_id,
            NavigationSessionCoordinateUpdateRequest(
                x=880,
                y=697,
                max_distance_px=250,
                use_congestion=False,
            ),
        )

        self.assertEqual(response.current_node.node_id, "WEST_HALL")
        self.assertEqual(response.route.path[0], "WEST_HALL")

    def test_navigation_session_coordinate_update_rejects_distant_snap(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )

        with self.assertRaises(HTTPException) as context:
            update_navigation_session_coordinate_api(
                session.session_id,
                NavigationSessionCoordinateUpdateRequest(
                    x=1088,
                    y=646,
                    max_distance_px=1,
                    use_congestion=False,
                ),
            )

        self.assertEqual(context.exception.status_code, 400)

    def test_navigation_guidance_reports_next_step_and_progress(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )
        initial = get_navigation_guidance(session.session_id)

        self.assertEqual(initial.current_node.node_id, "GATE_W1")
        self.assertEqual(initial.next_node.node_id, "WEST_HALL")
        self.assertEqual(initial.progress_ratio, 0)
        self.assertFalse(initial.off_route)
        self.assertGreater(initial.remaining_distance, 0)
        self.assertTrue(initial.instruction)

        update_navigation_session_position_api(
            session.session_id,
            NavigationSessionUpdateRequest(current_node_id="LOBBY_CENTER"),
        )
        progressed = get_navigation_guidance(session.session_id)

        self.assertEqual(progressed.current_node.node_id, "LOBBY_CENTER")
        self.assertGreater(progressed.progress_ratio, initial.progress_ratio)
        self.assertLess(progressed.remaining_distance, initial.remaining_distance)
        self.assertEqual(progressed.next_node.node_id, "BOOTH_GATE")

    def test_navigation_guidance_marks_off_route_current_node(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )
        update_navigation_session_position_api(
            session.session_id,
            NavigationSessionUpdateRequest(current_node_id="GATE_N"),
        )

        guidance = get_navigation_guidance(session.session_id)

        self.assertTrue(guidance.off_route)
        self.assertEqual(guidance.progress_ratio, 0)
        self.assertIn("GATE_W1", guidance.expected_path)
        self.assertIn("BOOTH_10", guidance.expected_path)
        self.assertNotIn("GATE_N", guidance.expected_path)
        self.assertEqual(guidance.route.path[0], "GATE_N")
        self.assertEqual(guidance.next_node.node_id, "NORTH_HALL")

    def test_navigation_session_marks_arrived_at_destination(self):
        session = start_navigation_session(
            NavigationSessionStartRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                log_route_intent=False,
            )
        )
        response = update_navigation_session_position_api(
            session.session_id,
            NavigationSessionUpdateRequest(current_node_id="BOOTH_10"),
        )

        self.assertEqual(response.status, "arrived")
        self.assertEqual(response.current_node.node_id, "BOOTH_10")
        self.assertEqual(response.route.path, ["BOOTH_10"])
        self.assertEqual(response.route.total_distance, 0)

    def test_navigation_eta_calibration_uses_position_update_timestamps(self):
        base_time = datetime(2026, 8, 5, 5, 0, tzinfo=timezone.utc)
        with patch("crowd_store.current_utc", return_value=base_time):
            session = start_navigation_session(
                NavigationSessionStartRequest(
                    start_id="GATE_W1",
                    destination_id="BOOTH_10",
                    log_route_intent=False,
                    use_congestion=False,
                    crowd_inputs=CrowdInputs(
                        lobby_people=25,
                        booth_people=55,
                        recent_inflow=20,
                        hour=14,
                        event_phase=2,
                    ),
                )
            )

        with patch(
            "crowd_store.current_utc",
            return_value=base_time + timedelta(seconds=5),
        ):
            update_navigation_session_position_api(
                session.session_id,
                NavigationSessionUpdateRequest(
                    current_node_id="WEST_HALL",
                    use_congestion=False,
                    crowd_inputs=CrowdInputs(
                        lobby_people=25,
                        booth_people=55,
                        recent_inflow=20,
                        hour=14,
                        event_phase=2,
                    ),
                ),
            )

        with patch(
            "crowd_store.current_utc",
            return_value=base_time + timedelta(seconds=25),
        ):
            update_navigation_session_position_api(
                session.session_id,
                NavigationSessionUpdateRequest(
                    current_node_id="LOBBY_CENTER",
                    use_congestion=False,
                    crowd_inputs=CrowdInputs(
                        lobby_people=25,
                        booth_people=55,
                        recent_inflow=20,
                        hour=14,
                        event_phase=2,
                    ),
                ),
            )

        calibration = get_navigation_eta_calibration(session.session_id)

        self.assertEqual(calibration.sample_count, 2)
        self.assertEqual(calibration.confidence, 0.4)
        self.assertLess(calibration.observed_walking_speed_mps, 1.2)
        self.assertLess(calibration.recommended_walking_speed_mps, 1.2)
        self.assertEqual(calibration.samples[0].from_node_id, "GATE_W1")
        self.assertEqual(calibration.samples[0].to_node_id, "WEST_HALL")

    def test_navigation_session_rejects_unknown_session(self):
        with self.assertRaises(HTTPException) as context:
            get_navigation_session_status("missing-session")

        self.assertEqual(context.exception.status_code, 404)

    def test_telemetry_snapshot_includes_recent_signals(self):
        update_manual_crowd(
            ManualCrowdRequest(
                map_id="default",
                lobby_people=30,
                booth_people=40,
            )
        )
        create_qr_scan(
            QrScanRequest(
                map_id="default",
                checkpoint_id="BOOTH_QR",
                region="booth",
                count=3,
            )
        )
        route(
            RouteRequest(
                start_id="GATE_W1",
                destination_id="BOOTH_10",
                crowd_inputs=CrowdInputs(
                    lobby_people=25,
                    booth_people=55,
                    recent_inflow=20,
                    hour=14,
                    event_phase=2,
                ),
            )
        )

        snapshot = get_telemetry("default", limit=10)

        self.assertEqual(snapshot["map_id"], "default")
        self.assertEqual(snapshot["crowd_estimate"]["signals"]["qr_scans"]["booth"], 3)
        self.assertEqual(len(snapshot["recent_qr_scans"]), 1)
        self.assertEqual(snapshot["recent_qr_scans"][0]["checkpoint_id"], "BOOTH_QR")
        self.assertEqual(len(snapshot["recent_route_intents"]), 1)
        self.assertEqual(
            snapshot["recent_route_intents"][0]["destination_id"],
            "BOOTH_10",
        )

    def test_telemetry_rejects_invalid_limit(self):
        with self.assertRaises(HTTPException) as context:
            get_telemetry("default", limit=0)

        self.assertEqual(context.exception.status_code, 400)

    def test_route_rejects_unknown_node_id(self):
        with self.assertRaises(HTTPException) as context:
            route(
                RouteRequest(
                    start_id="UNKNOWN",
                    destination_id="BOOTH_10",
                    crowd_inputs=CrowdInputs(
                        lobby_people=25,
                        booth_people=55,
                        recent_inflow=20,
                        hour=14,
                        event_phase=2,
                    ),
                )
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail["missing_node_ids"], ["UNKNOWN"])

    def test_map_validation_finds_broken_edge_reference(self):
        venue_map = get_map("default")
        broken_map = deepcopy(venue_map)
        broken_map["edges"][0]["to"] = "MISSING_NODE"

        validation = validate_map_data(broken_map)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("MISSING_NODE" in error for error in validation["errors"])
        )

    def test_map_validation_finds_broken_checkpoint_reference(self):
        venue_map = get_map("default")
        broken_map = deepcopy(venue_map)
        broken_map["checkpoints"][0]["node_id"] = "MISSING_NODE"

        validation = validate_map_data(broken_map)

        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("MISSING_NODE" in error for error in validation["errors"])
        )

    def test_map_validation_finds_llm_coordinate_and_enum_errors(self):
        venue_map = get_map("default")
        broken_map = deepcopy(venue_map)
        broken_map["nodes"][0]["x"] = broken_map["width"] + 100
        broken_map["nodes"][1]["type"] = "stairs"
        broken_map["edges"][0]["zone"] = "wall"
        broken_map["edges"][1]["distance"] = "far"
        broken_map["checkpoints"][0]["region"] = "outside"

        validation = validate_map_data(broken_map)

        self.assertFalse(validation["valid"])
        self.assertIn(broken_map["nodes"][0]["id"], validation["out_of_bounds_node_ids"])
        self.assertIn(broken_map["nodes"][1]["id"], validation["invalid_node_ids"])
        self.assertIn(broken_map["edges"][0]["id"], validation["invalid_edge_ids"])
        self.assertIn(broken_map["edges"][1]["id"], validation["invalid_edge_ids"])
        self.assertIn(
            broken_map["checkpoints"][0]["id"],
            validation["invalid_checkpoint_ids"],
        )
        self.assertTrue(any("outside map bounds" in error for error in validation["errors"]))
        self.assertTrue(any("zone is invalid" in error for error in validation["errors"]))
        self.assertTrue(any("distance must be a number" in error for error in validation["errors"]))
        suggestion_codes = {
            suggestion["code"] for suggestion in validation["fix_suggestions"]
        }
        self.assertIn("node_out_of_bounds", suggestion_codes)
        self.assertIn("invalid_nodes", suggestion_codes)
        self.assertIn("invalid_edges", suggestion_codes)
        self.assertIn("invalid_checkpoints", suggestion_codes)

    def test_map_validation_finds_unreachable_selectable_pairs(self):
        venue_map = get_map("default")
        broken_map = deepcopy(venue_map)
        broken_map["nodes"].append(
            {
                "id": "ISOLATED_DEST",
                "name": "고립 목적지",
                "x": 10,
                "y": 10,
                "type": "facility",
                "selectable": True,
            }
        )

        validation = validate_map_data(broken_map)

        self.assertFalse(validation["valid"])
        self.assertTrue(validation["unreachable_route_pairs"])
        self.assertEqual(validation["quality_summary"]["readiness"], "blocked")
        self.assertFalse(validation["quality_summary"]["can_save"])
        self.assertTrue(
            any(
                pair["destination_id"] == "ISOLATED_DEST"
                for pair in validation["unreachable_route_pairs"]
            )
        )
        suggestion = next(
            suggestion
            for suggestion in validation["fix_suggestions"]
            if suggestion["code"] == "unreachable_selectable_nodes"
        )
        self.assertIn("ISOLATED_DEST", suggestion["target_ids"])
        self.assertTrue(suggestion["sample_pairs"])


if __name__ == "__main__":
    unittest.main()
