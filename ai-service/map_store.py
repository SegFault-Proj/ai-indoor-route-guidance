import json
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

MAPS_DIR = Path(__file__).with_name("maps")
MAP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
PROTECTED_MAP_IDS = {"default"}
NODE_TYPES = {"entrance", "exit", "junction", "facility", "booth"}
ZONES = {"lobby", "booth", "gate", "facility"}
CROWD_REGIONS = {"west", "central", "north", "booth"}
CHECKPOINT_REGIONS = {"lobby", "booth"}


def reachable_node_ids(start_id: str, graph: dict[str, list[str]]) -> set[str]:
    visited: set[str] = set()
    stack = [start_id]

    while stack:
        current_id = stack.pop()
        if current_id in visited:
            continue
        visited.add(current_id)
        stack.extend(node_id for node_id in graph.get(current_id, []) if node_id not in visited)

    return visited


def unreachable_selectable_pairs(
    selectable_node_ids: list[str],
    edges: list[dict[str, Any]],
    node_id_set: set[str],
) -> list[dict[str, str]]:
    graph: dict[str, list[str]] = {node_id: [] for node_id in node_id_set}
    for edge in edges:
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id not in node_id_set or to_id not in node_id_set:
            continue

        graph[from_id].append(to_id)
        if edge.get("bidirectional", False):
            graph[to_id].append(from_id)

    unreachable_pairs: list[dict[str, str]] = []
    for start_id in selectable_node_ids:
        reachable_ids = reachable_node_ids(start_id, graph)
        for destination_id in selectable_node_ids:
            if start_id == destination_id:
                continue
            if destination_id not in reachable_ids:
                unreachable_pairs.append(
                    {
                        "start_id": start_id,
                        "destination_id": destination_id,
                    }
                )

    return unreachable_pairs


def empty_validation_result(errors: list[str], warnings: list[str]) -> dict[str, Any]:
    suggestions = fix_suggestions(
        errors=errors,
        warnings=warnings,
        unreachable_pairs=[],
        out_of_bounds_node_ids=[],
        invalid_node_ids=[],
        invalid_edge_ids=[],
        invalid_checkpoint_ids=[],
    )
    return {
        "valid": False,
        "errors": errors,
        "warnings": warnings,
        "node_count": 0,
        "edge_count": 0,
        "checkpoint_count": 0,
        "selectable_node_count": 0,
        "unreachable_route_pairs": [],
        "out_of_bounds_node_ids": [],
        "invalid_node_ids": [],
        "invalid_edge_ids": [],
        "invalid_checkpoint_ids": [],
        "fix_suggestions": suggestions,
        "quality_summary": quality_summary(
            errors=errors,
            warnings=warnings,
            unreachable_pairs=[],
            out_of_bounds_node_ids=[],
            invalid_node_ids=[],
            invalid_edge_ids=[],
            invalid_checkpoint_ids=[],
            fix_suggestions=suggestions,
        ),
    }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fix_suggestions(
    errors: list[str],
    warnings: list[str],
    unreachable_pairs: list[dict[str, str]],
    out_of_bounds_node_ids: list[str],
    invalid_node_ids: list[str],
    invalid_edge_ids: list[str],
    invalid_checkpoint_ids: list[str],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    if out_of_bounds_node_ids:
        suggestions.append(
            {
                "code": "node_out_of_bounds",
                "severity": "error",
                "target_ids": out_of_bounds_node_ids,
                "message": (
                    "Move these node coordinates inside the map image bounds "
                    "or correct the map width/height."
                ),
            }
        )

    if invalid_node_ids:
        suggestions.append(
            {
                "code": "invalid_nodes",
                "severity": "error",
                "target_ids": invalid_node_ids,
                "message": (
                    "Fix node ids, names, numeric x/y coordinates, node type, "
                    "and selectable boolean values."
                ),
                "allowed_values": {
                    "type": sorted(NODE_TYPES),
                },
            }
        )

    if invalid_edge_ids:
        suggestions.append(
            {
                "code": "invalid_edges",
                "severity": "error",
                "target_ids": invalid_edge_ids,
                "message": (
                    "Fix edge endpoints, positive numeric distance/widthM, "
                    "zone, crowdRegion, and bidirectional boolean values."
                ),
                "allowed_values": {
                    "zone": sorted(ZONES),
                    "crowdRegion": sorted(CROWD_REGIONS),
                },
            }
        )

    if invalid_checkpoint_ids:
        suggestions.append(
            {
                "code": "invalid_checkpoints",
                "severity": "error",
                "target_ids": invalid_checkpoint_ids,
                "message": (
                    "Fix checkpoint ids, names, node_id references, and region values."
                ),
                "allowed_values": {
                    "region": sorted(CHECKPOINT_REGIONS),
                },
            }
        )

    if unreachable_pairs:
        affected_ids = sorted(
            {
                node_id
                for pair in unreachable_pairs
                for node_id in [pair["start_id"], pair["destination_id"]]
            }
        )
        suggestions.append(
            {
                "code": "unreachable_selectable_nodes",
                "severity": "error",
                "target_ids": affected_ids,
                "message": (
                    "Add walkable edges between disconnected selectable nodes, "
                    "or mark unreachable destinations as not selectable."
                ),
                "sample_pairs": unreachable_pairs[:10],
            }
        )

    if any("Map has no selectable nodes" in warning for warning in warnings):
        suggestions.append(
            {
                "code": "no_selectable_nodes",
                "severity": "warning",
                "target_ids": [],
                "message": (
                    "Mark at least one usable start point and destination node as selectable."
                ),
            }
        )

    if not suggestions and errors:
        suggestions.append(
            {
                "code": "schema_errors",
                "severity": "error",
                "target_ids": [],
                "message": "Fix the required map schema fields reported in errors.",
            }
        )

    return suggestions


def quality_summary(
    errors: list[str],
    warnings: list[str],
    unreachable_pairs: list[dict[str, str]],
    out_of_bounds_node_ids: list[str],
    invalid_node_ids: list[str],
    invalid_edge_ids: list[str],
    invalid_checkpoint_ids: list[str],
    fix_suggestions: list[dict[str, Any]],
) -> dict[str, Any]:
    penalty = 0
    reasons: list[str] = []

    if errors:
        penalty += min(60, len(errors) * 12)
        reasons.append("validation_errors")
    if unreachable_pairs:
        penalty += min(30, len(unreachable_pairs) * 5)
        reasons.append("unreachable_selectable_nodes")
    if out_of_bounds_node_ids:
        penalty += min(25, len(out_of_bounds_node_ids) * 8)
        reasons.append("out_of_bounds_nodes")
    if invalid_node_ids:
        penalty += min(25, len(invalid_node_ids) * 8)
        reasons.append("invalid_nodes")
    if invalid_edge_ids:
        penalty += min(25, len(invalid_edge_ids) * 8)
        reasons.append("invalid_edges")
    if invalid_checkpoint_ids:
        penalty += min(20, len(invalid_checkpoint_ids) * 6)
        reasons.append("invalid_checkpoints")
    if warnings:
        penalty += min(15, len(warnings) * 5)
        reasons.append("warnings")

    suggestion_codes = {suggestion["code"] for suggestion in fix_suggestions}
    if "no_selectable_nodes" in suggestion_codes:
        penalty += 20
        reasons.append("no_selectable_nodes")

    score = max(0, 100 - penalty)
    blocking_reason_codes = {
        "validation_errors",
        "unreachable_selectable_nodes",
        "out_of_bounds_nodes",
        "invalid_nodes",
        "invalid_edges",
        "invalid_checkpoints",
        "no_selectable_nodes",
    }

    if any(reason in blocking_reason_codes for reason in reasons):
        readiness = "blocked"
        can_save = False
    elif score < 90 or warnings:
        readiness = "needs_review"
        can_save = True
    else:
        readiness = "ready"
        can_save = True

    return {
        "score": score,
        "readiness": readiness,
        "can_save": can_save,
        "reason_codes": reasons,
    }


def validate_map_data(venue_map: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    invalid_node_ids: set[str] = set()
    invalid_edge_ids: set[str] = set()
    invalid_checkpoint_ids: set[str] = set()
    out_of_bounds_node_ids: set[str] = set()

    required_fields = ["id", "name", "image", "width", "height", "nodes", "edges"]
    for field in required_fields:
        if field not in venue_map:
            errors.append(f"Missing required map field: {field}")

    if errors:
        return empty_validation_result(errors, warnings)

    nodes = venue_map["nodes"]
    edges = venue_map["edges"]
    checkpoints = venue_map.get("checkpoints", [])

    if not isinstance(venue_map["id"], str):
        errors.append("Map id must be a string")
    if not isinstance(venue_map["name"], str):
        errors.append("Map name must be a string")
    if not isinstance(venue_map["image"], str):
        errors.append("Map image must be a string")
    if not is_number(venue_map["width"]) or venue_map["width"] <= 0:
        errors.append("Map width must be a positive number")
    if not is_number(venue_map["height"]) or venue_map["height"] <= 0:
        errors.append("Map height must be a positive number")
    if not isinstance(nodes, list):
        errors.append("Map nodes must be a list")
    if not isinstance(edges, list):
        errors.append("Map edges must be a list")
    if not isinstance(checkpoints, list):
        errors.append("Map checkpoints must be a list")

    if errors:
        return empty_validation_result(errors, warnings)

    if any(not isinstance(node, dict) for node in nodes):
        errors.append("Every node must be an object")
    if any(not isinstance(edge, dict) for edge in edges):
        errors.append("Every edge must be an object")
    if any(not isinstance(checkpoint, dict) for checkpoint in checkpoints):
        errors.append("Every checkpoint must be an object")

    if errors:
        return empty_validation_result(errors, warnings)

    node_ids = [node.get("id") for node in nodes]
    edge_ids = [edge.get("id") for edge in edges]
    checkpoint_ids = [checkpoint.get("id") for checkpoint in checkpoints]
    node_id_set = set(node_ids)

    duplicate_node_ids = sorted(
        node_id for node_id in node_id_set if node_ids.count(node_id) > 1
    )
    duplicate_edge_ids = sorted(
        edge_id for edge_id in set(edge_ids) if edge_ids.count(edge_id) > 1
    )
    duplicate_checkpoint_ids = sorted(
        checkpoint_id
        for checkpoint_id in set(checkpoint_ids)
        if checkpoint_ids.count(checkpoint_id) > 1
    )

    if duplicate_node_ids:
        errors.append(f"Duplicate node ids: {duplicate_node_ids}")
    if duplicate_edge_ids:
        errors.append(f"Duplicate edge ids: {duplicate_edge_ids}")
    if duplicate_checkpoint_ids:
        errors.append(f"Duplicate checkpoint ids: {duplicate_checkpoint_ids}")

    for node in nodes:
        node_id = node.get("id", "<missing>")
        for field in ["id", "name", "x", "y", "type", "selectable"]:
            if field not in node:
                errors.append(f"Node {node_id} is missing field: {field}")
                invalid_node_ids.add(str(node_id))

        if "id" in node and not isinstance(node["id"], str):
            errors.append(f"Node {node_id} id must be a string")
            invalid_node_ids.add(str(node_id))
        if "name" in node and not isinstance(node["name"], str):
            errors.append(f"Node {node_id} name must be a string")
            invalid_node_ids.add(str(node_id))
        if "x" in node and not is_number(node["x"]):
            errors.append(f"Node {node_id} x must be a number")
            invalid_node_ids.add(str(node_id))
        if "y" in node and not is_number(node["y"]):
            errors.append(f"Node {node_id} y must be a number")
            invalid_node_ids.add(str(node_id))
        if "type" in node and node["type"] not in NODE_TYPES:
            errors.append(f"Node {node_id} type is invalid: {node['type']}")
            invalid_node_ids.add(str(node_id))
        if "selectable" in node and not isinstance(node["selectable"], bool):
            errors.append(f"Node {node_id} selectable must be a boolean")
            invalid_node_ids.add(str(node_id))

        if is_number(node.get("x")) and is_number(node.get("y")):
            if (
                node["x"] < 0
                or node["x"] > venue_map["width"]
                or node["y"] < 0
                or node["y"] > venue_map["height"]
            ):
                errors.append(f"Node {node_id} coordinates are outside map bounds")
                out_of_bounds_node_ids.add(str(node_id))

    connected_node_ids: set[str] = set()
    for edge in edges:
        edge_id = edge.get("id", "<missing>")
        for field in [
            "id",
            "from",
            "to",
            "distance",
            "widthM",
            "zone",
            "crowdRegion",
            "bidirectional",
        ]:
            if field not in edge:
                errors.append(f"Edge {edge_id} is missing field: {field}")
                invalid_edge_ids.add(str(edge_id))

        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id not in node_id_set:
            errors.append(f"Edge {edge_id} references unknown from node: {from_id}")
        else:
            connected_node_ids.add(from_id)

        if to_id not in node_id_set:
            errors.append(f"Edge {edge_id} references unknown to node: {to_id}")
        else:
            connected_node_ids.add(to_id)

        if "id" in edge and not isinstance(edge["id"], str):
            errors.append(f"Edge {edge_id} id must be a string")
            invalid_edge_ids.add(str(edge_id))
        if "from" in edge and not isinstance(edge["from"], str):
            errors.append(f"Edge {edge_id} from must be a string")
            invalid_edge_ids.add(str(edge_id))
        if "to" in edge and not isinstance(edge["to"], str):
            errors.append(f"Edge {edge_id} to must be a string")
            invalid_edge_ids.add(str(edge_id))
        if "distance" in edge and not is_number(edge["distance"]):
            errors.append(f"Edge {edge_id} distance must be a number")
            invalid_edge_ids.add(str(edge_id))
        elif edge.get("distance", 0) <= 0:
            errors.append(f"Edge {edge_id} distance must be greater than 0")
            invalid_edge_ids.add(str(edge_id))
        if "widthM" in edge and not is_number(edge["widthM"]):
            errors.append(f"Edge {edge_id} widthM must be a number")
            invalid_edge_ids.add(str(edge_id))
        elif edge.get("widthM", 0) <= 0:
            errors.append(f"Edge {edge_id} widthM must be greater than 0")
            invalid_edge_ids.add(str(edge_id))
        if "zone" in edge and edge["zone"] not in ZONES:
            errors.append(f"Edge {edge_id} zone is invalid: {edge['zone']}")
            invalid_edge_ids.add(str(edge_id))
        if "crowdRegion" in edge and edge["crowdRegion"] not in CROWD_REGIONS:
            errors.append(
                f"Edge {edge_id} crowdRegion is invalid: {edge['crowdRegion']}"
            )
            invalid_edge_ids.add(str(edge_id))
        if "bidirectional" in edge and not isinstance(edge["bidirectional"], bool):
            errors.append(f"Edge {edge_id} bidirectional must be a boolean")
            invalid_edge_ids.add(str(edge_id))

    selectable_nodes = [node for node in nodes if node.get("selectable", False)]
    if not selectable_nodes:
        warnings.append("Map has no selectable nodes.")

    for checkpoint in checkpoints:
        checkpoint_id = checkpoint.get("id", "<missing>")
        for field in ["id", "name", "node_id", "region"]:
            if field not in checkpoint:
                errors.append(f"Checkpoint {checkpoint_id} is missing field: {field}")
                invalid_checkpoint_ids.add(str(checkpoint_id))

        node_id = checkpoint.get("node_id")
        if node_id not in node_id_set:
            errors.append(
                f"Checkpoint {checkpoint_id} references unknown node: {node_id}"
            )
            invalid_checkpoint_ids.add(str(checkpoint_id))
        if "id" in checkpoint and not isinstance(checkpoint["id"], str):
            errors.append(f"Checkpoint {checkpoint_id} id must be a string")
            invalid_checkpoint_ids.add(str(checkpoint_id))
        if "name" in checkpoint and not isinstance(checkpoint["name"], str):
            errors.append(f"Checkpoint {checkpoint_id} name must be a string")
            invalid_checkpoint_ids.add(str(checkpoint_id))
        if "node_id" in checkpoint and not isinstance(checkpoint["node_id"], str):
            errors.append(f"Checkpoint {checkpoint_id} node_id must be a string")
            invalid_checkpoint_ids.add(str(checkpoint_id))
        if "region" in checkpoint and checkpoint["region"] not in CHECKPOINT_REGIONS:
            errors.append(
                f"Checkpoint {checkpoint_id} region is invalid: {checkpoint['region']}"
            )
            invalid_checkpoint_ids.add(str(checkpoint_id))

    isolated_selectable_ids = sorted(
        node["id"]
        for node in selectable_nodes
        if node["id"] not in connected_node_ids
    )
    if isolated_selectable_ids:
        warnings.append(f"Selectable nodes without any edge: {isolated_selectable_ids}")

    unreachable_pairs: list[dict[str, str]] = []
    if not errors:
        unreachable_pairs = unreachable_selectable_pairs(
            [node["id"] for node in selectable_nodes],
            edges,
            node_id_set,
        )
        if unreachable_pairs:
            errors.append(
                "Some selectable node pairs are not mutually reachable: "
                f"{unreachable_pairs[:20]}"
            )

    sorted_out_of_bounds_node_ids = sorted(out_of_bounds_node_ids)
    sorted_invalid_node_ids = sorted(invalid_node_ids)
    sorted_invalid_edge_ids = sorted(invalid_edge_ids)
    sorted_invalid_checkpoint_ids = sorted(invalid_checkpoint_ids)
    suggestions = fix_suggestions(
        errors=errors,
        warnings=warnings,
        unreachable_pairs=unreachable_pairs,
        out_of_bounds_node_ids=sorted_out_of_bounds_node_ids,
        invalid_node_ids=sorted_invalid_node_ids,
        invalid_edge_ids=sorted_invalid_edge_ids,
        invalid_checkpoint_ids=sorted_invalid_checkpoint_ids,
    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "checkpoint_count": len(checkpoints),
        "selectable_node_count": len(selectable_nodes),
        "unreachable_route_pairs": unreachable_pairs,
        "out_of_bounds_node_ids": sorted_out_of_bounds_node_ids,
        "invalid_node_ids": sorted_invalid_node_ids,
        "invalid_edge_ids": sorted_invalid_edge_ids,
        "invalid_checkpoint_ids": sorted_invalid_checkpoint_ids,
        "fix_suggestions": suggestions,
        "quality_summary": quality_summary(
            errors=errors,
            warnings=warnings,
            unreachable_pairs=unreachable_pairs,
            out_of_bounds_node_ids=sorted_out_of_bounds_node_ids,
            invalid_node_ids=sorted_invalid_node_ids,
            invalid_edge_ids=sorted_invalid_edge_ids,
            invalid_checkpoint_ids=sorted_invalid_checkpoint_ids,
            fix_suggestions=suggestions,
        ),
    }


def validate_map_id(map_id: str):
    if not MAP_ID_PATTERN.fullmatch(map_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "map id must start with a lowercase letter and contain only "
                "lowercase letters, numbers, underscores, or hyphens."
            ),
        )


def map_file_path(map_id: str) -> Path:
    validate_map_id(map_id)
    map_path = (MAPS_DIR / f"{map_id}.json").resolve()
    maps_dir = MAPS_DIR.resolve()

    if map_path.parent != maps_dir:
        raise HTTPException(status_code=400, detail="Invalid map path.")

    return map_path


def persistable_map_data(venue_map: dict[str, Any]) -> dict[str, Any]:
    data = {
        "id": venue_map["id"],
        "name": venue_map["name"],
        "image": venue_map["image"],
        "width": venue_map["width"],
        "height": venue_map["height"],
        "nodes": venue_map["nodes"],
        "checkpoints": venue_map.get("checkpoints", []),
        "edges": venue_map["edges"],
    }
    return data


def save_map_data(venue_map: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    map_id = venue_map.get("id")
    if not isinstance(map_id, str):
        raise HTTPException(status_code=400, detail="Map id is required.")

    validate_map_id(map_id)
    if map_id in PROTECTED_MAP_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Protected map cannot be overwritten through this API: {map_id}",
        )

    validation = validate_map_data(venue_map)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Map data is invalid.",
                "validation": {"map_id": map_id, **validation},
            },
        )

    map_path = map_file_path(map_id)
    existed_before = map_path.exists()
    if existed_before and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Map already exists: {map_id}",
        )

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", encoding="utf-8") as file:
        json.dump(persistable_map_data(venue_map), file, ensure_ascii=False, indent=2)
        file.write("\n")

    return {
        "map_id": map_id,
        "saved": True,
        "overwritten": existed_before and overwrite,
        "path": str(map_path),
        "validation": {"map_id": map_id, **validation},
    }


def delete_map_data(map_id: str) -> dict[str, Any]:
    validate_map_id(map_id)
    if map_id in PROTECTED_MAP_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Protected map cannot be deleted through this API: {map_id}",
        )

    map_path = map_file_path(map_id)
    if not map_path.exists():
        raise HTTPException(status_code=404, detail=f"Map not found: {map_id}")

    map_path.unlink()
    return {
        "map_id": map_id,
        "deleted": True,
        "path": str(map_path),
    }


def list_maps() -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    for map_path in sorted(MAPS_DIR.glob("*.json")):
        with map_path.open("r", encoding="utf-8") as file:
            venue_map = json.load(file)
        validation = validate_map_data(venue_map)

        maps.append(
            {
                "id": venue_map["id"],
                "name": venue_map["name"],
                "image": venue_map["image"],
                "width": venue_map["width"],
                "height": venue_map["height"],
                "node_count": validation.get("node_count", 0),
                "edge_count": validation.get("edge_count", 0),
                "checkpoint_count": validation.get("checkpoint_count", 0),
                "selectable_node_count": validation.get("selectable_node_count", 0),
                "valid": validation["valid"],
            }
        )
    return maps


def load_map(map_id: str) -> dict[str, Any]:
    map_path = map_file_path(map_id)
    if not map_path.exists():
        raise HTTPException(status_code=404, detail=f"Map not found: {map_id}")

    with map_path.open("r", encoding="utf-8") as file:
        venue_map = json.load(file)

    validation = validate_map_data(venue_map)
    if not validation["valid"]:
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Map data is invalid: {map_id}",
                "errors": validation["errors"],
                "warnings": validation["warnings"],
            },
        )

    venue_map["selectable_nodes"] = [
        node for node in venue_map["nodes"] if node.get("selectable", False)
    ]
    venue_map["checkpoints"] = venue_map.get("checkpoints", [])
    return venue_map


def validate_map(map_id: str) -> dict[str, Any]:
    map_path = map_file_path(map_id)
    if not map_path.exists():
        raise HTTPException(status_code=404, detail=f"Map not found: {map_id}")

    with map_path.open("r", encoding="utf-8") as file:
        venue_map = json.load(file)

    validation = validate_map_data(venue_map)
    return {
        "map_id": venue_map.get("id", map_id),
        **validation,
    }
