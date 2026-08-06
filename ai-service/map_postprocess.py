import math
import re
from copy import deepcopy
from typing import Any

from map_store import CROWD_REGIONS, NODE_TYPES, ZONES, validate_map_data

ID_TOKEN_PATTERN = re.compile(r"[^A-Z0-9_]+")


def normalize_identifier(value: Any, fallback_prefix: str, index: int) -> str:
    if isinstance(value, str) and value.strip():
        base = value.strip().upper()
    else:
        base = f"{fallback_prefix}_{index + 1}"

    base = base.replace("-", "_").replace(" ", "_")
    base = ID_TOKEN_PATTERN.sub("_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = f"{fallback_prefix}_{index + 1}"
    if base[0].isdigit():
        base = f"{fallback_prefix}_{base}"
    return base


def unique_identifier(base_id: str, used_ids: set[str]) -> str:
    candidate = base_id
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base_id}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def coerce_number(value: Any, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def nearest_allowed(value: Any, allowed_values: set[str], fallback: str) -> str:
    if isinstance(value, str) and value in allowed_values:
        return value
    return fallback


def infer_node_type(node: dict[str, Any]) -> str:
    raw_text = f"{node.get('id', '')} {node.get('name', '')}".lower()
    if "entrance" in raw_text or "entry" in raw_text or "입구" in raw_text:
        return "entrance"
    if "exit" in raw_text or "출구" in raw_text:
        return "exit"
    if "booth" in raw_text or "부스" in raw_text:
        return "booth"
    if "info" in raw_text or "desk" in raw_text or "안내" in raw_text:
        return "facility"
    return "junction"


def infer_edge_zone(edge: dict[str, Any], node_lookup: dict[str, dict[str, Any]]) -> str:
    from_node = node_lookup.get(edge.get("from"), {})
    to_node = node_lookup.get(edge.get("to"), {})
    raw_text = " ".join(
        [
            str(edge.get("id", "")),
            str(edge.get("zone", "")),
            str(from_node.get("name", "")),
            str(to_node.get("name", "")),
        ]
    ).lower()
    if "booth" in raw_text or "부스" in raw_text:
        return "booth"
    if "gate" in raw_text or "입구" in raw_text or "출구" in raw_text:
        return "gate"
    if "facility" in raw_text or "안내" in raw_text:
        return "facility"
    return "lobby"


def crowd_region_for_zone(zone: str) -> str:
    if zone == "booth":
        return "booth"
    if zone == "gate":
        return "west"
    if zone == "facility":
        return "central"
    return "central"


def euclidean_distance(from_node: dict[str, Any], to_node: dict[str, Any]) -> float:
    dx = float(from_node["x"]) - float(to_node["x"])
    dy = float(from_node["y"]) - float(to_node["y"])
    return round(math.hypot(dx, dy) / 10, 1)


def close_node_pairs(nodes: list[dict[str, Any]], threshold_px: float) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index, left_node in enumerate(nodes):
        for right_node in nodes[left_index + 1:]:
            distance = math.hypot(
                float(left_node["x"]) - float(right_node["x"]),
                float(left_node["y"]) - float(right_node["y"]),
            )
            if distance <= threshold_px:
                pairs.append(
                    {
                        "node_ids": [left_node["id"], right_node["id"]],
                        "distance_px": round(distance, 1),
                    }
                )
    return pairs


def weak_label_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weak_nodes: list[dict[str, Any]] = []
    seen_names: dict[str, int] = {}
    for node in nodes:
        name = str(node.get("name", "")).strip()
        normalized_name = name.lower()
        seen_names[normalized_name] = seen_names.get(normalized_name, 0) + 1
        generic_id = re.fullmatch(r"(NODE|POINT|JUNCTION)_?\d+", node["id"], re.I)
        generic_name = re.fullmatch(
            r"(node|point|junction|area|place|unknown)[ _-]?\d*",
            normalized_name,
        )
        if not name or generic_id or generic_name:
            weak_nodes.append(
                {
                    "node_id": node["id"],
                    "name": name,
                    "reason": "generic_or_missing_label",
                }
            )

    for node in nodes:
        name = str(node.get("name", "")).strip()
        if name and seen_names.get(name.lower(), 0) > 1:
            weak_nodes.append(
                {
                    "node_id": node["id"],
                    "name": name,
                    "reason": "duplicate_label",
                }
            )

    return weak_nodes


def graph_components(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[list[str]]:
    node_ids = {node["id"] for node in nodes}
    graph: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id not in node_ids or to_id not in node_ids:
            continue
        graph[from_id].add(to_id)
        graph[to_id].add(from_id)

    components: list[list[str]] = []
    visited: set[str] = set()
    for node_id in sorted(node_ids):
        if node_id in visited:
            continue
        stack = [node_id]
        component: list[str] = []
        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            component.append(current_id)
            stack.extend(sorted(graph[current_id] - visited))
        components.append(sorted(component))
    return components


def nearest_component_edge_candidates(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    max_candidates: int,
    max_distance_px: float,
) -> list[dict[str, Any]]:
    node_lookup = {node["id"]: node for node in nodes}
    components = graph_components(nodes, edges)
    if len(components) <= 1:
        return []

    candidates: list[dict[str, Any]] = []
    for left_index, left_component in enumerate(components):
        for right_component in components[left_index + 1:]:
            best_pair: dict[str, Any] | None = None
            best_distance = math.inf
            for left_id in left_component:
                for right_id in right_component:
                    left_node = node_lookup[left_id]
                    right_node = node_lookup[right_id]
                    distance_px = math.hypot(
                        float(left_node["x"]) - float(right_node["x"]),
                        float(left_node["y"]) - float(right_node["y"]),
                    )
                    if distance_px < best_distance:
                        best_distance = distance_px
                        best_pair = {
                            "from": left_id,
                            "to": right_id,
                            "distance_px": round(distance_px, 1),
                        }
            if best_pair and best_distance <= max_distance_px:
                candidates.append(best_pair)

    candidates.sort(key=lambda candidate: candidate["distance_px"])
    return candidates[:max_candidates]


def candidate_edge(
    candidate: dict[str, Any],
    node_lookup: dict[str, dict[str, Any]],
    used_edge_ids: set[str],
) -> dict[str, Any]:
    from_id = candidate["from"]
    to_id = candidate["to"]
    base_id = normalize_identifier(f"E_{from_id}_{to_id}", "EDGE", len(used_edge_ids))
    edge_id = unique_identifier(base_id, used_edge_ids)
    zone = infer_edge_zone({"from": from_id, "to": to_id}, node_lookup)
    return {
        "id": edge_id,
        "from": from_id,
        "to": to_id,
        "distance": euclidean_distance(node_lookup[from_id], node_lookup[to_id]),
        "widthM": 2.5,
        "zone": zone,
        "crowdRegion": crowd_region_for_zone(zone),
        "bidirectional": True,
    }


def postprocess_map_data(
    venue_map: dict[str, Any],
    *,
    recalculate_edge_distance: bool = True,
    clamp_coordinates: bool = True,
    close_node_threshold_px: float = 18,
    suggest_connection_edges: bool = True,
    apply_connection_suggestions: bool = False,
    max_connection_suggestions: int = 5,
    max_connection_distance_px: float = 260,
) -> dict[str, Any]:
    processed_map = deepcopy(venue_map)
    changes: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []

    width = coerce_number(processed_map.get("width"), 1)
    height = coerce_number(processed_map.get("height"), 1)
    processed_map["width"] = max(1, width)
    processed_map["height"] = max(1, height)
    processed_map["nodes"] = processed_map.get("nodes") if isinstance(processed_map.get("nodes"), list) else []
    processed_map["edges"] = processed_map.get("edges") if isinstance(processed_map.get("edges"), list) else []
    processed_map["checkpoints"] = (
        processed_map.get("checkpoints")
        if isinstance(processed_map.get("checkpoints"), list)
        else []
    )

    node_id_map: dict[Any, str] = {}
    used_node_ids: set[str] = set()
    normalized_nodes: list[dict[str, Any]] = []
    for index, raw_node in enumerate(processed_map["nodes"]):
        if not isinstance(raw_node, dict):
            changes.append(
                {
                    "code": "drop_invalid_node",
                    "message": "Dropped a non-object node entry.",
                    "index": index,
                }
            )
            continue

        node = dict(raw_node)
        original_id = node.get("id")
        normalized_id = unique_identifier(
            normalize_identifier(original_id, "NODE", index),
            used_node_ids,
        )
        if original_id != normalized_id:
            changes.append(
                {
                    "code": "normalize_node_id",
                    "from": original_id,
                    "to": normalized_id,
                }
            )
        node_id_map[original_id] = normalized_id
        node["id"] = normalized_id
        node["name"] = str(node.get("name") or normalized_id)
        node["x"] = coerce_number(node.get("x"), 0)
        node["y"] = coerce_number(node.get("y"), 0)
        if clamp_coordinates:
            clamped_x = clamp(node["x"], 0, processed_map["width"])
            clamped_y = clamp(node["y"], 0, processed_map["height"])
            if clamped_x != node["x"] or clamped_y != node["y"]:
                changes.append(
                    {
                        "code": "clamp_node_coordinates",
                        "node_id": normalized_id,
                        "from": {"x": node["x"], "y": node["y"]},
                        "to": {"x": clamped_x, "y": clamped_y},
                    }
                )
            node["x"] = clamped_x
            node["y"] = clamped_y
        node["type"] = nearest_allowed(
            node.get("type"),
            NODE_TYPES,
            infer_node_type(node),
        )
        node["selectable"] = bool(node.get("selectable", False))
        normalized_nodes.append(node)

    processed_map["nodes"] = normalized_nodes
    node_lookup = {node["id"]: node for node in processed_map["nodes"]}

    used_edge_ids: set[str] = set()
    normalized_edges: list[dict[str, Any]] = []
    for index, raw_edge in enumerate(processed_map["edges"]):
        if not isinstance(raw_edge, dict):
            changes.append(
                {
                    "code": "drop_invalid_edge",
                    "message": "Dropped a non-object edge entry.",
                    "index": index,
                }
            )
            continue

        edge = dict(raw_edge)
        edge["from"] = node_id_map.get(edge.get("from"), edge.get("from"))
        edge["to"] = node_id_map.get(edge.get("to"), edge.get("to"))
        if edge.get("from") not in node_lookup or edge.get("to") not in node_lookup:
            suggestions.append(
                {
                    "code": "edge_references_missing_node",
                    "edge_id": edge.get("id"),
                    "from": edge.get("from"),
                    "to": edge.get("to"),
                    "message": "Create the missing node or remove this edge.",
                }
            )

        original_id = edge.get("id")
        fallback_edge_id = f"E_{edge.get('from', 'FROM')}_{edge.get('to', 'TO')}"
        normalized_id = unique_identifier(
            normalize_identifier(original_id or fallback_edge_id, "EDGE", index),
            used_edge_ids,
        )
        if original_id != normalized_id:
            changes.append(
                {
                    "code": "normalize_edge_id",
                    "from": original_id,
                    "to": normalized_id,
                }
            )
        edge["id"] = normalized_id

        zone = nearest_allowed(
            edge.get("zone"),
            ZONES,
            infer_edge_zone(edge, node_lookup),
        )
        edge["zone"] = zone
        edge["crowdRegion"] = nearest_allowed(
            edge.get("crowdRegion"),
            CROWD_REGIONS,
            crowd_region_for_zone(zone),
        )
        edge["widthM"] = max(0.8, coerce_number(edge.get("widthM"), 2.5))
        edge["bidirectional"] = bool(edge.get("bidirectional", True))

        if (
            recalculate_edge_distance
            and edge.get("from") in node_lookup
            and edge.get("to") in node_lookup
        ):
            calculated_distance = euclidean_distance(
                node_lookup[edge["from"]],
                node_lookup[edge["to"]],
            )
            if edge.get("distance") != calculated_distance:
                changes.append(
                    {
                        "code": "recalculate_edge_distance",
                        "edge_id": normalized_id,
                        "from": edge.get("distance"),
                        "to": calculated_distance,
                    }
                )
            edge["distance"] = max(0.1, calculated_distance)
        else:
            edge["distance"] = max(0.1, coerce_number(edge.get("distance"), 1))

        normalized_edges.append(edge)

    processed_map["edges"] = normalized_edges

    used_checkpoint_ids: set[str] = set()
    normalized_checkpoints: list[dict[str, Any]] = []
    for index, raw_checkpoint in enumerate(processed_map["checkpoints"]):
        if not isinstance(raw_checkpoint, dict):
            changes.append(
                {
                    "code": "drop_invalid_checkpoint",
                    "message": "Dropped a non-object checkpoint entry.",
                    "index": index,
                }
            )
            continue

        checkpoint = dict(raw_checkpoint)
        original_id = checkpoint.get("id")
        normalized_id = unique_identifier(
            normalize_identifier(original_id, "CHECKPOINT", index),
            used_checkpoint_ids,
        )
        if original_id != normalized_id:
            changes.append(
                {
                    "code": "normalize_checkpoint_id",
                    "from": original_id,
                    "to": normalized_id,
                }
            )
        checkpoint["id"] = normalized_id
        checkpoint["name"] = str(checkpoint.get("name") or normalized_id)
        checkpoint["node_id"] = node_id_map.get(
            checkpoint.get("node_id"),
            checkpoint.get("node_id"),
        )
        if checkpoint.get("node_id") not in node_lookup:
            suggestions.append(
                {
                    "code": "checkpoint_references_missing_node",
                    "checkpoint_id": normalized_id,
                    "node_id": checkpoint.get("node_id"),
                    "message": "Point this checkpoint to an existing node.",
                }
            )
        checkpoint["region"] = "booth" if "booth" in checkpoint["name"].lower() else "lobby"
        normalized_checkpoints.append(checkpoint)

    processed_map["checkpoints"] = normalized_checkpoints

    close_pairs = close_node_pairs(processed_map["nodes"], close_node_threshold_px)
    if close_pairs:
        suggestions.append(
            {
                "code": "close_nodes_detected",
                "message": "Review these close node pairs; merge them manually if they represent the same place.",
                "pairs": close_pairs[:20],
            }
        )

    weak_labels = weak_label_nodes(processed_map["nodes"])
    if weak_labels:
        suggestions.append(
            {
                "code": "weak_node_labels",
                "message": (
                    "Improve generic, missing, or duplicated node names before using "
                    "the map for spoken or text navigation."
                ),
                "nodes": weak_labels[:30],
            }
        )

    if suggest_connection_edges:
        used_edge_ids = {edge["id"] for edge in processed_map["edges"]}
        connection_candidates = nearest_component_edge_candidates(
            processed_map["nodes"],
            processed_map["edges"],
            max_candidates=max_connection_suggestions,
            max_distance_px=max_connection_distance_px,
        )
        if connection_candidates:
            candidate_edges = [
                candidate_edge(candidate, node_lookup, used_edge_ids)
                for candidate in connection_candidates
            ]
            suggestions.append(
                {
                    "code": "connect_disconnected_components",
                    "message": (
                        "Review these candidate edges for disconnected map components. "
                        "Apply only when the straight segment is a walkable corridor."
                    ),
                    "candidate_edges": candidate_edges,
                }
            )
            if apply_connection_suggestions:
                for edge in candidate_edges:
                    processed_map["edges"].append(edge)
                    changes.append(
                        {
                            "code": "add_connection_edge",
                            "edge_id": edge["id"],
                            "from": edge["from"],
                            "to": edge["to"],
                        }
                    )

    validation = validate_map_data(processed_map)
    quality_summary = dict(validation["quality_summary"])
    postprocess_reason_codes = list(quality_summary["reason_codes"])
    postprocess_penalty = 0
    suggestion_codes = {suggestion["code"] for suggestion in suggestions}
    if "close_nodes_detected" in suggestion_codes:
        postprocess_penalty += 5
        postprocess_reason_codes.append("close_nodes_detected")
    if "connect_disconnected_components" in suggestion_codes and not apply_connection_suggestions:
        postprocess_penalty += 10
        postprocess_reason_codes.append("connection_edges_need_review")
    if "edge_references_missing_node" in suggestion_codes:
        postprocess_penalty += 20
        postprocess_reason_codes.append("edge_references_missing_node")
    if "checkpoint_references_missing_node" in suggestion_codes:
        postprocess_penalty += 15
        postprocess_reason_codes.append("checkpoint_references_missing_node")
    if "weak_node_labels" in suggestion_codes:
        postprocess_penalty += min(15, len(weak_labels) * 3)
        postprocess_reason_codes.append("weak_node_labels")

    if postprocess_penalty:
        quality_summary["score"] = max(0, quality_summary["score"] - postprocess_penalty)
        quality_summary["reason_codes"] = postprocess_reason_codes
        if quality_summary["readiness"] == "ready":
            quality_summary["readiness"] = "needs_review"
        if (
            "edge_references_missing_node" in suggestion_codes
            or "checkpoint_references_missing_node" in suggestion_codes
        ):
            quality_summary["readiness"] = "blocked"
            quality_summary["can_save"] = False

    validation["quality_summary"] = quality_summary
    return {
        "processed_map": processed_map,
        "changes": changes,
        "suggestions": suggestions,
        "validation": {"map_id": processed_map.get("id"), **validation},
    }
