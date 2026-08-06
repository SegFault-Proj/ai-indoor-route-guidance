from heapq import heappop, heappush
import math
from typing import Any


def calculate_shortest_route(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    multipliers: dict[str, float],
    start_id: str,
    destination_id: str,
    *,
    blocked_edge_ids: set[str] | None = None,
    turn_penalty: float = 0.0,
    algorithm: str = "astar",
) -> dict[str, Any] | None:
    node_ids = {node["id"] for node in nodes}
    if start_id not in node_ids or destination_id not in node_ids:
        return None

    blocked_edge_ids = blocked_edge_ids or set()
    node_lookup = {node["id"]: node for node in nodes}
    graph = build_graph(node_ids, edges, multipliers, blocked_edge_ids)
    heuristic = build_astar_heuristic(
        node_lookup,
        edges,
        multipliers,
        destination_id,
        blocked_edge_ids,
    )
    use_astar = algorithm == "astar"

    costs: dict[tuple[str, str | None], float] = {(start_id, None): 0.0}
    distances: dict[tuple[str, str | None], float] = {(start_id, None): 0.0}
    previous_state: dict[
        tuple[str, str | None],
        tuple[tuple[str, str | None], str] | None,
    ] = {(start_id, None): None}
    queue_index = 0
    initial_priority = heuristic(start_id) if use_astar else 0.0
    queue: list[tuple[float, float, int, str, str | None]] = [
        (initial_priority, 0.0, queue_index, start_id, None)
    ]
    best_destination_state: tuple[str, str | None] | None = None
    expanded_state_count = 0

    while queue:
        _priority, queued_cost, _queue_index, current_id, current_edge_id = heappop(queue)
        current_state = (current_id, current_edge_id)
        if queued_cost > costs.get(current_state, float("inf")):
            continue
        expanded_state_count += 1
        if current_id == destination_id:
            best_destination_state = current_state
            break

        for neighbor in graph[current_id]:
            next_id = neighbor["node_id"]
            next_edge_id = neighbor["edge_id"]
            next_state = (next_id, next_edge_id)
            turn_cost = route_turn_penalty(
                node_lookup,
                previous_state.get(current_state),
                current_id,
                next_id,
                turn_penalty,
            )
            next_cost = (
                costs[current_state]
                + neighbor["distance"] * neighbor["multiplier"]
                + turn_cost
            )
            next_distance = distances[current_state] + neighbor["distance"]

            if next_cost < costs.get(next_state, float("inf")):
                costs[next_state] = next_cost
                distances[next_state] = next_distance
                previous_state[next_state] = (current_state, next_edge_id)
                queue_index += 1
                priority = next_cost + (heuristic(next_id) if use_astar else 0.0)
                heappush(queue, (priority, next_cost, queue_index, next_id, next_edge_id))

    if best_destination_state is None:
        return None

    path: list[str] = []
    edge_ids: list[str] = []
    current_state: tuple[str, str | None] | None = best_destination_state

    while current_state is not None:
        path.insert(0, current_state[0])
        previous = previous_state[current_state]
        if previous is None:
            break
        current_state, edge_id = previous
        edge_ids.insert(0, edge_id)

    return {
        "path": path,
        "edge_ids": edge_ids,
        "total_distance": round(distances[best_destination_state], 1),
        "weighted_cost": round(costs[best_destination_state], 1),
        "algorithm": "astar" if use_astar else "dijkstra",
        "expanded_state_count": expanded_state_count,
    }


def build_graph(
    node_ids: set[str],
    edges: list[dict[str, Any]],
    multipliers: dict[str, float],
    blocked_edge_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    graph: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_ids}

    for edge in edges:
        if edge["id"] in blocked_edge_ids:
            continue
        multiplier = multipliers.get(edge["id"], 1.0)
        neighbor = {
            "node_id": edge["to"],
            "edge_id": edge["id"],
            "distance": float(edge["distance"]),
            "multiplier": multiplier,
        }
        graph[edge["from"]].append(neighbor)

        if edge.get("bidirectional", False):
            graph[edge["to"]].append(
                {
                    "node_id": edge["from"],
                    "edge_id": edge["id"],
                    "distance": float(edge["distance"]),
                    "multiplier": multiplier,
                }
            )

    return graph


def build_astar_heuristic(
    node_lookup: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    multipliers: dict[str, float],
    destination_id: str,
    blocked_edge_ids: set[str],
):
    destination = node_lookup[destination_id]
    min_distance_per_pixel: float | None = None
    min_multiplier: float | None = None

    for edge in edges:
        if edge["id"] in blocked_edge_ids:
            continue
        start = node_lookup.get(edge["from"])
        end = node_lookup.get(edge["to"])
        if start is None or end is None:
            continue
        pixel_length = math.hypot(
            float(end["x"]) - float(start["x"]),
            float(end["y"]) - float(start["y"]),
        )
        if pixel_length <= 0:
            continue
        distance_per_pixel = float(edge["distance"]) / pixel_length
        min_distance_per_pixel = (
            distance_per_pixel
            if min_distance_per_pixel is None
            else min(min_distance_per_pixel, distance_per_pixel)
        )
        multiplier = max(0.0, float(multipliers.get(edge["id"], 1.0)))
        min_multiplier = multiplier if min_multiplier is None else min(min_multiplier, multiplier)

    if min_distance_per_pixel is None or min_multiplier is None:
        return lambda _node_id: 0.0

    scale = min_distance_per_pixel * min_multiplier

    def heuristic(node_id: str) -> float:
        node = node_lookup[node_id]
        return math.hypot(
            float(destination["x"]) - float(node["x"]),
            float(destination["y"]) - float(node["y"]),
        ) * scale

    return heuristic


def calculate_route_alternatives(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    multipliers: dict[str, float],
    start_id: str,
    destination_id: str,
    *,
    max_routes: int = 3,
    overlap_penalty: float = 1.8,
    blocked_edge_ids: set[str] | None = None,
    turn_penalty: float = 0.0,
    algorithm: str = "astar",
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, ...]] = set()
    penalized_multipliers = dict(multipliers)

    for _index in range(max_routes * 3):
        route = calculate_shortest_route(
            nodes,
            edges,
            penalized_multipliers,
            start_id,
            destination_id,
            blocked_edge_ids=blocked_edge_ids,
            turn_penalty=turn_penalty,
            algorithm=algorithm,
        )
        if route is None:
            break

        path_key = tuple(route["path"])
        if path_key not in seen_paths:
            seen_paths.add(path_key)
            if routes:
                annotate_alternative_quality(route, routes[0])
            else:
                route["overlap_ratio"] = 1.0
                route["detour_ratio"] = 1.0
            routes.append(route)
            if len(routes) >= max_routes:
                break

        for edge_id in route["edge_ids"]:
            current_multiplier = penalized_multipliers.get(edge_id, 1.0)
            penalized_multipliers[edge_id] = current_multiplier * overlap_penalty

    return routes


def annotate_alternative_quality(
    route: dict[str, Any],
    best_route: dict[str, Any],
) -> None:
    route_edge_ids = set(route["edge_ids"])
    best_edge_ids = set(best_route["edge_ids"])
    overlap_count = len(route_edge_ids & best_edge_ids)
    route["overlap_ratio"] = round(
        overlap_count / max(len(best_edge_ids), 1),
        3,
    )
    route["detour_ratio"] = round(
        float(route["weighted_cost"]) / max(float(best_route["weighted_cost"]), 0.001),
        3,
    )


def route_turn_penalty(
    node_lookup: dict[str, dict[str, Any]],
    previous: tuple[tuple[str, str | None], str] | None,
    current_id: str,
    next_id: str,
    turn_penalty: float,
) -> float:
    if turn_penalty <= 0 or previous is None:
        return 0.0

    previous_state, _edge_id = previous
    previous_id = previous_state[0]
    if previous_id == current_id:
        return 0.0

    previous_node = node_lookup[previous_id]
    current_node = node_lookup[current_id]
    next_node = node_lookup[next_id]
    incoming = (
        float(current_node["x"]) - float(previous_node["x"]),
        float(current_node["y"]) - float(previous_node["y"]),
    )
    outgoing = (
        float(next_node["x"]) - float(current_node["x"]),
        float(next_node["y"]) - float(current_node["y"]),
    )
    incoming_length = math.hypot(*incoming)
    outgoing_length = math.hypot(*outgoing)
    if incoming_length == 0 or outgoing_length == 0:
        return 0.0

    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    cosine = max(-1.0, min(1.0, dot / (incoming_length * outgoing_length)))
    angle = math.degrees(math.acos(cosine))
    if angle < 35:
        return 0.0
    if angle < 80:
        return turn_penalty * 0.5
    return turn_penalty
