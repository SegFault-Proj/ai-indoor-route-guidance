import type {
  MapEdge,
  MapNode,
  PredictedEdgeWeight,
  RouteResult,
} from "./types";

interface Neighbor {
  nodeId: string;
  edgeId: string;
  distance: number;
  multiplier: number;
}

interface QueueEntry {
  nodeId: string;
  cost: number;
}

class MinHeap {
  private values: QueueEntry[] = [];

  push(entry: QueueEntry): void {
    this.values.push(entry);
    let index = this.values.length - 1;

    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);

      if (this.values[parent].cost <= this.values[index].cost) {
        break;
      }

      [this.values[parent], this.values[index]] = [
        this.values[index],
        this.values[parent],
      ];
      index = parent;
    }
  }

  pop(): QueueEntry | undefined {
    if (this.values.length === 0) {
      return undefined;
    }

    const first = this.values[0];
    const last = this.values.pop();

    if (!last || this.values.length === 0) {
      return first;
    }

    this.values[0] = last;
    let index = 0;

    while (true) {
      const left = index * 2 + 1;
      const right = index * 2 + 2;
      let smallest = index;

      if (
        left < this.values.length &&
        this.values[left].cost < this.values[smallest].cost
      ) {
        smallest = left;
      }

      if (
        right < this.values.length &&
        this.values[right].cost < this.values[smallest].cost
      ) {
        smallest = right;
      }

      if (smallest === index) {
        break;
      }

      [this.values[index], this.values[smallest]] = [
        this.values[smallest],
        this.values[index],
      ];
      index = smallest;
    }

    return first;
  }

  get size(): number {
    return this.values.length;
  }
}

export function calculateShortestRoute(
  nodes: MapNode[],
  edges: MapEdge[],
  predictions: PredictedEdgeWeight[],
  startId: string,
  destinationId: string
): RouteResult | null {
  const predictionMap = new Map(
    predictions.map((prediction) => [prediction.edgeId, prediction])
  );

  const graph = new Map<string, Neighbor[]>();
  for (const node of nodes) {
    graph.set(node.id, []);
  }

  for (const edge of edges) {
    const multiplier = predictionMap.get(edge.id)?.multiplier ?? 1;

    graph.get(edge.from)?.push({
      nodeId: edge.to,
      edgeId: edge.id,
      distance: edge.distance,
      multiplier,
    });

    if (edge.bidirectional) {
      graph.get(edge.to)?.push({
        nodeId: edge.from,
        edgeId: edge.id,
        distance: edge.distance,
        multiplier,
      });
    }
  }

  const costs: Record<string, number> = {};
  const distances: Record<string, number> = {};
  const previousNode: Record<string, string | null> = {};
  const previousEdge: Record<string, string | null> = {};

  for (const node of nodes) {
    costs[node.id] = Number.POSITIVE_INFINITY;
    distances[node.id] = Number.POSITIVE_INFINITY;
    previousNode[node.id] = null;
    previousEdge[node.id] = null;
  }

  costs[startId] = 0;
  distances[startId] = 0;

  const heap = new MinHeap();
  heap.push({ nodeId: startId, cost: 0 });

  while (heap.size > 0) {
    const current = heap.pop();
    if (!current) break;

    if (current.cost > costs[current.nodeId]) continue;
    if (current.nodeId === destinationId) break;

    for (const neighbor of graph.get(current.nodeId) ?? []) {
      const nextCost =
        costs[current.nodeId] + neighbor.distance * neighbor.multiplier;
      const nextDistance =
        distances[current.nodeId] + neighbor.distance;

      if (nextCost < costs[neighbor.nodeId]) {
        costs[neighbor.nodeId] = nextCost;
        distances[neighbor.nodeId] = nextDistance;
        previousNode[neighbor.nodeId] = current.nodeId;
        previousEdge[neighbor.nodeId] = neighbor.edgeId;

        heap.push({
          nodeId: neighbor.nodeId,
          cost: nextCost,
        });
      }
    }
  }

  if (!Number.isFinite(costs[destinationId])) {
    return null;
  }

  const path: string[] = [];
  const edgeIds: string[] = [];

  let currentNode: string | null = destinationId;
  while (currentNode !== null) {
    path.unshift(currentNode);

    const edgeId = previousEdge[currentNode];
    if (edgeId !== null) {
      edgeIds.unshift(edgeId);
    }

    currentNode = previousNode[currentNode];
  }

  return {
    path,
    edgeIds,
    totalDistance: Number(distances[destinationId].toFixed(1)),
    weightedCost: Number(costs[destinationId].toFixed(1)),
  };
}
