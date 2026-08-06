import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapNode, RouteResult } from "./types";

export type GuidanceStatus = "idle" | "running" | "paused" | "arrived";

export interface AnimatedPosition {
  x: number;
  y: number;
}

interface RoutePoint extends AnimatedPosition {
  nodeId: string;
}

interface UseRouteAnimationResult {
  position: AnimatedPosition | null;
  status: GuidanceStatus;
  progressPercent: number;
  currentSegmentIndex: number;
  currentTargetNodeId: string | null;
  start: () => void;
  pause: () => void;
  reset: () => void;
}

/**
 * SVG 좌표를 기준으로 계산된 경로 위에서 캐릭터를 이동시킵니다.
 * 실제 위치 추적이 아니라 해커톤 시연용 경로 애니메이션입니다.
 */
export function useRouteAnimation(
  route: RouteResult | null,
  nodes: MapNode[],
  speedSvgUnitsPerSecond = 92
): UseRouteAnimationResult {
  const nodeMap = useMemo(
    () => new Map(nodes.map((node) => [node.id, node])),
    [nodes]
  );

  const points = useMemo<RoutePoint[]>(() => {
    if (!route) return [];

    return route.path
      .map((nodeId) => {
        const node = nodeMap.get(nodeId);
        return node ? { nodeId, x: node.x, y: node.y } : null;
      })
      .filter((point): point is RoutePoint => point !== null);
  }, [nodeMap, route]);

  const segmentLengths = useMemo(() => {
    return points.slice(0, -1).map((point, index) => {
      const next = points[index + 1];
      return Math.hypot(next.x - point.x, next.y - point.y);
    });
  }, [points]);

  const totalPixelDistance = useMemo(
    () => segmentLengths.reduce((sum, length) => sum + length, 0),
    [segmentLengths]
  );

  const [position, setPosition] = useState<AnimatedPosition | null>(null);
  const [status, setStatus] = useState<GuidanceStatus>("idle");
  const [progressPercent, setProgressPercent] = useState(0);
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);

  const segmentIndexRef = useRef(0);
  const segmentTravelRef = useRef(0);
  const completedDistanceRef = useRef(0);
  const frameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const positionRef = useRef<AnimatedPosition | null>(null);

  const cancelFrame = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    lastTimeRef.current = null;
  }, []);

  const reset = useCallback(() => {
    cancelFrame();
    segmentIndexRef.current = 0;
    segmentTravelRef.current = 0;
    completedDistanceRef.current = 0;
    setCurrentSegmentIndex(0);
    setProgressPercent(0);

    if (points.length === 0) {
      positionRef.current = null;
      setPosition(null);
      setStatus("idle");
      return;
    }

    const startPosition = { x: points[0].x, y: points[0].y };
    positionRef.current = startPosition;
    setPosition(startPosition);
    setStatus(points.length === 1 ? "arrived" : "paused");
  }, [cancelFrame, points]);

  const start = useCallback(() => {
    if (points.length <= 1) {
      setStatus(points.length === 1 ? "arrived" : "idle");
      return;
    }

    if (segmentIndexRef.current >= points.length - 1) {
      segmentIndexRef.current = 0;
      segmentTravelRef.current = 0;
      completedDistanceRef.current = 0;
      setCurrentSegmentIndex(0);
      setProgressPercent(0);
      const startPosition = { x: points[0].x, y: points[0].y };
      positionRef.current = startPosition;
      setPosition(startPosition);
    }

    lastTimeRef.current = null;
    setStatus("running");
  }, [points]);

  const pause = useCallback(() => {
    cancelFrame();
    setStatus((current) => (current === "running" ? "paused" : current));
  }, [cancelFrame]);

  // 경로가 바뀌면 시작점으로 초기화한 뒤 자동 안내를 시작합니다.
  useEffect(() => {
    reset();

    if (points.length > 1) {
      const timer = window.setTimeout(() => setStatus("running"), 180);
      return () => window.clearTimeout(timer);
    }

    return undefined;
  }, [points, reset]);

  useEffect(() => {
    if (status !== "running" || points.length <= 1) {
      return undefined;
    }

    const animate = (timestamp: number) => {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = timestamp;
        frameRef.current = requestAnimationFrame(animate);
        return;
      }

      // 브라우저 탭 복귀 시 캐릭터가 순간이동하지 않도록 시간 차를 제한합니다.
      const deltaSeconds = Math.min(
        (timestamp - lastTimeRef.current) / 1000,
        0.06
      );
      lastTimeRef.current = timestamp;

      let movementLeft = speedSvgUnitsPerSecond * deltaSeconds;
      let segmentIndex = segmentIndexRef.current;
      let segmentTravel = segmentTravelRef.current;
      let completedDistance = completedDistanceRef.current;
      let nextPosition = positionRef.current ?? { x: points[0].x, y: points[0].y };
      let arrived = false;

      while (movementLeft > 0 && segmentIndex < points.length - 1) {
        const from = points[segmentIndex];
        const to = points[segmentIndex + 1];
        const segmentLength = segmentLengths[segmentIndex];

        if (segmentLength <= 0) {
          segmentIndex += 1;
          continue;
        }

        const availableDistance = segmentLength - segmentTravel;

        if (movementLeft >= availableDistance) {
          movementLeft -= availableDistance;
          completedDistance += availableDistance;
          segmentIndex += 1;
          segmentTravel = 0;
          nextPosition = { x: to.x, y: to.y };

          if (segmentIndex >= points.length - 1) {
            arrived = true;
            break;
          }
        } else {
          segmentTravel += movementLeft;
          completedDistance += movementLeft;
          movementLeft = 0;

          const ratio = segmentTravel / segmentLength;
          nextPosition = {
            x: from.x + (to.x - from.x) * ratio,
            y: from.y + (to.y - from.y) * ratio,
          };
        }
      }

      segmentIndexRef.current = segmentIndex;
      segmentTravelRef.current = segmentTravel;
      completedDistanceRef.current = completedDistance;
      setCurrentSegmentIndex(segmentIndex);
      positionRef.current = nextPosition;
      setPosition(nextPosition);
      setProgressPercent(
        totalPixelDistance > 0
          ? Math.min(100, (completedDistance / totalPixelDistance) * 100)
          : 100
      );

      if (arrived) {
        setProgressPercent(100);
        setStatus("arrived");
        frameRef.current = null;
        return;
      }

      frameRef.current = requestAnimationFrame(animate);
    };

    frameRef.current = requestAnimationFrame(animate);

    return cancelFrame;
  }, [
    cancelFrame,
    points,
    segmentLengths,
    speedSvgUnitsPerSecond,
    status,
    totalPixelDistance,
  ]);

  const currentTargetNodeId =
    points.length > 1
      ? points[Math.min(currentSegmentIndex + 1, points.length - 1)]?.nodeId ??
        null
      : points[0]?.nodeId ?? null;

  return {
    position,
    status,
    progressPercent,
    currentSegmentIndex,
    currentTargetNodeId,
    start,
    pause,
    reset,
  };
}
