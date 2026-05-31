import { useEffect, useRef, useState } from "react";
import type { ProgressEvent } from "@/types";
import { API_BASE } from "@/lib/api";

interface JobProgressState {
  /** Latest overall progress 0..1. */
  progress: number;
  status?: string;
  /** Most recent event per branch_index (DAG jobs). */
  branches: Record<number, ProgressEvent>;
  /** Most recent event per DAG node id (canvas execution overlay). */
  nodes: Record<string, ProgressEvent>;
  connected: boolean;
  lastEvent: ProgressEvent | null;
}

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

/**
 * Subscribes to /ws/jobs/{id}/progress and tracks overall + per-branch progress.
 * Reconnects on unexpected close until the job reaches a terminal status.
 */
export function useJobProgress(jobId: string | undefined): JobProgressState {
  const [state, setState] = useState<JobProgressState>({
    progress: 0,
    branches: {},
    nodes: {},
    connected: false,
    lastEvent: null,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const doneRef = useRef(false);

  useEffect(() => {
    if (!jobId) return;
    doneRef.current = false;
    // Clear any accumulated state from a previous job before tracking this one.
    setState({ progress: 0, branches: {}, nodes: {}, connected: false, lastEvent: null });
    let retry: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      const wsBase = API_BASE.replace(/^http/, "ws");
      const ws = new WebSocket(`${wsBase}/ws/jobs/${jobId}/progress`);
      wsRef.current = ws;

      ws.onopen = () => setState((s) => ({ ...s, connected: true }));
      ws.onmessage = (e) => {
        const evt = JSON.parse(e.data) as ProgressEvent;
        setState((s) => {
          const branches = { ...s.branches };
          const nodes = { ...s.nodes };
          if (typeof evt.branch_index === "number") branches[evt.branch_index] = evt;
          if (evt.node_id) nodes[evt.node_id] = evt;
          return {
            ...s,
            // Per-node events carry their own node progress; only job-level
            // events (no node_id) move the overall job progress/status.
            progress: !evt.node_id && typeof evt.progress === "number" ? evt.progress : s.progress,
            status: !evt.node_id && evt.status ? evt.status : s.status,
            branches,
            nodes,
            lastEvent: evt,
          };
        });
        // Close only on a job-level terminal event — a single node reporting
        // "failed" must not tear down the stream before the rest arrives.
        if (!evt.node_id && evt.status && TERMINAL.has(evt.status)) {
          doneRef.current = true;
          ws.close();
        }
      };
      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (!doneRef.current) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      doneRef.current = true;
      if (retry) clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [jobId]);

  return state;
}
