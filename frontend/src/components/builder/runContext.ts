import { createContext, useContext } from "react";
import type { ProgressEvent } from "@/types";

/**
 * Per-node run state keyed by DAG/canvas node id, supplied while a canvas
 * execution is live. `null` when no run is active. Node components read their
 * own slice via {@link useNodeRun} to overlay live progress/status.
 */
export const RunProgressContext = createContext<Record<string, ProgressEvent> | null>(null);

export function useNodeRun(nodeId: string): ProgressEvent | undefined {
  return useContext(RunProgressContext)?.[nodeId];
}
