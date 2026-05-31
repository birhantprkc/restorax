import type { BuilderNodeType } from "./types";

/**
 * Semantic socket types. These refine the backend's raw Python port types
 * (`graph.py` validates by `type_hint` equality) into meaningful names so the
 * canvas can colour-code sockets and reject semantically-wrong wiring early.
 *
 * Backend Python type → semantic type:
 *   list   → "frames" (chunked frame stream) or "branches" (parallel outputs)
 *   object → "meta"   (video metadata)
 *   str    → "path"
 *   float  → "fps"
 *   None   → "any"    (pass-through, matches anything — mirrors a null type_hint)
 *
 * This mapping is a strict refinement of the backend rule: any edge the canvas
 * accepts, the backend also accepts (same-name list ports stay list↔list).
 */
export type PortType =
  | "frames"
  | "meta"
  | "branches"
  | "path"
  | "fps"
  | "any";

export interface PortDef {
  /** Port name — must match the backend node's Port(name) exactly. */
  name: string;
  type: PortType;
}

export interface NodePorts {
  inputs: PortDef[];
  outputs: PortDef[];
}

/**
 * Per-node port contract, mirroring the backend DAG nodes
 * (`restorax/dag/nodes/*.py`). Port names MUST match the backend so
 * serialised edges pass `Graph._check_port_names`.
 */
export const NODE_PORTS: Record<BuilderNodeType, NodePorts> = {
  video_input: {
    inputs: [],
    outputs: [
      { name: "chunks", type: "frames" },
      { name: "meta", type: "meta" },
    ],
  },
  video_output: {
    inputs: [
      { name: "chunks", type: "frames" },
      { name: "meta", type: "meta" },
      { name: "fps", type: "fps" },
    ],
    outputs: [],
  },
  restore: {
    inputs: [{ name: "chunks", type: "frames" }],
    outputs: [{ name: "chunks", type: "frames" }],
  },
  parallel: {
    inputs: [
      { name: "chunks", type: "frames" },
      { name: "meta", type: "meta" },
    ],
    outputs: [
      { name: "branch_outputs", type: "branches" },
      { name: "meta", type: "meta" },
    ],
  },
  merge: {
    inputs: [
      { name: "branch_outputs", type: "branches" },
      { name: "meta", type: "meta" },
    ],
    outputs: [
      { name: "chunks", type: "frames" },
      { name: "meta", type: "meta" },
    ],
  },
  pass: {
    inputs: [{ name: "data", type: "any" }],
    outputs: [{ name: "data", type: "any" }],
  },
};

/** Tailwind background utilities used for each socket type's handle. */
export const PORT_COLOR: Record<PortType, string> = {
  frames: "!bg-primary",
  meta: "!bg-violet-500",
  branches: "!bg-warning",
  path: "!bg-success",
  fps: "!bg-cyan-500",
  any: "!bg-muted-foreground",
};

/**
 * Whether an output of type `a` may connect to an input of type `b`.
 * Mirrors the backend rule: compatible if either side is the wildcard
 * (`"any"`, a null type_hint) or the types match.
 */
export function portsCompatible(a: PortType, b: PortType): boolean {
  return a === "any" || b === "any" || a === b;
}

/** Look up an output port's type by node type + port name. */
export function outputPortType(node: string | undefined, port: string): PortType | undefined {
  if (!node || !(node in NODE_PORTS)) return undefined;
  return NODE_PORTS[node as BuilderNodeType].outputs.find((p) => p.name === port)?.type;
}

/** Look up an input port's type by node type + port name. */
export function inputPortType(node: string | undefined, port: string): PortType | undefined {
  if (!node || !(node in NODE_PORTS)) return undefined;
  return NODE_PORTS[node as BuilderNodeType].inputs.find((p) => p.name === port)?.type;
}
