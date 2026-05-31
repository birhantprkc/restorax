import { useCallback, useEffect, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Connection,
  type Edge,
  type XYPosition,
} from "@xyflow/react";
import { Undo2, Redo2, Upload, Download, Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { createDag, fetchModels } from "@/lib/api";
import { Palette } from "@/components/builder/Palette";
import { ConfigPanel } from "@/components/builder/ConfigPanel";
import { NodeSearch } from "@/components/builder/NodeSearch";
import { CanvasMenu, type CanvasMenuState } from "@/components/builder/CanvasMenu";
import { nodeTypes } from "@/components/builder/nodes";
import { createNode, duplicateNode } from "@/components/builder/factory";
import { serializeDag, slugify } from "@/components/builder/serialize";
import { inputPortType, outputPortType, portsCompatible } from "@/components/builder/ports";
import { useUndoRedo } from "@/components/builder/useUndoRedo";
import { downloadWorkflow, parseWorkflow } from "@/components/builder/workflow";
import { DRAG_MIME, type BuilderNode, type BuilderNodeData, type PaletteDragPayload } from "@/components/builder/types";

/** True when focus is in a text field, so global hotkeys defer to native editing. */
function isEditableTarget(el: EventTarget | null): boolean {
  const node = el as HTMLElement | null;
  if (!node) return false;
  const tag = node.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    node.isContentEditable
  );
}

const initialNodes: BuilderNode[] = [
  {
    id: "video_input",
    type: "video_input",
    position: { x: 40, y: 160 },
    data: { kind: "video_input", label: "Video Input" },
  },
  {
    id: "video_output",
    type: "video_output",
    position: { x: 560, y: 160 },
    data: { kind: "video_output", label: "Video Output" },
  },
];

type SaveState =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "ok"; id: string }
  | { kind: "error"; message: string };

function Builder() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { screenToFlowPosition, fitView, getNodes } = useReactFlow();

  const [nodes, setNodes, onNodesChange] = useNodesState<BuilderNode>(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [name, setName] = useState("Untitled DAG");
  const [save, setSave] = useState<SaveState>({ kind: "idle" });

  const history = useUndoRedo();
  const snapshot = useCallback(
    () => history.takeSnapshot({ nodes, edges }),
    [history, nodes, edges],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      snapshot();
      setEdges((eds) => addEdge(c, eds));
    },
    [snapshot, setEdges],
  );

  // Type-check a candidate connection against the typed socket contract.
  // React Flow uses this both to block invalid drops and to dim sockets that
  // can't accept the wire being dragged. Mirrors backend port-type validation.
  const isValidConnection = useCallback(
    (c: Connection | Edge) => {
      if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return false;
      const byId = new Map(getNodes().map((n) => [n.id, n.type]));
      const src = outputPortType(byId.get(c.source), c.sourceHandle);
      const tgt = inputPortType(byId.get(c.target), c.targetHandle);
      if (!src || !tgt) return false;
      return portsCompatible(src, tgt);
    },
    [getNodes],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData(DRAG_MIME);
      if (!raw) return;
      const payload = JSON.parse(raw) as PaletteDragPayload;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      snapshot();
      setNodes((nds) => nds.concat(createNode(payload, position)));
    },
    [screenToFlowPosition, snapshot, setNodes],
  );

  const handleUndo = useCallback(() => {
    const prev = history.undo({ nodes, edges });
    if (prev) {
      setNodes(prev.nodes);
      setEdges(prev.edges);
    }
  }, [history, nodes, edges, setNodes, setEdges]);

  const handleRedo = useCallback(() => {
    const next = history.redo({ nodes, edges });
    if (next) {
      setNodes(next.nodes);
      setEdges(next.edges);
    }
  }, [history, nodes, edges, setNodes, setEdges]);

  // Undo/redo hotkeys, deferring to native editing while a text field is focused.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || isEditableTarget(e.target)) return;
      const key = e.key.toLowerCase();
      if (key === "z" && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      } else if ((key === "z" && e.shiftKey) || key === "y") {
        e.preventDefault();
        handleRedo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleUndo, handleRedo]);

  const onExport = useCallback(() => {
    downloadWorkflow(name, nodes, edges);
  }, [name, nodes, edges]);

  const onImportFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = ""; // allow re-importing the same file
      if (!file) return;
      try {
        const wf = parseWorkflow(await file.text());
        snapshot();
        setNodes(wf.nodes);
        setEdges(wf.edges);
        setName(wf.name);
        setSelectedId(null);
        setSave({ kind: "idle" });
      } catch (err) {
        setSave({
          kind: "error",
          message: err instanceof Error ? err.message : "Import failed",
        });
      }
    },
    [snapshot, setNodes, setEdges],
  );

  // ── Node search + context menu ──────────────────────────────────────────────
  const [restorers, setRestorers] = useState<string[]>([]);
  const [searchOpen, setSearchOpen] = useState(false);
  const [pendingPos, setPendingPos] = useState<XYPosition | null>(null);
  const [menu, setMenu] = useState<CanvasMenuState | null>(null);

  useEffect(() => {
    let alive = true;
    fetchModels()
      .then((m) => alive && setRestorers(m.map((r) => r.name)))
      .catch(() => alive && setRestorers([]));
    return () => {
      alive = false;
    };
  }, []);

  const addNode = useCallback(
    (payload: PaletteDragPayload, position: XYPosition) => {
      snapshot();
      setNodes((nds) => nds.concat(createNode(payload, position)));
    },
    [snapshot, setNodes],
  );

  const openSearchAt = useCallback((flowPosition: XYPosition) => {
    setPendingPos(flowPosition);
    setSearchOpen(true);
  }, []);

  const onPaneContextMenu = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      e.preventDefault();
      const flowPosition = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      setMenu({ x: e.clientX, y: e.clientY, flowPosition });
    },
    [screenToFlowPosition],
  );

  const onNodeContextMenu = useCallback(
    (e: React.MouseEvent, node: BuilderNode) => {
      e.preventDefault();
      const flowPosition = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      setMenu({ x: e.clientX, y: e.clientY, flowPosition, nodeId: node.id });
    },
    [screenToFlowPosition],
  );

  const duplicate = useCallback(
    (nodeId: string) => {
      const target = nodes.find((n) => n.id === nodeId);
      if (!target) return;
      snapshot();
      setNodes((nds) => nds.concat(duplicateNode(target)));
    },
    [nodes, snapshot, setNodes],
  );

  const deleteNode = useCallback(
    (nodeId: string) => {
      snapshot();
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
      setSelectedId((id) => (id === nodeId ? null : id));
    },
    [snapshot, setNodes, setEdges],
  );

  const patchNode = useCallback(
    (id: string, patch: Partial<BuilderNodeData>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, ...patch } as BuilderNodeData } : n,
        ),
      );
    },
    [setNodes],
  );

  const selectedNode = nodes.find((n) => n.id === selectedId) ?? null;

  const onSave = useCallback(async () => {
    const id = slugify(name);
    const config = serializeDag(id, name, nodes, edges);
    setSave({ kind: "saving" });
    try {
      const res = await createDag({ id, name, config });
      setSave({ kind: "ok", id: res.id });
    } catch (err) {
      setSave({ kind: "error", message: err instanceof Error ? err.message : "Save failed" });
    }
  }, [name, nodes, edges]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-4 border-b border-border px-6 py-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Pipeline Builder</h1>
          <p className="text-xs text-muted-foreground">
            Drag restorers onto the canvas, wire up branches, and save a DAG.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {save.kind === "ok" && (
            <Badge variant="success">saved: {save.id}</Badge>
          )}
          {save.kind === "error" && (
            <Badge variant="destructive" title={save.message}>
              {save.message.slice(0, 40)}
            </Badge>
          )}

          <Button
            variant="ghost"
            size="icon"
            onClick={handleUndo}
            disabled={!history.canUndo}
            title="Undo (Ctrl/Cmd+Z)"
          >
            <Undo2 />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleRedo}
            disabled={!history.canRedo}
            title="Redo (Ctrl/Cmd+Shift+Z)"
          >
            <Redo2 />
          </Button>

          <Separator orientation="vertical" className="h-6" />

          <Button
            variant="outline"
            size="sm"
            onClick={() => openSearchAt(screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 }))}
            title="Add node (double-click canvas)"
          >
            <Plus />
            Add node
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            title="Import workflow (.json)"
          >
            <Upload />
            Import
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onExport}
            title="Export workflow (.json)"
          >
            <Download />
            Export
          </Button>

          <Separator orientation="vertical" className="h-6" />

          <Input
            className="w-56"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="DAG name"
          />
          <Button onClick={onSave} disabled={save.kind === "saving" || !name.trim()}>
            {save.kind === "saving" ? "Saving…" : "Save DAG"}
          </Button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={onImportFile}
          />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <Palette />
        <div
          ref={wrapperRef}
          className="min-w-0 flex-1"
          onDragOver={onDragOver}
          onDrop={onDrop}
          onDoubleClick={(e) =>
            openSearchAt(screenToFlowPosition({ x: e.clientX, y: e.clientY }))
          }
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onNodeDragStart={snapshot}
            onNodesDelete={snapshot}
            onEdgesDelete={snapshot}
            onSelectionChange={({ nodes: sel }) => setSelectedId(sel[0]?.id ?? null)}
            onPaneContextMenu={onPaneContextMenu}
            onNodeContextMenu={onNodeContextMenu}
            onPaneClick={() => setMenu(null)}
            deleteKeyCode={["Backspace", "Delete"]}
            zoomOnDoubleClick={false}
            fitView
            colorMode="dark"
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>
        </div>
        <ConfigPanel node={selectedNode} onChange={patchNode} />

        <NodeSearch
          open={searchOpen}
          onOpenChange={setSearchOpen}
          restorers={restorers}
          onAdd={(payload) =>
            addNode(
              payload,
              pendingPos ??
                screenToFlowPosition({
                  x: window.innerWidth / 2,
                  y: window.innerHeight / 2,
                }),
            )
          }
        />
        <CanvasMenu
          menu={menu}
          onClose={() => setMenu(null)}
          onAddNode={openSearchAt}
          onDuplicate={duplicate}
          onDelete={deleteNode}
          onFitView={() => fitView()}
        />
      </div>
    </div>
  );
}

export default function PipelineBuilder() {
  return (
    <ReactFlowProvider>
      <Builder />
    </ReactFlowProvider>
  );
}
