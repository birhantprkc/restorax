import { useEffect } from "react";
import type { XYPosition } from "@xyflow/react";
import { Copy, Maximize, Plus, Trash2 } from "lucide-react";

import { cn } from "@/lib/utils";

export interface CanvasMenuState {
  /** screen coords to position the menu at (from event.clientX/clientY) */
  x: number;
  y: number;
  /** the canvas-space position (flow coords) at the click, for inserting nodes */
  flowPosition: XYPosition;
  /** present when the right-click was on a node */
  nodeId?: string;
}

interface CanvasMenuProps {
  menu: CanvasMenuState | null;
  onClose: () => void;
  /** open node search to add a node at this flow position (pane menu) */
  onAddNode: (flowPosition: XYPosition) => void;
  onDuplicate: (nodeId: string) => void;
  onDelete: (nodeId: string) => void;
  onFitView: () => void;
}

const itemClass =
  "flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent hover:text-accent-foreground";

export function CanvasMenu({
  menu,
  onClose,
  onAddNode,
  onDuplicate,
  onDelete,
  onFitView,
}: CanvasMenuProps) {
  useEffect(() => {
    if (!menu) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    const handlePointerDown = () => onClose();

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("pointerdown", handlePointerDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [menu, onClose]);

  if (!menu) return null;

  return (
    <div
      role="menu"
      className="fixed z-50 min-w-44 rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-md"
      style={{ left: menu.x, top: menu.y }}
      onPointerDown={(event) => event.stopPropagation()}
    >
      {menu.nodeId ? (
        <>
          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={() => {
              onDuplicate(menu.nodeId!);
              onClose();
            }}
          >
            <Copy className="size-4" />
            Duplicate
          </button>
          <button
            type="button"
            role="menuitem"
            className={cn(itemClass, "text-destructive")}
            onClick={() => {
              onDelete(menu.nodeId!);
              onClose();
            }}
          >
            <Trash2 className="size-4" />
            Delete
          </button>
        </>
      ) : (
        <>
          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={() => {
              onAddNode(menu.flowPosition);
              onClose();
            }}
          >
            <Plus className="size-4" />
            Add node…
          </button>
          <button
            type="button"
            role="menuitem"
            className={itemClass}
            onClick={() => {
              onFitView();
              onClose();
            }}
          >
            <Maximize className="size-4" />
            Fit view
          </button>
        </>
      )}
    </div>
  );
}
