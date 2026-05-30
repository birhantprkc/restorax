import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";

interface NodeShellProps {
  selected?: NodeProps["selected"];
  title: string;
  subtitle?: string;
  /** Tailwind accent class for the left bar / handles, e.g. "bg-primary". */
  accent: string;
  /** Render a target (input) handle on the left. */
  hasInput?: boolean;
  /** Render a source (output) handle on the right. */
  hasOutput?: boolean;
}

/** Shared visual chrome for builder nodes — keeps handles + theme consistent. */
export function NodeShell({
  selected,
  title,
  subtitle,
  accent,
  hasInput = true,
  hasOutput = true,
}: NodeShellProps) {
  return (
    <div
      className={cn(
        "relative min-w-[160px] rounded-lg border bg-card px-3 py-2 shadow-sm transition-colors",
        selected
          ? "border-ring ring-1 ring-ring"
          : "border-border",
      )}
    >
      <div className={cn("absolute inset-y-0 left-0 w-1 rounded-l-lg", accent)} />
      {hasInput && (
        <Handle
          id="video"
          type="target"
          position={Position.Left}
          className="!h-2.5 !w-2.5 !border-0 !bg-muted-foreground"
        />
      )}
      <div className="pl-1.5">
        <p className="text-sm font-medium leading-tight text-foreground">{title}</p>
        {subtitle && (
          <p className="truncate text-xs text-muted-foreground">{subtitle}</p>
        )}
      </div>
      {hasOutput && (
        <Handle
          id="video"
          type="source"
          position={Position.Right}
          className="!h-2.5 !w-2.5 !border-0 !bg-muted-foreground"
        />
      )}
    </div>
  );
}
