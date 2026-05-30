import { useEffect, useState } from "react";
import type { RestorerInfo } from "@/types";
import { fetchModels } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  BuilderNode,
  BuilderNodeData,
  MergeNodeData,
  RestoreNodeData,
} from "./types";

const labelCls = "mb-1.5 text-xs font-medium text-muted-foreground";

interface Props {
  node: BuilderNode | null;
  /** Patch the selected node's data (shallow merge). */
  onChange: (id: string, patch: Partial<BuilderNodeData>) => void;
}

export function ConfigPanel({ node, onChange }: Props) {
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    let alive = true;
    fetchModels()
      .then((m: RestorerInfo[]) => alive && setModels(m.map((r) => r.name)))
      .catch(() => alive && setModels([]));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <aside className="flex w-72 shrink-0 flex-col overflow-y-auto border-l border-border p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Node Config
      </h2>
      {!node ? (
        <p className="text-sm text-muted-foreground">
          Select a node to edit its settings.
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          <div>
            <Label className={labelCls}>Label</Label>
            <Input
              value={node.data.label}
              onChange={(e) => onChange(node.id, { label: e.target.value })}
            />
          </div>

          {node.data.kind === "restore" && (
            <RestoreFields node={node} models={models} onChange={onChange} />
          )}
          {node.data.kind === "merge" && (
            <MergeFields node={node} onChange={onChange} />
          )}
          {(node.data.kind === "parallel" ||
            node.data.kind === "pass" ||
            node.data.kind === "video_input" ||
            node.data.kind === "video_output") && (
            <p className="text-sm text-muted-foreground">
              Structural node — no extra settings.
            </p>
          )}
        </div>
      )}
    </aside>
  );
}

function RestoreFields({
  node,
  models,
  onChange,
}: {
  node: BuilderNode;
  models: string[];
  onChange: Props["onChange"];
}) {
  const data = node.data as RestoreNodeData;
  // Edit params as raw JSON; keep it simple and forgiving.
  const [paramsText, setParamsText] = useState(() =>
    JSON.stringify(data.params, null, 2),
  );
  const [paramsErr, setParamsErr] = useState(false);

  useEffect(() => {
    setParamsText(JSON.stringify(data.params, null, 2));
    setParamsErr(false);
    // Re-sync when a different node is selected.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node.id]);

  const options =
    data.restorer_name && !models.includes(data.restorer_name)
      ? [data.restorer_name, ...models]
      : models;

  return (
    <>
      <div>
        <Label className={labelCls}>Restorer</Label>
        <Select
          value={data.restorer_name || undefined}
          onValueChange={(v) => onChange(node.id, { restorer_name: v })}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="— select —" />
          </SelectTrigger>
          <SelectContent>
            {options.map((m) => (
              <SelectItem key={m} value={m}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div>
        <Label className={labelCls}>Params (JSON)</Label>
        <Textarea
          className={cn(
            "h-28 resize-y font-mono text-xs",
            paramsErr && "border-destructive",
          )}
          value={paramsText}
          onChange={(e) => {
            const text = e.target.value;
            setParamsText(text);
            try {
              const parsed = JSON.parse(text || "{}");
              setParamsErr(false);
              onChange(node.id, { params: parsed });
            } catch {
              setParamsErr(true);
            }
          }}
        />
        {paramsErr && (
          <p className="mt-1 text-xs text-destructive">
            Invalid JSON — not saved.
          </p>
        )}
      </div>
    </>
  );
}

function MergeFields({
  node,
  onChange,
}: {
  node: BuilderNode;
  onChange: Props["onChange"];
}) {
  const data = node.data as MergeNodeData;
  return (
    <>
      <div>
        <Label className={labelCls}>Strategy</Label>
        <Select
          value={data.strategy}
          onValueChange={(v) =>
            onChange(node.id, { strategy: v as MergeNodeData["strategy"] })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="blend">blend</SelectItem>
            <SelectItem value="select">select</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {data.strategy === "select" && (
        <div>
          <Label className={labelCls}>Select index</Label>
          <Input
            type="number"
            min={0}
            value={data.select_index}
            onChange={(e) =>
              onChange(node.id, {
                select_index: Math.max(0, Number(e.target.value) || 0),
              })
            }
          />
        </div>
      )}
    </>
  );
}
