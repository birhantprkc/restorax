import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { mergeJobBranches } from "@/lib/api";
import type { Job } from "@/types";
import type { CombinedBranch } from "./branches";

interface MergePanelProps {
  jobId: string;
  branches: CombinedBranch[];
  onMerged: (job: Job) => void;
}

type Strategy = "blend" | "select";

/** Merge DAG branches via blend (combine all) or select (pick one). */
export function MergePanel({ jobId, branches, onMerged }: MergePanelProps) {
  const [strategy, setStrategy] = useState<Strategy>("blend");
  const [branchIndex, setBranchIndex] = useState<number>(branches[0]?.branch_index ?? 0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  async function handleMerge() {
    setBusy(true);
    setError(null);
    setSuccess(false);
    try {
      const body =
        strategy === "select"
          ? { strategy, branch_index: branchIndex }
          : { strategy };
      const job = await mergeJobBranches(jobId, body);
      setSuccess(true);
      onMerged(job);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Merge failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Merge branches</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs text-muted-foreground">Strategy</Label>
            <Select value={strategy} onValueChange={(v) => setStrategy(v as Strategy)}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="blend">Blend (all)</SelectItem>
                <SelectItem value="select">Select one</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {strategy === "select" && (
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs text-muted-foreground">Branch</Label>
              <Select
                value={String(branchIndex)}
                onValueChange={(v) => setBranchIndex(Number(v))}
              >
                <SelectTrigger className="w-52">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {branches.map((b) => (
                    <SelectItem key={b.branch_index} value={String(b.branch_index)}>
                      {b.name || b.node_id || `branch ${b.branch_index}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <Button onClick={handleMerge} disabled={busy}>
            {busy ? "Merging…" : "Merge branches"}
          </Button>
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
        {success && (
          <p className="text-sm text-success">Branches merged.</p>
        )}
      </CardContent>
    </Card>
  );
}
