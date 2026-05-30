import { useEffect, useState } from "react";
import { Cpu, Loader2 } from "lucide-react";
import { fetchCeleryHealth } from "@/lib/api";
import type { CeleryHealth } from "@/types";
import { cn } from "@/lib/utils";

/** Compact worker/queue status, polled from GET /health/celery. */
export function GpuStatusWidget({ className }: { className?: string }) {
  const [health, setHealth] = useState<CeleryHealth | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const h = await fetchCeleryHealth();
        if (active) {
          setHealth(h);
          setError(false);
        }
      } catch {
        if (active) setError(true);
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const online = !error && health && health.workers > 0;

  return (
    <div className={cn("rounded-lg border border-border p-3 text-sm", className)}>
      <div className="mb-2 flex items-center gap-2 font-medium">
        <Cpu className="size-4 text-primary" />
        Workers
        <span
          className={cn(
            "ml-auto size-2 rounded-full",
            online ? "bg-success" : "bg-destructive",
          )}
        />
      </div>
      {!health && !error ? (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="size-3 animate-spin" /> connecting…
        </div>
      ) : error ? (
        <div className="text-muted-foreground">API unreachable</div>
      ) : (
        <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-muted-foreground">
          <dt>Online</dt>
          <dd className="text-right text-foreground">{health!.workers}</dd>
          <dt>Active</dt>
          <dd className="text-right text-foreground">{health!.active_tasks}</dd>
          <dt>Queued</dt>
          <dd className="text-right text-foreground">{health!.queued_tasks}</dd>
        </dl>
      )}
    </div>
  );
}
