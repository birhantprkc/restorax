import type { LucideIcon } from "lucide-react";
import {
  FileVideo,
  Video,
  Wand2,
  Split,
  GitMerge,
  ArrowRight,
  Sparkles,
} from "lucide-react";

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import type { BuilderNodeType, PaletteDragPayload } from "./types";

interface NodeSearchProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** restorer names (from /models), offered as "add a restore node preset to this restorer" */
  restorers: string[];
  /** called when the user picks an item; the parent inserts a node from this payload */
  onAdd: (payload: PaletteDragPayload) => void;
}

interface StructuralEntry {
  type: BuilderNodeType;
  label: string;
  icon: LucideIcon;
}

const STRUCTURAL: StructuralEntry[] = [
  { type: "video_input", label: "Video Input", icon: FileVideo },
  { type: "video_output", label: "Video Output", icon: Video },
  { type: "restore", label: "Restore", icon: Wand2 },
  { type: "parallel", label: "Parallel", icon: Split },
  { type: "merge", label: "Merge", icon: GitMerge },
  { type: "pass", label: "Pass", icon: ArrowRight },
];

export function NodeSearch({
  open,
  onOpenChange,
  restorers,
  onAdd,
}: NodeSearchProps) {
  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput placeholder="Search nodes…" />
      <CommandList>
        <CommandEmpty>No nodes found.</CommandEmpty>
        <CommandGroup heading="Nodes">
          {STRUCTURAL.map(({ type, label, icon: Icon }) => (
            <CommandItem
              key={type}
              value={`${label} ${type}`}
              onSelect={() => {
                onAdd({ type, label });
                onOpenChange(false);
              }}
            >
              <Icon className="mr-2 size-4" />
              {label}
            </CommandItem>
          ))}
        </CommandGroup>
        {restorers.length > 0 && (
          <CommandGroup heading="Restorers">
            {restorers.map((name) => (
              <CommandItem
                key={name}
                value={`restorer ${name}`}
                onSelect={() => {
                  onAdd({ type: "restore", label: name, restorer_name: name });
                  onOpenChange(false);
                }}
              >
                <Sparkles className="mr-2 size-4" />
                {name}
              </CommandItem>
            ))}
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  );
}
