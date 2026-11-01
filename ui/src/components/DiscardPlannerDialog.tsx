import React from "react";
import { Button, Dialog } from "@mui/material";
import cn from "classnames";

import type { ResourceCard } from "../utils/api.types";
import { RESOURCES, RESOURCE_NAMES } from "../utils/gameLogic";
import ResourceIcon from "./ResourceIcon";
import "./ResourceSelector.scss";
import "./DiscardPlannerDialog.scss";

type DiscardResourceCounts = Partial<Record<ResourceCard, number>>;

type DiscardPlannerDialogProps = {
  open: boolean;
  onClose: () => void;
  onConfirm: (resources: ResourceCard[]) => void;
  remainingDiscardCount: number;
  discardResourceCounts: DiscardResourceCounts;
  submitting?: boolean;
};

export default function DiscardPlannerDialog({
  open,
  onClose,
  onConfirm,
  remainingDiscardCount,
  discardResourceCounts,
  submitting = false,
}: DiscardPlannerDialogProps) {
  const [selections, setSelections] = React.useState<ResourceCard[]>([]);

  React.useEffect(() => {
    if (!open) {
      setSelections([]);
    }
  }, [open]);

  const selectedCount = selections.length;
  const countOf = (resource: ResourceCard) => selections.filter((selection) => selection === resource).length;
  const canAdd = (resource: ResourceCard) =>
    selectedCount < remainingDiscardCount && countOf(resource) < (discardResourceCounts[resource] ?? 0);
  const remove = (resource: ResourceCard) =>
    setSelections((current) => {
      const index = current.lastIndexOf(resource);
      return index === -1 ? current : [...current.slice(0, index), ...current.slice(index + 1)];
    });
  const left = remainingDiscardCount - selectedCount;

  return (
    <Dialog
      open={open}
      onClose={submitting ? undefined : onClose}
      className="discard-planner game-dialog"
      maxWidth="xs"
      fullWidth
    >
      <h2>Discard {remainingDiscardCount} card{remainingDiscardCount === 1 ? "" : "s"}</h2>
      <p>
        {left === 0
          ? "Ready. Tap a number to put a card back."
          : `A 7 was rolled and you hold too many. ${left} more to choose.`}
      </p>
      <div className="resource-grid">
        {RESOURCES.filter((resource) => discardResourceCounts[resource] != null).map((resource) => (
          <div key={resource} className="discard-column">
            <button
              className="resource-button"
              onClick={() => setSelections((current) => [...current, resource])}
              disabled={submitting || !canAdd(resource)}
              aria-label={`Discard a ${RESOURCE_NAMES[resource]}`}
            >
              <ResourceIcon resource={resource} size={38} />
              <span>{(discardResourceCounts[resource] ?? 0) - countOf(resource)} left</span>
            </button>
            <button
              className={cn("discard-count", { chosen: countOf(resource) > 0 })}
              onClick={() => remove(resource)}
              disabled={submitting || countOf(resource) === 0}
              aria-label={`Put back a ${RESOURCE_NAMES[resource]}`}
            >
              {countOf(resource)}
            </button>
          </div>
        ))}
      </div>
      <div className="dialog-actions">
        <Button color="inherit" onClick={onClose} disabled={submitting}>
          Look at board
        </Button>
        <Button
          variant="contained"
          onClick={() => onConfirm(selections)}
          disabled={submitting || selectedCount !== remainingDiscardCount}
        >
          Confirm
        </Button>
      </div>
    </Dialog>
  );
}
