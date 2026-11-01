import { useEffect, useState } from "react";
import { Button, Dialog } from "@mui/material";

import type { ResourceCard } from "../utils/api.types";
import { RESOURCES, RESOURCE_NAMES } from "../utils/gameLogic";
import ResourceIcon from "./ResourceIcon";
import "./ResourceSelector.scss";

type SelectorOption = ResourceCard | ResourceCard[];

type ResourceSelectorProps = {
  open: boolean;
  onClose: () => void;
  onSelect: (option: SelectorOption) => void;
  options: SelectorOption[];
  mode: "monopoly" | "yearOfPlenty";
};

export default function ResourceSelector({ open, onClose, options, onSelect, mode }: ResourceSelectorProps) {
  // Year of Plenty picks one card at a time instead of listing all 15 pairs.
  const [first, setFirst] = useState<ResourceCard | null>(null);
  useEffect(() => {
    if (!open) setFirst(null);
  }, [open]);

  const choices = options.filter((option): option is ResourceCard[] => Array.isArray(option));
  const pairs = choices.filter((option) => option.length === 2);
  const matchingPair = (a: ResourceCard, b: ResourceCard) =>
    pairs.find(([x, y]) => (x === a && y === b) || (x === b && y === a));

  const enabled = (resource: ResourceCard) => {
    if (mode === "monopoly") return true;
    if (pairs.length === 0) return choices.some(([only]) => only === resource);
    if (first) return !!matchingPair(first, resource);
    return pairs.some((pair) => pair.includes(resource));
  };

  const choose = (resource: ResourceCard) => {
    if (mode === "monopoly") {
      onSelect(resource);
    } else if (pairs.length === 0) {
      onSelect([resource]);
    } else if (first) {
      const pair = matchingPair(first, resource);
      if (pair) onSelect(pair);
    } else {
      setFirst(resource);
    }
  };

  const title = mode === "monopoly" ? "Monopoly" : "Year of Plenty";
  const prompt =
    mode === "monopoly"
      ? "Take every card of one resource from the other players."
      : pairs.length === 0
        ? "The bank is nearly empty, so you can take 1 card."
        : first
          ? `You'll get ${RESOURCE_NAMES[first]}. Pick your second card.`
          : "Pick your first card from the bank.";

  return (
    <Dialog open={open} onClose={onClose} className="resource-selector game-dialog" maxWidth="xs" fullWidth>
      <h2>{title}</h2>
      <p>{prompt}</p>
      <div className="resource-grid">
        {RESOURCES.map((resource) => (
          <button
            key={resource}
            className="resource-button"
            disabled={!enabled(resource)}
            onClick={() => choose(resource)}
          >
            <ResourceIcon resource={resource} size={38} />
            <span>{RESOURCE_NAMES[resource]}</span>
          </button>
        ))}
      </div>
      <div className="dialog-actions">
        {first && (
          <Button color="inherit" onClick={() => setFirst(null)}>
            Start over
          </Button>
        )}
        <Button color="inherit" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </Dialog>
  );
}
