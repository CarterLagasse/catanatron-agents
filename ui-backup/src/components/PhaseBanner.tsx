import { useContext } from "react";
import { store } from "../store";
import { isPlayersTurn } from "../utils/stateUtils";
import "./PhaseBanner.scss";

const PROMPT_LABELS: Record<string, string> = {
  PLAY_TURN: "Roll the dice",
  BUILD_INITIAL_SETTLEMENT: "Place a settlement",
  BUILD_INITIAL_ROAD: "Place a road",
  DISCARD: "Discard cards",
  MOVE_ROBBER: "Move the robber",
  PLAY_YEAR_OF_PLENTY: "Pick resources",
  PLAY_MONOPOLY: "Pick a resource to steal",
};

export default function PhaseBanner({ isBotThinking }: { isBotThinking: boolean }) {
  const { state } = useContext(store);
  const gs = state.gameState;
  if (!gs) return null;

  if (gs.winning_color) {
    return (
      <div className="phase-banner win">
        <span className="phase-dot" />
        Game over — {gs.winning_color} wins! 🎉
      </div>
    );
  }
  if (isBotThinking) {
    return (
      <div className="phase-banner bot">
        <span className="phase-spinner" />
        {gs.current_color} is thinking…
      </div>
    );
  }
  const isPlayersTurnNow = isPlayersTurn(gs);
  const raw = gs.current_prompt as string;
  const label = PROMPT_LABELS[raw] || raw.replace(/_/g, " ").toLowerCase();

  return (
    <div className={`phase-banner ${isPlayersTurnNow ? "you" : "wait"}`}>
      <span className={`turn-chip ${gs.current_color.toLowerCase()}`}>{gs.current_color}</span>
      <span className="phase-text">
        {isPlayersTurnNow ? "Your turn" : `${gs.current_color}'s turn`} — {label}
      </span>
      {!isPlayersTurnNow && <span className="phase-sub">watch the board</span>}
    </div>
  );
}
