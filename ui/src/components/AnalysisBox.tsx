import { useCallback, useContext, useEffect, useRef, useState } from "react";
import { useParams } from "react-router";
import { Button, CircularProgress } from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";

import { store } from "../store";
import type { Color } from "../utils/api.types";
import { type MCTSProbabilities, type StateIndex, getMctsAnalysis } from "../utils/apiClient";
import { isHumanTurn, turnsPlayed } from "../utils/gameLogic";
import { getHumanColor } from "../utils/stateUtils";
import "./AnalysisBox.scss";

type AnalysisBoxProps = {
  stateIndex: StateIndex;
  name: (color: Color) => string;
};

export default function AnalysisBox({ stateIndex, name }: AnalysisBoxProps) {
  const { gameId } = useParams();
  const { state } = useContext(store);
  const gs = state.gameState;
  const [odds, setOdds] = useState<{ probabilities: MCTSProbabilities; turn: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const latestRequest = useRef(0);
  const newestAnswer = useRef(0);

  const turn = gs ? turnsPlayed(gs) : 0;
  const run = useCallback(async () => {
    if (!gameId) return;
    const request = ++latestRequest.current;
    setLoading(true);
    setError("");
    try {
      const result = await getMctsAnalysis(gameId, stateIndex);
      // A slower, older answer still beats nothing, unless a newer one already landed.
      if (request < newestAnswer.current) return;
      newestAnswer.current = request;
      if (result.success) {
        setOdds({ probabilities: result.probabilities, turn });
      } else {
        setError(result.error || "Couldn't work out the odds.");
      }
    } catch (err) {
      console.error("Win odds failed:", err);
      if (request === latestRequest.current) setError("Couldn't reach the game server.");
    } finally {
      if (request === latestRequest.current) setLoading(false);
    }
  }, [gameId, stateIndex, turn]);

  // Refresh once at the start of each of the player's turns, and not while replaying.
  const refreshNow =
    stateIndex === "latest" && !!gs && isHumanTurn(gs) && !gs.is_initial_build_phase;
  useEffect(() => {
    if (refreshNow) run();
  }, [refreshNow, turn, run]);

  if (!gs) return null;

  const final = gs.winning_color
    ? (Object.fromEntries(
        gs.colors.map((color) => [color, color === gs.winning_color ? 100 : 0]),
      ) as MCTSProbabilities)
    : null;
  const shown = final ?? odds?.probabilities;
  const stale = !final && odds && odds.turn !== turn;

  return (
    <section className="analysis-box" aria-label="Win odds" aria-busy={loading}>
      <header>
        <h4>Win odds</h4>
        {stale && <small>from turn {odds.turn}</small>}
        {!final && (
          <Button
            size="small"
            color="inherit"
            onClick={run}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={13} color="inherit" /> : <RefreshIcon fontSize="small" />}
          >
            {loading ? "Working" : "Update"}
          </Button>
        )}
      </header>
      {error && <p className="odds-error" role="alert">{error}</p>}
      {shown ? (
        <ul>
          {gs.colors.map((color) => {
            const value = Math.round(shown[color] ?? 0);
            return (
              <li key={color} className={color.toLowerCase()} title={`${name(color)}: ${value}% chance of winning`}>
                <span className="odds-name">{name(color)}</span>
                <span className="odds-track">
                  {value > 0 && <span className="odds-bar" style={{ width: `${value}%` }} />}
                </span>
                <b>{value}%</b>
              </li>
            );
          })}
        </ul>
      ) : (
        !error && (
          <p className="odds-note">
            {loading
              ? "Playing out 100 random games from here."
              : stateIndex === "latest" && getHumanColor(gs)
                ? "Updates at the start of each of your turns."
                : "Press Update for the odds at this move."}
          </p>
        )
      )}
    </section>
  );
}
