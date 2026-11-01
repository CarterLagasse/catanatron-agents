import { useNavigate, useParams } from "react-router-dom";
import { useSnackbar } from "notistack";
import { useState } from "react";
import { Button, Dialog } from "@mui/material";

import type { Color, GameState } from "../utils/api.types";
import { createGame } from "../utils/apiClient";
import { turnsPlayed } from "../utils/gameLogic";
import { getGame, saveGame } from "../utils/history";
import { getHumanColor, playerKey } from "../utils/stateUtils";
import "./GameOverDialog.scss";

export default function GameOverDialog({
  gameState: gs,
  name,
  open,
  onClose,
}: {
  gameState: GameState;
  name: (color: Color) => string;
  open: boolean;
  onClose: () => void;
}) {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [starting, setStarting] = useState(false);
  const game = getGame(gameId);
  const human = getHumanColor(gs);
  const winner = gs.winning_color as Color;
  const humanWon = winner === human;
  const standings = [...gs.colors].sort(
    (a, b) =>
      gs.player_state[`${playerKey(gs, b)}_ACTUAL_VICTORY_POINTS`] -
      gs.player_state[`${playerKey(gs, a)}_ACTUAL_VICTORY_POINTS`],
  );

  const rematch = async () => {
    if (!game) return;
    setStarting(true);
    try {
      const id = await createGame({ players: game.players, ...game.settings });
      saveGame({ ...game, id, createdAt: Date.now(), result: undefined });
      onClose();
      navigate(`/games/${id}`);
    } catch (error) {
      console.error(error);
      enqueueSnackbar("Couldn't start a rematch. Is the game server running?", { variant: "error" });
    } finally {
      setStarting(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth slotProps={{ paper: { className: "game-over" } }}>
      <h2 className={humanWon ? "won" : undefined}>
        {human ? (humanWon ? "You won!" : `${name(winner)} won`) : `${name(winner)} won`}
      </h2>
      <p>Finished in {turnsPlayed(gs)} turns.</p>
      <ol className="standings">
        {standings.map((color) => {
          const key = playerKey(gs, color);
          const hidden = gs.player_state[`${key}_VICTORY_POINT_IN_HAND`] ?? 0;
          return (
            <li key={color} className={color.toLowerCase()}>
              <span className="swatch" aria-hidden />
              <span className="standing-name">{name(color)}</span>
              {hidden > 0 && <small>{hidden} hidden</small>}
              <b>{gs.player_state[`${key}_ACTUAL_VICTORY_POINTS`]}</b>
            </li>
          );
        })}
      </ol>
      <div className="game-over-actions">
        {game && (
          <Button variant="contained" onClick={rematch} disabled={starting}>
            {starting ? "Starting" : "Rematch"}
          </Button>
        )}
        <Button variant="outlined" color="inherit" onClick={() => navigate("/")}>
          New game
        </Button>
        <Button variant="outlined" color="inherit" onClick={() => navigate(`/replays/${gameId}`)}>
          Watch replay
        </Button>
        <Button color="inherit" onClick={onClose}>
          See board
        </Button>
      </div>
    </Dialog>
  );
}
