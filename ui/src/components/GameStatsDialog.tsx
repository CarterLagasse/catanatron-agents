import { Button, Dialog } from "@mui/material";
import cn from "classnames";

import type { Color, GameState } from "../utils/api.types";
import { playerStats, turnsPlayed } from "../utils/gameLogic";
import DiceChart from "./DiceChart";
import WinOddsChart from "./WinOddsChart";
import "./GameStatsDialog.scss";

const ROWS: { key: keyof ReturnType<typeof playerStats>; label: string; hint?: string }[] = [
  { key: "points", label: "Points" },
  { key: "settlements", label: "Settlements", hint: "still standing at the end" },
  { key: "cities", label: "Cities" },
  { key: "roads", label: "Roads built" },
  { key: "longestRoad", label: "Longest road" },
  { key: "knights", label: "Knights played" },
  { key: "devCards", label: "Dev cards bought" },
  { key: "trades", label: "Bank trades" },
  { key: "stolen", label: "Cards stolen" },
  { key: "discarded", label: "Cards discarded" },
  { key: "income", label: "Income", hint: "rolls out of 36 that pay, at the end" },
];

export default function GameStatsDialog({
  gameState: gs,
  name,
  open,
  onClose,
  onBack,
}: {
  gameState: GameState;
  name: (color: Color) => string;
  open: boolean;
  onClose: () => void;
  onBack: () => void;
}) {
  const stats = new Map(gs.colors.map((color) => [color, playerStats(gs, color)]));
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth slotProps={{ paper: { className: "game-stats" } }}>
      <header>
        <h2>Game stats</h2>
        <p>
          {gs.winning_color ? `${name(gs.winning_color)} won in ${turnsPlayed(gs)} turns.` : `${turnsPlayed(gs)} turns so far.`}
        </p>
      </header>
      <div className="stats-grid">
        <section className="stats-players">
          <table>
            <thead>
              <tr>
                <th scope="col" aria-label="Stat" />
                {gs.colors.map((color) => (
                  <th key={color} scope="col" className={cn(color.toLowerCase(), { winner: color === gs.winning_color })}>
                    <i aria-hidden />
                    {name(color)}
                    {color === gs.winning_color && <small>Winner</small>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map(({ key, label, hint }) => (
                <tr key={key} className={cn({ "points-row": key === "points" })}>
                  <th scope="row">
                    {label}
                    {hint && <small>{hint}</small>}
                  </th>
                  {gs.colors.map((color) => (
                    <td key={color}>{stats.get(color)?.[key]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <div className="stats-charts">
          <section>
            <WinOddsChart gameState={gs} name={name} live />
          </section>
          <section>
            <DiceChart gameState={gs} />
          </section>
        </div>
      </div>
      <div className="game-stats-actions">
        <Button variant="outlined" color="inherit" onClick={onBack}>
          Back to results
        </Button>
        <Button variant="contained" onClick={onClose}>
          See board
        </Button>
      </div>
    </Dialog>
  );
}
