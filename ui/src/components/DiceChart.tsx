import { Tooltip } from "@mui/material";

import type { GameState } from "../utils/api.types";
import { pips, rollHistogram } from "../utils/gameLogic";
import "./DiceChart.scss";

export default function DiceChart({ gameState }: { gameState: GameState }) {
  const histogram = rollHistogram(gameState);
  const rolls = histogram.reduce((sum, { count }) => sum + count, 0);
  const expected = (total: number) => (pips(total) / 36) * rolls;
  const top = Math.max(1, ...histogram.map(({ count, total }) => Math.max(count, expected(total))));
  return (
    <figure className="dice-chart">
      <figcaption>
        <strong>Dice rolls</strong>
        <span>
          {rolls} roll{rolls === 1 ? "" : "s"}. Bars are what happened, ticks are what's expected.
        </span>
      </figcaption>
      <div className="bars">
        {histogram.map(({ total, count }) => (
          <Tooltip
            key={total}
            arrow
            title={`${total}: rolled ${count} time${count === 1 ? "" : "s"}, expected ${expected(total).toFixed(1)}`}
          >
            <div className="bar-slot" aria-label={`${total}: rolled ${count}, expected ${expected(total).toFixed(1)}`}>
              <div className="plot">
                {count > 0 && <span className="bar" style={{ height: `${(count / top) * 100}%` }} />}
                {rolls > 0 && <span className="tick" style={{ bottom: `${(expected(total) / top) * 100}%` }} />}
              </div>
              <small>{total}</small>
            </div>
          </Tooltip>
        ))}
      </div>
    </figure>
  );
}
