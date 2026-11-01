import cn from "classnames";
import { Tooltip } from "@mui/material";
import MilitaryTechIcon from "@mui/icons-material/MilitaryTech";
import RouteIcon from "@mui/icons-material/Route";
import StyleIcon from "@mui/icons-material/Style";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";

import type { Color, GameState } from "../utils/api.types";
import { cardCount, devCardCount, DEV_CARDS, hand, RESOURCES } from "../utils/gameLogic";
import { getHumanColor, playerKey } from "../utils/stateUtils";
import ResourceIcon from "./ResourceIcon";
import "./Scoreboard.scss";

function Stat({
  icon,
  value,
  label,
  active,
}: {
  icon: React.ReactNode;
  value: number | string;
  label: string;
  active?: boolean;
}) {
  return (
    <Tooltip title={label} arrow>
      <span className={cn("stat", { active })}>
        {icon}
        <b>{value}</b>
      </span>
    </Tooltip>
  );
}

export default function Scoreboard({
  gameState: gs,
  name,
  revealHidden,
}: {
  gameState: GameState;
  name: (color: Color) => string;
  revealHidden: boolean;
}) {
  const human = getHumanColor(gs);
  return (
    <section className="scoreboard" aria-label="Scoreboard">
      {gs.colors.map((color) => {
        const key = playerKey(gs, color);
        const state = gs.player_state;
        const open = color === human || revealHidden || !!gs.winning_color;
        const vps = state[open ? `${key}_ACTUAL_VICTORY_POINTS` : `${key}_VICTORY_POINTS`] ?? 0;
        const cards = cardCount(gs, color);
        const devCards = devCardCount(gs, color);
        const overLimit = cards > gs.discard_limit;
        const progress = Math.min(100, (vps / gs.vps_to_win) * 100);
        const isCurrent = gs.current_color === color && !gs.winning_color;
        return (
          <article
            key={color}
            className={cn("score-card", color.toLowerCase(), {
              current: isCurrent,
              winner: gs.winning_color === color,
            })}
          >
            <header>
              <span className="swatch" aria-hidden />
              <span className="player-name" title={name(color)}>
                {name(color)}
              </span>
              {isCurrent && <span className="turn-tag">Playing</span>}
              <span className="vp" aria-label={`${vps} of ${gs.vps_to_win} victory points`}>
                <strong>{vps}</strong>
                <small>/{gs.vps_to_win}</small>
              </span>
            </header>
            <div className="vp-track" aria-hidden>
              <span style={{ width: `${progress}%` }} />
            </div>
            <div className="stats">
              <Stat
                icon={<StyleIcon fontSize="inherit" />}
                value={cards}
                label={overLimit ? `${cards} cards: over the discard limit of ${gs.discard_limit}` : `${cards} resource cards`}
                active={overLimit}
              />
              <Stat icon={<span className="dev-glyph" />} value={devCards} label={`${devCards} development cards`} />
              <Stat
                icon={<MilitaryTechIcon fontSize="inherit" />}
                value={state[`${key}_PLAYED_KNIGHT`] ?? 0}
                label={state[`${key}_HAS_ARMY`] ? "Largest Army: worth 2 points" : "Knights played"}
                active={!!state[`${key}_HAS_ARMY`]}
              />
              <Stat
                icon={<RouteIcon fontSize="inherit" />}
                value={state[`${key}_LONGEST_ROAD_LENGTH`] ?? 0}
                label={state[`${key}_HAS_ROAD`] ? "Longest Road: worth 2 points" : "Longest road length"}
                active={!!state[`${key}_HAS_ROAD`]}
              />
              {!open && (
                <Tooltip title="Their hand and any hidden victory points are secret" arrow>
                  <span className="secret" aria-label="Hidden information">
                    <VisibilityOffIcon fontSize="inherit" />
                  </span>
                </Tooltip>
              )}
            </div>
            {open && color !== human && (
              <div className="revealed">
                {RESOURCES.map((resource) => (
                  <span key={resource}>
                    <ResourceIcon resource={resource} size={14} />
                    {hand(gs, color)[resource]}
                  </span>
                ))}
                {DEV_CARDS.filter(({ card }) => (state[`${key}_${card}_IN_HAND`] ?? 0) > 0).map(({ card, name: cardName }) => (
                  <span key={card} className="revealed-dev">
                    {cardName} x{state[`${key}_${card}_IN_HAND`]}
                  </span>
                ))}
              </div>
            )}
          </article>
        );
      })}
    </section>
  );
}
