import cn from "classnames";

import "./PlayerStateBox.scss";
import { type Color, type PlayerState } from "../utils/api.types";
import ResourceCards from "./ResourceCards";

export default function PlayerStateBox({ playerState, playerKey, color }: {
  playerState: PlayerState; playerKey: string; color: Color }) {
  const actualVps = playerState[`${playerKey}_ACTUAL_VICTORY_POINTS`];
  const isCurrent = (playerState as any).__isCurrent as boolean | undefined;
  const hasArmy = playerState[`${playerKey}_HAS_ARMY`];
  const hasRoad = playerState[`${playerKey}_HAS_ROAD`];
  return (
    <div className={cn("player-state-box foreground", color, { "is-current": isCurrent })}>
      <div className="player-header">
        <div className="player-name">
          <span className={cn("avatar", color)}>{color[0]}</span>
          <span>{color}</span>
          {isCurrent && <span className="turn-dot" aria-label="Current turn" />}
        </div>
        <span className="vp-badge">{actualVps} VP</span>
      </div>
      <ResourceCards playerState={playerState} playerKey={playerKey} />
      <div className="scores">
        <div
          className={cn("num-knights center-text", { bold: hasArmy })}
          title={hasArmy ? "Largest Army" : "Knights Played"}
        >
          <span>{playerState[`${playerKey}_PLAYED_KNIGHT`]}</span>
          <small>{hasArmy ? "army" : "knights"}</small>
        </div>
        <div
          className={cn("num-roads center-text", { bold: hasRoad })}
          title={hasRoad ? "Longest Road" : "Longest Road Length"}
        >
          {playerState[`${playerKey}_LONGEST_ROAD_LENGTH`]}
          <small>{hasRoad ? "longest" : "roads"}</small>
        </div>
        <div
          className={cn("victory-points center-text", { bold: actualVps >= 10 })}
          title="Victory Points"
        >
          {actualVps}
          <small>VPs</small>
        </div>
      </div>
    </div>
  );
}
