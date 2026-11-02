import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import cn from "classnames";
import { Button } from "@mui/material";

import type { Color, GameState } from "../utils/api.types";
import type { StateIndex } from "../utils/apiClient";
import { RESOURCE_NAMES, RESOURCES, income } from "../utils/gameLogic";
import { buildLog } from "../utils/gameLog";
import { getHumanColor } from "../utils/stateUtils";
import type { RollGains } from "../utils/gameLogic";
import AnalysisBox from "./AnalysisBox";
import BankTray from "./BankTray";
import DiceChart from "./DiceChart";
import { Die } from "./DiceRoll";
import ResourceIcon from "./ResourceIcon";
import Scoreboard from "./Scoreboard";
import WinOddsChart from "./WinOddsChart";
import "./SidePanel.scss";

function GameLog({
  gameState,
  name,
  gainsByRecord,
}: {
  gameState: GameState;
  name: (color: Color) => string;
  gainsByRecord: Record<number, RollGains>;
}) {
  const turns = buildLog(gameState, name, gainsByRecord);
  if (turns.length === 0) {
    return <p className="empty-note">Moves will appear here as the game goes on.</p>;
  }
  return (
    <ol className="game-log" aria-label="Game log">
      {turns.map((turn) => (
        <li key={turn.key} className={cn("log-turn", turn.color.toLowerCase())}>
          <h4>
            {turn.label}
            {turn.label !== "Setup" && <small>{name(turn.color)}</small>}
          </h4>
          <ul>
            {turn.entries.map((entry) => (
              <li key={entry.index}>
                {entry.dice && (
                  <span className="log-dice" aria-hidden>
                    <Die value={entry.dice[0]} size={16} />
                    <Die value={entry.dice[1]} size={16} />
                  </span>
                )}
                <span>
                  {entry.resources ? (
                    <>
                      {entry.text.slice(0, entry.text.lastIndexOf(" ") + 1)}
                      <span className="log-tail">
                        {entry.text.slice(entry.text.lastIndexOf(" ") + 1)}{" "}
                        <span className="log-cards">
                          {entry.resources.map((resource, i) => (
                            <ResourceIcon key={i} resource={resource} size={15} />
                          ))}
                        </span>
                      </span>
                    </>
                  ) : (
                    entry.text
                  )}
                </span>
                {entry.gains?.map(({ color, resources }) => (
                  <span key={color} className="log-gain">
                    {name(color)} +
                    {resources.map((resource, i) => (
                      <ResourceIcon key={i} resource={resource} size={15} />
                    ))}
                  </span>
                ))}
              </li>
            ))}
          </ul>
        </li>
      ))}
    </ol>
  );
}

function Stats({
  gameState: gs,
  name,
  live,
  onSeeStats,
}: {
  gameState: GameState;
  name: (color: Color) => string;
  live: boolean;
  onSeeStats?: () => void;
}) {
  const human = getHumanColor(gs);
  const yours = human ? income(gs, human) : null;
  const total = yours ? Object.values(yours).reduce((sum, value) => sum + value, 0) : 0;
  return (
    <div className="stats-tab">
      {gs.winning_color && onSeeStats && (
        <Button className="see-all-stats" variant="outlined" color="inherit" onClick={onSeeStats}>
          See all game stats
        </Button>
      )}
      <DiceChart gameState={gs} />
      {yours && (
        <section>
          <h4>
            Your income
            <small>Rolls out of 36 that pay you, by resource. {total} in total.</small>
          </h4>
          <ul className="income">
            {RESOURCES.map((resource) => (
              <li key={resource} title={`${RESOURCE_NAMES[resource]}: ${yours[resource]} of 36 rolls`}>
                <ResourceIcon resource={resource} size={20} />
                <b>{yours[resource]}</b>
              </li>
            ))}
          </ul>
        </section>
      )}
      <WinOddsChart gameState={gs} name={name} live={live} />
    </div>
  );
}

type Tab = "log" | "stats";

export default function SidePanel({
  gameState,
  name,
  gainsByRecord,
  revealHidden,
  analysisIndex = "latest",
  footer,
  onSeeStats,
}: {
  gameState: GameState;
  name: (color: Color) => string;
  gainsByRecord: Record<number, RollGains>;
  revealHidden: boolean;
  analysisIndex?: StateIndex;
  footer?: ReactNode;
  onSeeStats?: () => void;
}) {
  const [tab, setTab] = useState<Tab>("log");
  const tabBody = useRef<HTMLDivElement>(null);
  const followNewest = useRef(true);
  const moves = gameState.action_records.length;
  useLayoutEffect(() => {
    if (tab === "log" && followNewest.current && tabBody.current) {
      tabBody.current.scrollTop = tabBody.current.scrollHeight;
    }
  }, [tab, moves]);
  const tabs: { id: Tab; label: string }[] = [
    { id: "log", label: "Log" },
    { id: "stats", label: "Stats" },
  ];
  return (
    <div className="side-panel-inner">
      <Scoreboard gameState={gameState} name={name} revealHidden={revealHidden} />
      <BankTray gameState={gameState} />
      <AnalysisBox stateIndex={analysisIndex} name={name} />
      <div className="tab-row" role="tablist">
        {tabs.map(({ id, label }) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={cn({ selected: tab === id })}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>
      <div
        className="tab-body"
        ref={tabBody}
        onScroll={({ currentTarget: body }) => {
          if (tab === "log") {
            followNewest.current = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
          }
        }}
      >
        {tab === "log" && <GameLog gameState={gameState} name={name} gainsByRecord={gainsByRecord} />}
        {tab === "stats" && (
          <Stats gameState={gameState} name={name} live={analysisIndex === "latest"} onSeeStats={onSeeStats} />
        )}
      </div>
      {footer}
    </div>
  );
}
