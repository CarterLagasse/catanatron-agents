import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import cn from "classnames";
import { Tooltip } from "@mui/material";

import type { Color, GameState } from "../utils/api.types";
import type { StateIndex } from "../utils/apiClient";
import { RESOURCE_NAMES, RESOURCES, income, rollHistogram, pips } from "../utils/gameLogic";
import { buildLog } from "../utils/gameLog";
import { getHumanColor } from "../utils/stateUtils";
import type { RollGains } from "../utils/gameLogic";
import AnalysisBox from "./AnalysisBox";
import BankTray from "./BankTray";
import { Die } from "./DiceRoll";
import ResourceIcon from "./ResourceIcon";
import Scoreboard from "./Scoreboard";
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
                <span>{entry.text}</span>
                {entry.resources?.map((resource, i) => (
                  <ResourceIcon key={i} resource={resource} size={15} />
                ))}
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

function DiceChart({ gameState }: { gameState: GameState }) {
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

function Stats({ gameState: gs }: { gameState: GameState }) {
  const human = getHumanColor(gs);
  const yours = human ? income(gs, human) : null;
  const total = yours ? Object.values(yours).reduce((sum, value) => sum + value, 0) : 0;
  return (
    <div className="stats-tab">
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
}: {
  gameState: GameState;
  name: (color: Color) => string;
  gainsByRecord: Record<number, RollGains>;
  revealHidden: boolean;
  analysisIndex?: StateIndex;
  footer?: ReactNode;
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
        {tab === "stats" && <Stats gameState={gameState} />}
      </div>
      {footer}
    </div>
  );
}
