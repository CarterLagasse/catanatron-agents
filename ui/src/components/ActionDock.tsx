import { useContext, useEffect, useRef, useState, type ElementType } from "react";
import cn from "classnames";
import { Popover, Tooltip } from "@mui/material";
import CasinoIcon from "@mui/icons-material/Casino";
import HomeIcon from "@mui/icons-material/Home";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import HourglassTopIcon from "@mui/icons-material/HourglassTop";
import PanToolIcon from "@mui/icons-material/PanTool";

import ACTIONS from "../actions";
import useGameActions from "../hooks/useGameActions";
import { store, type BuildMode } from "../store";
import type {
  DevelopmentCard,
  GameAction,
  GameState,
  ResourceCard,
} from "../utils/api.types";
import {
  actionsOfType,
  BUILD_COSTS,
  DEV_CARDS,
  hand,
  hasRolled,
  isFreeRoadMode,
  isHumanTurn,
  missingCards,
  RESOURCE_NAMES,
  RESOURCES,
  tradeOptions,
} from "../utils/gameLogic";
import { getHumanColor, playerKey } from "../utils/stateUtils";
import DiscardPlannerDialog from "./DiscardPlannerDialog";
import { CityArt, DevCardArt, DevCardLogo, RoadArt, SettlementArt } from "./PieceArt";
import ResourceIcon from "./ResourceIcon";
import ResourceSelector from "./ResourceSelector";
import "./ActionDock.scss";

const BUILDS = [
  { type: "BUILD_ROAD", mode: "ROAD", label: "Road", Icon: RoadArt, keyHint: "1" },
  { type: "BUILD_SETTLEMENT", mode: "SETTLEMENT", label: "Settlement", Icon: SettlementArt, keyHint: "2" },
  { type: "BUILD_CITY", mode: "CITY", label: "City", Icon: CityArt, keyHint: "3" },
  { type: "BUY_DEVELOPMENT_CARD", mode: null, label: "Dev card", Icon: DevCardArt, keyHint: "4" },
] as const;

const PLAY_ACTIONS: Record<DevelopmentCard, GameAction[1]> = {
  KNIGHT: "PLAY_KNIGHT_CARD",
  YEAR_OF_PLENTY: "PLAY_YEAR_OF_PLENTY",
  MONOPOLY: "PLAY_MONOPOLY",
  ROAD_BUILDING: "PLAY_ROAD_BUILDING",
};

function Hand({ gameState: gs }: { gameState: GameState }) {
  const human = getHumanColor(gs);
  const cards = hand(gs, human);
  return (
    <div className="hand" aria-label="Your resource cards">
      {RESOURCES.map((resource) => (
        <Tooltip key={resource} title={RESOURCE_NAMES[resource]} arrow>
          <div className={cn("card", resource.toLowerCase(), { empty: cards[resource] === 0 })}>
            <ResourceIcon resource={resource} size={26} />
            <b aria-label={`${cards[resource]} ${RESOURCE_NAMES[resource]}`}>{cards[resource]}</b>
          </div>
        </Tooltip>
      ))}
    </div>
  );
}

function TradePanel({
  gameState: gs,
  anchor,
  onClose,
  onTrade,
}: {
  gameState: GameState;
  anchor: HTMLElement | null;
  onClose: () => void;
  onTrade: (action: GameAction) => void;
}) {
  const [give, setGive] = useState<ResourceCard | null>(null);
  const options = tradeOptions(gs);
  const chosen = options.find((option) => option.give === give);
  const cards = getHumanColor(gs) ? hand(gs, getHumanColor(gs)) : null;
  useEffect(() => {
    if (!anchor) setGive(null);
  }, [anchor]);
  return (
    <Popover
      open={!!anchor}
      anchorEl={anchor}
      onClose={onClose}
      anchorOrigin={{ vertical: "top", horizontal: "center" }}
      transformOrigin={{ vertical: "bottom", horizontal: "center" }}
      slotProps={{ paper: { className: "trade-panel" } }}
    >
      <h3>Trade with the bank</h3>
      <p>1. Give</p>
      <div className="trade-row">
        {RESOURCES.map((resource) => {
          const option = options.find(({ give: g }) => g === resource);
          return (
            <button
              key={resource}
              disabled={!option}
              className={cn("trade-card", { selected: give === resource })}
              aria-label={option ? `Give ${option.rate} ${RESOURCE_NAMES[resource]}` : `Not enough ${RESOURCE_NAMES[resource]}`}
              onClick={() => setGive(resource)}
            >
              <ResourceIcon resource={resource} size={26} />
              <b>{option ? `${option.rate}:1` : (cards?.[resource] ?? 0)}</b>
            </button>
          );
        })}
      </div>
      <p>2. Get 1</p>
      <div className="trade-row">
        {RESOURCES.map((resource) => {
          const action = chosen?.gets[resource];
          return (
            <button
              key={resource}
              disabled={!action}
              className="trade-card"
              aria-label={`Get 1 ${RESOURCE_NAMES[resource]}`}
              onClick={() => action && onTrade(action)}
            >
              <ResourceIcon resource={resource} size={26} />
            </button>
          );
        })}
      </div>
      {!chosen && <small>Rates include your ports. You need that many of a card to trade it.</small>}
    </Popover>
  );
}

export default function ActionDock({ isBotThinking }: { isBotThinking: boolean }) {
  const { state, dispatch } = useContext(store);
  const { act, busy } = useGameActions();
  const [tradeAnchor, setTradeAnchor] = useState<HTMLElement | null>(null);
  const [selector, setSelector] = useState<"monopoly" | "yearOfPlenty" | null>(null);
  const [discardOpen, setDiscardOpen] = useState(false);
  const tradeButton = useRef<HTMLButtonElement>(null);
  const gs = state.gameState as GameState;
  const human = getHumanColor(gs);
  const humanTurn = !!human && isHumanTurn(gs);
  const prompt = gs.current_prompt;
  const isMain = humanTurn && prompt === "PLAY_TURN" && !isFreeRoadMode(gs);
  const rolled = hasRolled(gs);
  const isDiscard = humanTurn && prompt === "DISCARD";
  const cards = human ? hand(gs, human) : null;
  const playable = new Set(humanTurn ? gs.current_playable_actions.map((action) => action[1]) : []);

  // A discard can't be skipped, so ask for it as soon as it's due.
  useEffect(() => {
    setDiscardOpen(isDiscard);
  }, [isDiscard, gs.state_index]);

  const setBuildMode = (mode: BuildMode) => dispatch({ type: ACTIONS.SET_BUILD_MODE, data: mode });

  const pressBuild = (build: (typeof BUILDS)[number]) => {
    if (!playable.has(build.type) || gs.is_initial_build_phase || !isMain) return;
    if (build.mode === null) {
      act([human, "BUY_DEVELOPMENT_CARD", null]);
    } else {
      setBuildMode(state.buildMode === build.mode ? null : build.mode);
    }
  };
  const playDev = (card: DevelopmentCard) => {
    if (!playable.has(PLAY_ACTIONS[card])) return;
    if (card === "MONOPOLY") setSelector("monopoly");
    else if (card === "YEAR_OF_PLENTY") setSelector("yearOfPlenty");
    else if (card === "KNIGHT") act([human, "PLAY_KNIGHT_CARD", null]);
    else act([human, "PLAY_ROAD_BUILDING", null]);
  };
  const canRoll = isMain && playable.has("ROLL");
  const canEnd = isMain && playable.has("END_TURN");
  const canTrade = isMain && rolled && tradeOptions(gs).length > 0;

  // Keyboard shortcuts, ignored while typing or while a dialog is up.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) return;
      if (document.querySelector(".MuiDialog-root, .MuiPopover-root")) return;
      const key = event.key.toLowerCase();
      const build = BUILDS.find(({ keyHint }) => keyHint === key);
      if (key === "escape") setBuildMode(null);
      else if (key === "r" && canRoll) act([human, "ROLL", null]);
      else if (key === "e" && canEnd) act([human, "END_TURN", null]);
      else if (key === "t" && canTrade) setTradeAnchor(tradeButton.current);
      else if (build) pressBuild(build);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const whyNot = (type: string, label: string) => {
    if (!human || gs.winning_color) return label;
    if (!humanTurn) return `${label}: wait for your turn`;
    if (gs.is_initial_build_phase) return `${label}: finish placing your starting pieces`;
    if (!isMain) return `${label}: finish the current step first`;
    if (!rolled) return `${label}: roll the dice first`;
    const missing = cards ? missingCards(cards, BUILD_COSTS[type]) : [];
    if (missing.length > 0) {
      return `${label}: you need ${missing.map((r) => RESOURCE_NAMES[r].toLowerCase()).join(", ")}`;
    }
    if (type === "BUY_DEVELOPMENT_CARD") return `${label}: the deck is empty`;
    return `${label}: no legal spot right now`;
  };

  let primary: { label: string; Icon: ElementType; onClick?: () => void; hint?: string };
  if (gs.winning_color) {
    primary = { label: "Game over", Icon: SkipNextIcon };
  } else if (!humanTurn) {
    primary = { label: isBotThinking ? "Thinking" : "Waiting", Icon: HourglassTopIcon };
  } else if (isDiscard) {
    primary = { label: `Discard ${gs.current_discard_count}`, Icon: PanToolIcon, onClick: () => setDiscardOpen(true) };
  } else if (prompt === "MOVE_ROBBER") {
    primary = { label: "Move robber", Icon: PanToolIcon };
  } else if (gs.is_initial_build_phase) {
    primary = { label: "Place piece", Icon: HomeIcon };
  } else if (isFreeRoadMode(gs)) {
    primary = {
      label: "Place road",
      Icon: () => (
        <span className={cn("build-art", "piece-colors", human.toLowerCase())}>
          <RoadArt />
        </span>
      ),
    };
  } else if (canRoll) {
    primary = { label: "Roll", Icon: CasinoIcon, onClick: () => act([human, "ROLL", null]), hint: "R" };
  } else {
    primary = {
      label: "End turn",
      Icon: SkipNextIcon,
      onClick: canEnd ? () => act([human, "END_TURN", null]) : undefined,
      hint: "E",
    };
  }

  const key = human ? playerKey(gs, human) : "";
  const ownedDev = DEV_CARDS.filter(({ card }) => (gs.player_state[`${key}_${card}_IN_HAND`] ?? 0) > 0);
  const yearOptions = actionsOfType(gs, "PLAY_YEAR_OF_PLENTY").map((action) => action[2] as ResourceCard[]);

  if (!human) {
    return (
      <div className="action-dock spectating">
        <span>Watching the bots play.</span>
      </div>
    );
  }

  return (
    <div className="action-dock">
      <div className="dock-hand">
        <Hand gameState={gs} />
        {ownedDev.length > 0 && (
          <div className={cn("dev-row", { crowded: ownedDev.length > 2 })} aria-label="Your development cards">
            {ownedDev.map(({ card, name, short, hint }) => {
              const count = gs.player_state[`${key}_${card}_IN_HAND`];
              const canPlay =
                card !== "VICTORY_POINT" && playable.has(PLAY_ACTIONS[card]) && !busy;
              return (
                <Tooltip
                  key={card}
                  arrow
                  title={
                    card === "VICTORY_POINT" || canPlay
                      ? `${name}: ${hint}`
                      : `${name}: ${hint}. Playable on a later turn: one per turn, and not the turn you bought it.`
                  }
                >
                  <span className="dev-wrap">
                    <button
                      className={cn("dev-card", { playable: canPlay, passive: card === "VICTORY_POINT" })}
                      disabled={!canPlay}
                      aria-label={count > 1 ? `${name} x${count}` : name}
                      onClick={() => playDev(card as DevelopmentCard)}
                    >
                      <DevCardLogo card={card} />
                      <small>{short}</small>
                      {count > 1 && <b>{count}</b>}
                    </button>
                  </span>
                </Tooltip>
              );
            })}
          </div>
        )}
      </div>
      <div className="dock-actions">
        {BUILDS.map((build) => {
          const enabled = isMain && playable.has(build.type) && !busy;
          const missing = cards ? missingCards(cards, BUILD_COSTS[build.type]) : [];
          const remaining = [...missing];
          return (
            <Tooltip key={build.type} arrow title={enabled ? `${build.label} (press ${build.keyHint})` : whyNot(build.type, build.label)}>
              <span className="build-wrap">
                <button
                  className={cn("build-button", { active: state.buildMode === build.mode && build.mode !== null })}
                  disabled={!enabled}
                  aria-pressed={build.mode ? state.buildMode === build.mode : undefined}
                  onClick={() => pressBuild(build)}
                >
                  <span className="build-title">
                    <span className={cn("build-art", "piece-colors", human.toLowerCase())}>
                      <build.Icon />
                    </span>
                    {build.label}
                  </span>
                  <span className="build-cost">
                    {BUILD_COSTS[build.type].map((resource, i) => {
                      const short = remaining.indexOf(resource);
                      if (short >= 0) remaining.splice(short, 1);
                      return (
                        <span key={i} className={cn({ short: short >= 0 })}>
                          <ResourceIcon resource={resource} size={14} />
                        </span>
                      );
                    })}
                  </span>
                </button>
              </span>
            </Tooltip>
          );
        })}
        <Tooltip arrow title={canTrade ? "Trade with the bank (press T)" : isMain && rolled ? "Trade: you don't have enough of any card" : "Trade: available after you roll"}>
          <span className="build-wrap">
            <button
              ref={tradeButton}
              className="build-button trade"
              disabled={!canTrade || busy}
              onClick={(event) => setTradeAnchor(event.currentTarget)}
            >
              <span className="build-title">
                <SwapHorizIcon fontSize="inherit" />
                Trade
              </span>
              <span className="build-cost muted">Bank</span>
            </button>
          </span>
        </Tooltip>
        <button
          className={cn("primary-action", { idle: !primary.onClick })}
          disabled={!primary.onClick || busy}
          onClick={primary.onClick}
        >
          <primary.Icon fontSize="small" />
          {primary.label}
          {primary.hint && primary.onClick && <kbd>{primary.hint}</kbd>}
        </button>
      </div>

      <TradePanel
        gameState={gs}
        anchor={tradeAnchor}
        onClose={() => setTradeAnchor(null)}
        onTrade={(action) => {
          setTradeAnchor(null);
          act(action);
        }}
      />
      <ResourceSelector
        open={selector !== null}
        onClose={() => setSelector(null)}
        options={yearOptions}
        mode={selector ?? "monopoly"}
        onSelect={(selected) => {
          const type = selector === "monopoly" ? "PLAY_MONOPOLY" : "PLAY_YEAR_OF_PLENTY";
          setSelector(null);
          act([human, type, selected] as GameAction);
        }}
      />
      <DiscardPlannerDialog
        open={discardOpen}
        onClose={() => setDiscardOpen(false)}
        onConfirm={(resources) => {
          setDiscardOpen(false);
          act(...resources.map((resource): GameAction => [human, "DISCARD_RESOURCE", resource]));
        }}
        remainingDiscardCount={gs.current_discard_count}
        discardResourceCounts={Object.fromEntries(
          RESOURCES.filter((resource) => (cards?.[resource] ?? 0) > 0).map((resource) => [resource, cards?.[resource]]),
        )}
        submitting={busy}
      />
    </div>
  );
}
