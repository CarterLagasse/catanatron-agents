import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Button, CircularProgress, Drawer, useMediaQuery } from "@mui/material";

import ACTIONS from "../actions";
import ActionDock from "../components/ActionDock";
import DiceRoll from "../components/DiceRoll";
import GameOverDialog from "../components/GameOverDialog";
import GameStatsDialog from "../components/GameStatsDialog";
import SidePanel from "../components/SidePanel";
import TopBar, { type BotSpeed } from "../components/TopBar";
import { CLASSIC_UI_URL } from "../configuration";
import { store } from "../store";
import type { Color, GameState } from "../utils/api.types";
import { type StateIndex, getState, postAction } from "../utils/apiClient";
import { gameStatus, turnsPlayed } from "../utils/gameLogic";
import { getGame, loadSetting, playerLabel, saveResult, saveSetting } from "../utils/history";
import { getHumanColor } from "../utils/stateUtils";
import { useWinOddsTracker } from "../utils/winOdds";
import ZoomableBoard from "./ZoomableBoard";
import "./GameScreen.scss";

// Least time a bot's move stays on screen before its next one.
const BOT_PACE_MS: Record<BotSpeed, number> = { slow: 1200, normal: 550, fast: 120 };

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Names by color: "You" for the human, otherwise as chosen when the game was created. */
export function usePlayerNames(gameId: string | undefined, gameState: GameState | null) {
  return useMemo(() => {
    const record = getGame(gameId);
    const human = gameState ? getHumanColor(gameState) : undefined;
    return (color: Color) => (color === human ? "You" : playerLabel(record, color));
  }, [gameId, gameState?.bot_colors.join()]);
}

export function GameMessage({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <main className="game-message">
      <h2>{title}</h2>
      {children}
      <Link to="/">Back to the start</Link>
    </main>
  );
}

export default function GameScreen({ replayMode }: { replayMode: boolean }) {
  const { gameId, stateIndex } = useParams();
  const { state, dispatch } = useContext(store);
  const [isBotThinking, setIsBotThinking] = useState(false);
  const [botFailed, setBotFailed] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  const [gameOverOpen, setGameOverOpen] = useState(false);
  const [statsOpen, setStatsOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [speed, setSpeed] = useState<BotSpeed>(() => loadSetting("botSpeed", "normal"));
  const [reveal, setReveal] = useState(() => loadSetting<string>("revealHidden", "no") === "yes");
  const isWide = useMediaQuery("(min-width: 961px)");
  const botMoveInFlight = useRef(false);
  const currentGame = useRef(gameId);
  currentGame.current = gameId;
  const gameState = state.gameState;

  const load = useCallback(async () => {
    if (!gameId) return;
    setLoadFailed(false);
    try {
      const loaded = await getState(gameId, stateIndex as StateIndex);
      if (currentGame.current === gameId) {
        dispatch({ type: ACTIONS.SET_GAME_STATE, data: loaded });
      }
    } catch (error) {
      console.error("Couldn't load game", error);
      setLoadFailed(true);
    }
  }, [gameId, stateIndex, dispatch]);

  // Never show or play on the previous game's board while this one loads.
  useEffect(() => {
    dispatch({ type: ACTIONS.RESET_GAME });
    setGameOverOpen(false);
    setStatsOpen(false);
    load();
  }, [load, dispatch]);

  useWinOddsTracker(gameId, gameState, !replayMode);

  // Play the bots' moves one at a time, at a pace that can be followed.
  useEffect(() => {
    if (!gameState || replayMode || !gameId || botFailed || botMoveInFlight.current) return;
    if (!gameState.bot_colors.includes(gameState.current_color) || gameState.winning_color) return;
    botMoveInFlight.current = true;
    setIsBotThinking(true);
    (async () => {
      const started = Date.now();
      try {
        const next = await postAction(gameId);
        await sleep(Math.max(0, BOT_PACE_MS[speed] - (Date.now() - started)));
        if (currentGame.current === gameId) {
          dispatch({ type: ACTIONS.SET_GAME_STATE, data: next });
        }
      } catch (error) {
        console.error("Bot move failed", error);
        setBotFailed(true);
      } finally {
        botMoveInFlight.current = false;
        setIsBotThinking(false);
      }
    })();
  }, [gameId, replayMode, gameState, botFailed, speed, dispatch]);

  // Record the finish once, and celebrate it.
  const winner = gameState?.winning_color;
  useEffect(() => {
    if (!winner || !gameState || !gameId || replayMode) return;
    const human = getHumanColor(gameState);
    saveResult(gameId, {
      winner,
      humanWon: human ? winner === human : null,
      turns: turnsPlayed(gameState),
    });
    setGameOverOpen(true);
  }, [winner, gameId, replayMode]);

  const name = usePlayerNames(gameId, gameState);

  if (loadFailed) {
    return (
      <GameMessage title="Couldn't load this game">
        <p>The game server may be stopped, or this game no longer exists.</p>
        <Button variant="contained" onClick={load}>
          Try again
        </Button>
      </GameMessage>
    );
  }
  if (!gameState) {
    return (
      <main className="game-message">
        <CircularProgress aria-label="Loading game" />
      </main>
    );
  }

  const record = getGame(gameId);
  const opponents = gameState.colors.filter((color) => color !== getHumanColor(gameState));
  const panel = (
    <SidePanel
      gameState={gameState}
      name={name}
      gainsByRecord={state.gainsByRecord}
      revealHidden={reveal || replayMode}
      onSeeStats={replayMode ? undefined : () => setStatsOpen(true)}
    />
  );

  return (
    <div className="game-shell">
      <TopBar
        status={
          replayMode
            ? { title: `Viewing move ${stateIndex}`, detail: "Read only", tone: "wait" }
            : gameStatus(gameState, name, state.buildMode)
        }
        matchup={record ? `vs ${opponents.map(name).join(", ")}` : undefined}
        speed={speed}
        onSpeedChange={(value) => {
          setSpeed(value);
          saveSetting("botSpeed", value);
        }}
        revealHidden={reveal}
        onRevealChange={(value) => {
          setReveal(value);
          saveSetting("revealHidden", value ? "yes" : "no");
        }}
        replayHref={`/replays/${gameId}`}
        classicHref={CLASSIC_UI_URL ? `${CLASSIC_UI_URL}/games/${gameId}` : undefined}
        onOpenPanel={() => setPanelOpen(true)}
      />
      <div className="game-body">
        <div className="board-column">
          <div className="board-stage">
            <DiceRoll gameState={gameState} name={name} />
            <ZoomableBoard replayMode={replayMode} name={name} />
            {botFailed && (
              <div className="bot-error" role="alert">
                <span>Lost contact with the game server during the bot's move.</span>
                <Button size="small" variant="contained" onClick={() => setBotFailed(false)}>
                  Retry
                </Button>
              </div>
            )}
          </div>
          {!replayMode && <ActionDock isBotThinking={isBotThinking} />}
        </div>
        {isWide ? (
          <aside className="side-panel">{panel}</aside>
        ) : (
          <Drawer
            anchor="right"
            open={panelOpen}
            onClose={() => setPanelOpen(false)}
            slotProps={{ paper: { className: "side-drawer" } }}
          >
            {panel}
          </Drawer>
        )}
      </div>
      {winner && !replayMode && (
        <GameOverDialog
          gameState={gameState}
          name={name}
          open={gameOverOpen}
          onClose={() => setGameOverOpen(false)}
          onSeeStats={() => {
            setGameOverOpen(false);
            setStatsOpen(true);
          }}
        />
      )}
      {winner && !replayMode && (
        <GameStatsDialog
          gameState={gameState}
          name={name}
          open={statsOpen}
          onClose={() => setStatsOpen(false)}
          onBack={() => {
            setStatsOpen(false);
            setGameOverOpen(true);
          }}
        />
      )}
    </div>
  );
}
