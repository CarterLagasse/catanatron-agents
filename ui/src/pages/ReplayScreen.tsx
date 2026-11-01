import { useContext, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Button, CircularProgress, Drawer, useMediaQuery } from "@mui/material";

import ACTIONS from "../actions";
import DiceRoll from "../components/DiceRoll";
import ReplayBox from "../components/ReplayBox";
import SidePanel from "../components/SidePanel";
import TopBar from "../components/TopBar";
import { store } from "../store";
import { getState } from "../utils/apiClient";
import { GameMessage, usePlayerNames } from "./GameScreen";
import ZoomableBoard from "./ZoomableBoard";
import "./GameScreen.scss";

const PLAY_INTERVAL_MS = 500;

export default function ReplayScreen() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const [latestStateIndex, setLatestStateIndex] = useState<number>(0);
  const [stateIndex, setStateIndex] = useState<number>(0);
  const [playing, setPlaying] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const isWide = useMediaQuery("(min-width: 961px)");
  const name = usePlayerNames(gameId, state.gameState);

  useEffect(() => {
    if (!gameId) return;
    dispatch({ type: ACTIONS.RESET_GAME });
    getState(gameId, "latest")
      .then((latest) => setLatestStateIndex(latest.state_index))
      .catch((error) => {
        console.error(error);
        setFailed(true);
      });
  }, [gameId, dispatch]);

  useEffect(() => {
    if (!gameId) return;
    let stale = false;
    getState(gameId, stateIndex)
      .then((gameState) => {
        if (!stale) dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      })
      .catch((error) => {
        console.error(error);
        setFailed(true);
      });
    return () => {
      stale = true;
    };
  }, [gameId, stateIndex, dispatch]);

  useEffect(() => {
    if (!playing) return;
    if (stateIndex >= latestStateIndex) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setStateIndex((index) => index + 1), PLAY_INTERVAL_MS);
    return () => clearTimeout(timer);
  }, [playing, stateIndex, latestStateIndex]);

  if (failed) {
    return (
      <GameMessage title="Couldn't load this replay">
        <p>The game server may be stopped, or this game no longer exists.</p>
        <Button variant="contained" onClick={() => setFailed(false)}>
          Try again
        </Button>
      </GameMessage>
    );
  }
  if (!state.gameState) {
    return (
      <main className="game-message">
        <CircularProgress aria-label="Loading replay" />
      </main>
    );
  }

  const panel = (
    <SidePanel
      gameState={state.gameState}
      name={name}
      gainsByRecord={state.gainsByRecord}
      revealHidden
      analysisIndex={stateIndex}
    />
  );
  return (
    <div className="game-shell">
      <TopBar
        status={{ title: "Replay", detail: `Move ${stateIndex} of ${latestStateIndex}`, tone: "wait" }}
        onOpenPanel={() => setPanelOpen(true)}
      />
      <div className="game-body">
        <div className="board-column">
          <div className="board-stage">
            <DiceRoll gameState={state.gameState} name={name} />
            <ZoomableBoard replayMode name={name} />
          </div>
          <ReplayBox
            stateIndex={stateIndex}
            latestStateIndex={latestStateIndex}
            playing={playing}
            onTogglePlay={() => {
              if (stateIndex >= latestStateIndex) setStateIndex(0);
              setPlaying((value) => !value);
            }}
            onPrevMove={() => {
              setPlaying(false);
              setStateIndex((index) => Math.max(index - 1, 0));
            }}
            onNextMove={() => {
              setPlaying(false);
              setStateIndex((index) => Math.min(index + 1, latestStateIndex));
            }}
            onSeekMove={(index) => {
              setPlaying(false);
              setStateIndex(index);
            }}
          />
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
    </div>
  );
}
