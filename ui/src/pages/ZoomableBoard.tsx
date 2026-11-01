import { useContext, useEffect, useRef, useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { Dialog, Button } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";

import "./Board.scss";
import { store, type BuildMode } from "../store";
import useGameActions from "../hooks/useGameActions";
import Board from "./Board";
import type {
  BuildRoadAction,
  BuildSettlementAction,
  Color,
  GameAction,
  GameState,
  MoveRobberAction,
  TileCoordinate,
} from "../utils/api.types";
import {
  actionsOfType,
  isFreeRoadMode,
  isHumanTurn,
  robberActionsFor,
} from "../utils/gameLogic";

/** Returns actions by node id, e.g. {3: ["BLUE", "BUILD_CITY", 3]}. */
function buildNodeActions(gs: GameState, buildMode: BuildMode) {
  const nodeActions: Record<number, GameAction> = {};
  if (!isHumanTurn(gs)) return nodeActions;
  const type = gs.is_initial_build_phase || buildMode === "SETTLEMENT"
    ? "BUILD_SETTLEMENT"
    : buildMode === "CITY"
      ? "BUILD_CITY"
      : null;
  if (type) {
    (actionsOfType(gs, type) as BuildSettlementAction[]).forEach((action) => {
      nodeActions[action[2]] = action;
    });
  }
  return nodeActions;
}

function buildEdgeActions(gs: GameState, buildMode: BuildMode) {
  const edgeActions: Record<`${number},${number}`, GameAction> = {};
  if (!isHumanTurn(gs)) return edgeActions;
  if (gs.is_initial_build_phase || buildMode === "ROAD" || isFreeRoadMode(gs)) {
    (actionsOfType(gs, "BUILD_ROAD") as BuildRoadAction[]).forEach((action) => {
      edgeActions[`${action[2][0]},${action[2][1]}`] = action;
    });
  }
  return edgeActions;
}

function useElementSize() {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return { ref, ...size };
}

export default function ZoomableBoard({
  replayMode,
  name,
}: {
  replayMode: boolean;
  name: (color: Color) => string;
}) {
  const { state } = useContext(store);
  const { act } = useGameActions();
  const { ref, width, height } = useElementSize();
  const [victimChoices, setVictimChoices] = useState<MoveRobberAction[]>([]);
  const gameState = state.gameState as GameState;

  const nodeActions = replayMode ? {} : buildNodeActions(gameState, state.buildMode);
  const edgeActions = replayMode ? {} : buildEdgeActions(gameState, state.buildMode);
  const isMovingRobber =
    !replayMode && isHumanTurn(gameState) && gameState.current_prompt === "MOVE_ROBBER";
  const robberCoordinates = new Set(
    isMovingRobber
      ? (actionsOfType(gameState, "MOVE_ROBBER") as MoveRobberAction[]).map(
          (action) => `${action[2][0]}`,
        )
      : [],
  );

  const handleTileClick = (coordinate: TileCoordinate) => {
    if (!isMovingRobber) return;
    const actions = robberActionsFor(gameState, coordinate) as MoveRobberAction[];
    if (actions.length === 1) {
      act(actions[0]);
    } else if (actions.length > 1) {
      setVictimChoices(actions);
    }
  };

  return (
    <div className="board-container" ref={ref}>
      {width > 0 && height > 0 && (
        <TransformWrapper
          minScale={0.6}
          maxScale={3}
          initialScale={1}
          limitToBounds={false}
          wheel={{ step: 0.12 }}
          doubleClick={{ mode: "reset" }}
          panning={{ velocityDisabled: true }}
        >
          {({ zoomIn, zoomOut, resetTransform }) => (
            <>
              <div className="board-controls">
                <button className="board-ctrl" onClick={() => zoomIn()} aria-label="Zoom in">
                  <AddIcon fontSize="small" />
                </button>
                <button className="board-ctrl" onClick={() => zoomOut()} aria-label="Zoom out">
                  <RemoveIcon fontSize="small" />
                </button>
                <button className="board-ctrl" onClick={() => resetTransform()} aria-label="Reset view">
                  <CenterFocusStrongIcon fontSize="small" />
                </button>
              </div>
              <TransformComponent
                wrapperStyle={{ width: "100%", height: "100%" }}
                contentStyle={{ width: "100%", height: "100%" }}
              >
                <Board
                  width={width}
                  height={height}
                  onNodeAction={(action) => act(action)}
                  onEdgeAction={(action) => act(action)}
                  handleTileClick={handleTileClick}
                  nodeActions={nodeActions}
                  edgeActions={edgeActions}
                  replayMode={replayMode}
                  gameState={gameState}
                  isMovingRobber={isMovingRobber}
                  robberCoordinates={robberCoordinates}
                />
              </TransformComponent>
            </>
          )}
        </TransformWrapper>
      )}
      <Dialog
        open={victimChoices.length > 0}
        onClose={() => setVictimChoices([])}
        slotProps={{ paper: { className: "victim-dialog" } }}
      >
        <h3>Steal from who?</h3>
        {victimChoices.map((action) => (
          <Button
            key={String(action[2][1])}
            variant="contained"
            onClick={() => {
              setVictimChoices([]);
              act(action);
            }}
          >
            {action[2][1] ? name(action[2][1] as Color) : "Nobody"}
          </Button>
        ))}
      </Dialog>
    </div>
  );
}
