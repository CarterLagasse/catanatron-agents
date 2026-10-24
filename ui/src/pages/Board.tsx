import classnames from "classnames";

import { SQRT3 } from "../utils/coordinates";
import type { GameAction, GameState, TileCoordinate } from "../utils/api.types";
import Tile from "./Tile";
import Node from "./Node";
import Edge, { toEdgeId, type EdgeId } from "./Edge";
import Robber from "./Robber";

import "./Board.scss";

/**
 * This uses the formulas: W = SQRT3 * size and H = 2 * size.
 * Math comes from https://www.redblobgames.com/grids/hexagons/.
 */
function computeDefaultSize(divWidth: number, divHeight: number): number {
  const numLevels = 6; // 3 rings + 1/2 a tile for the outer water ring
  // divHeight = numLevels * (3h/4) + (h/4), implies:
  const maxSizeThatRespectsHeight = (4 * divHeight) / (3 * numLevels + 1) / 2;
  const correspondingWidth = SQRT3 * maxSizeThatRespectsHeight;
  let size: number;
  if (numLevels * correspondingWidth < divWidth) {
    // thus complete board would fit if we pick size based on height (height is limiting factor)
    size = maxSizeThatRespectsHeight;
  } else {
    // we'll have to decide size based on width.
    const maxSizeThatRespectsWidth = divWidth / numLevels / SQRT3;
    size = maxSizeThatRespectsWidth;
  }
  return size; // CSS padding on .board-container handles sand rim consistently
}

type BoardProps = {
  width: number;
  height: number;
  buildOnNodeClick: (id: number, action?: GameAction) => React.MouseEventHandler<HTMLDivElement>;
  buildOnEdgeClick: (id: [number, number], action?: GameAction) => React.MouseEventHandler<HTMLDivElement>;
  handleTileClick: (coordinate: TileCoordinate) => void;
  nodeActions?: Record<number, GameAction>;
  edgeActions?: Record<EdgeId, GameAction>;
  replayMode: boolean;
  gameState: GameState;
  isMobile: boolean;
  show: boolean;
  isMovingRobber: boolean;
  robberCoordinates?: Set<string>;
}

export default function Board({
  width,
  height,
  buildOnNodeClick,
  buildOnEdgeClick,
  handleTileClick,
  nodeActions,
  edgeActions,
  replayMode,
  gameState,
  isMobile,
  show,
  isMovingRobber,
  robberCoordinates,
}: BoardProps) {
  // Board container is .board-container - keep in sync with fixed CSS heights:
  // PhaseBanner 40px + margins 14px + logo ~30px + CostStrip 36px+8px = ~128px header
  // .state-summary 56px + .actions-toolbar 72px = 128px toolbars
  // .board-container has 10px padding for sand rim, so subtract 20px + 24px gutters
  const TOOLBAR_HEIGHT = 128; // .state-summary 56 + .actions-toolbar 72 (fixed via min-height)
  const HEADER_HEIGHT = 128; // logo ~30 + PhaseBanner 40+14 + CostStrip 36+8 (all fixed)
  const isDesktop = isMobile; // useMediaQuery(up("md")) true = desktop
  const DRAWER_OFFSET = isDesktop ? 560 : 0;
  const containerHeight = Math.max(360, height - TOOLBAR_HEIGHT - HEADER_HEIGHT - 24 - 20);
  const containerWidth = Math.max(360, width - DRAWER_OFFSET - 24 - 20);
  const center: [number, number] = [containerWidth / 2, containerHeight / 2];
  const size = computeDefaultSize(containerWidth, containerHeight);
  if (!size) {
    return null;
  }

  const tiles = gameState.tiles.map(({ coordinate, tile }) => (
    <Tile
      key={`${coordinate}`}
      center={center}
      coordinate={coordinate}
      tile={tile}
      size={size}
      flashing={!!robberCoordinates?.has(`${coordinate}`)}
      onClick={
        !isMovingRobber || robberCoordinates?.has(`${coordinate}`)
          ? () => handleTileClick(coordinate)
          : undefined
      }
    />
  ));
  const nodes = Object.values(gameState.nodes).map(
    ({ color, building, direction, tile_coordinate, id }) => (
      <Node
        key={id}
        center={center}
        size={size}
        coordinate={tile_coordinate}
        direction={direction}
        building={building}
        color={color}
        flashing={!replayMode && !!nodeActions && id in nodeActions}
        onClick={buildOnNodeClick(
          id,
          nodeActions ? nodeActions[id] : undefined
        )}
      />
    )
  );
  const edges = Object.values(gameState.edges).map(
    ({ color, direction, tile_coordinate, id }) => (
      <Edge
        id={`${id[0]},${id[1]}`}
        key={`${id}`}
        center={center}
        size={size}
        coordinate={tile_coordinate}
        direction={direction}
        color={color}
        flashing={!!edgeActions && toEdgeId(id) in edgeActions}
        onClick={buildOnEdgeClick(id, edgeActions ? edgeActions[toEdgeId(id)]: undefined)}
      />
    )
  );

  // Island: 19 background hexes behind land tiles (3-4-5-4-3), then white + blue beach rims
  const landTiles = gameState.tiles.filter(({ tile }) => tile.type === "RESOURCE_TILE" || tile.type === "DESERT");
  const toRad = (d: number) => (d * Math.PI) / 180;
  const vertAngles = [-30, 30, 90, 150, 210, 270];
  const hexPoly = (cx: number, cy: number, r: number) =>
    vertAngles.map((a) => `${cx + r * Math.cos(toRad(a))},${cy + r * Math.sin(toRad(a))}`).join(" ");
  const bgHexes = landTiles.map(({ coordinate }) => {
    const q = coordinate[0];
    const r = coordinate[2];
    const cx = size * (SQRT3 * q + (SQRT3 / 2) * r) + center[0];
    const cy = size * (3 / 2) * r + center[1];
    return { key: `${coordinate}`, cx, cy };
  });

  return (
    <div className={classnames("board", { show })}>
      <svg
        className="island"
        aria-hidden
        style={{ position: "absolute", left: 0, top: 0, width: containerWidth, height: containerHeight, pointerEvents: "none", zIndex: 0, overflow: "visible" }}
        width={containerWidth}
        height={containerHeight}
      >
        {bgHexes.map(({ key, cx, cy }) => (
          <polygon key={`${key}-blue`} points={hexPoly(cx, cy, size * 1.22)} fill="#7ec8f0" stroke="none" />
        ))}
        {bgHexes.map(({ key, cx, cy }) => (
          <polygon key={`${key}-white`} points={hexPoly(cx, cy, size * 1.12)} fill="#ffffff" stroke="none" />
        ))}
        {bgHexes.map(({ key, cx, cy }) => (
          <polygon key={`${key}-sand`} points={hexPoly(cx, cy, size * 1.06)} fill="#f0d9a0" stroke="none" />
        ))}
      </svg>
      {tiles}
      {edges}
      {nodes}
      <Robber
        center={center}
        size={size}
        coordinate={gameState.robber_coordinate}
        isDesert={gameState.tiles.find(t => t.coordinate[0] === gameState.robber_coordinate[0] && t.coordinate[1] === gameState.robber_coordinate[1] && t.coordinate[2] === gameState.robber_coordinate[2])?.tile.type === "DESERT"}
      />
    </div>
  );
}
