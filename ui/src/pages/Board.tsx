import { SQRT3, tilePixelVector } from "../utils/coordinates";
import { currentRollNumber, recentPlacements } from "../utils/gameLogic";
import type { GameAction, GameState, TileCoordinate } from "../utils/api.types";
import Tile, { HEX_POINTS, Port } from "./Tile";
import Node from "./Node";
import Edge, { toEdgeId, type EdgeId } from "./Edge";
import Robber from "./Robber";

import "./Board.scss";

// Room left around the island for the coast and port signs, in hex radii.
const COAST_MARGIN = 1.0;

/**
 * The largest hex radius that fits the island and its ports, where `rings` counts the outer
 * ring of water. The land is (2R - 1) rows of 1.5 * size plus half a size tall, and (2R - 1)
 * tiles of SQRT3 * size wide. Math comes from https://www.redblobgames.com/grids/hexagons/.
 */
function computeDefaultSize(width: number, height: number, rings: number): number {
  const rows = 2 * rings - 1;
  return Math.min(
    height / (1.5 * rows + 0.5 + 2 * COAST_MARGIN),
    width / (SQRT3 * rows + 2 * COAST_MARGIN),
  );
}

type BoardProps = {
  width: number;
  height: number;
  onNodeAction: (action: GameAction) => void;
  onEdgeAction: (action: GameAction) => void;
  handleTileClick: (coordinate: TileCoordinate) => void;
  nodeActions?: Record<number, GameAction>;
  edgeActions?: Record<EdgeId, GameAction>;
  replayMode: boolean;
  gameState: GameState;
  isMovingRobber: boolean;
  robberCoordinates?: Set<string>;
};

export default function Board({
  width,
  height,
  onNodeAction,
  onEdgeAction,
  handleTileClick,
  nodeActions,
  edgeActions,
  replayMode,
  gameState,
  isMovingRobber,
  robberCoordinates,
}: BoardProps) {
  const containerWidth = width;
  const containerHeight = height;
  const center: [number, number] = [containerWidth / 2, containerHeight / 2];
  const rings = Math.max(...gameState.tiles.map(({ coordinate }) => Math.max(...coordinate.map(Math.abs))));
  const size = computeDefaultSize(containerWidth, containerHeight, rings);
  if (!size) {
    return null;
  }
  const rolledNumber = currentRollNumber(gameState);
  const recent = recentPlacements(gameState);
  const robberKey = `${gameState.robber_coordinate}`;

  const tiles = gameState.tiles.map(({ coordinate, tile }) => (
    <Tile
      key={`${coordinate}`}
      center={center}
      coordinate={coordinate}
      tile={tile}
      size={size}
      rolled={tile.type === "RESOURCE_TILE" && tile.number === rolledNumber}
      blocked={`${coordinate}` === robberKey}
      onClick={
        isMovingRobber && robberCoordinates?.has(`${coordinate}`)
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
        recent={recent.nodes.has(id)}
        onClick={() => {
          if (nodeActions && id in nodeActions) onNodeAction(nodeActions[id]);
        }}
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
        recent={recent.edges.has(toEdgeId(id))}
        onClick={() => {
          const action = edgeActions?.[toEdgeId(id)];
          if (action) onEdgeAction(action);
        }}
      />
    )
  );

  const land = gameState.tiles.flatMap(({ coordinate, tile }) => {
    if (tile.type !== "RESOURCE_TILE" && tile.type !== "DESERT") return [];
    const [x, y] = tilePixelVector(coordinate, size, center[0], center[1]);
    return [{ key: `${coordinate}`, transform: `translate(${x} ${y}) scale(${size / 100})` }];
  });
  // Every land hex outlined in the same colour merges into one rounded coastline.
  const coast = (className: string, widthInRadii: number) => (
    <g className={`coast ${className}`} strokeWidth={widthInRadii * 100}>
      {land.map(({ key, transform }) => (
        <polygon key={key} points={HEX_POINTS} transform={transform} />
      ))}
    </g>
  );

  return (
    <div className="board show">
      <svg className="board-art" width={containerWidth} height={containerHeight}>
        <defs>
          <clipPath id="hex-clip">
            <polygon points={HEX_POINTS} />
          </clipPath>
        </defs>
        {coast("coast-shallow", 0.86)}
        {coast("coast-foam", 0.38)}
        {coast("coast-wet", 0.3)}
        {gameState.tiles.flatMap(({ coordinate, tile }) =>
          tile.type === "PORT"
            ? [
                <Port
                  key={`${coordinate}`}
                  center={center}
                  coordinate={coordinate}
                  size={size}
                  direction={tile.direction}
                  resource={tile.resource}
                />,
              ]
            : [],
        )}
        {coast("coast-sand", 0.22)}
        {tiles}
      </svg>
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
