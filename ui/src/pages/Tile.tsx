import cn from "classnames";

import ResourceIcon from "../components/ResourceIcon";
import { pips } from "../utils/gameLogic";
import { tilePixelVector, type CubeCoordinate } from "../utils/coordinates";
import type { Direction, ResourceCard, Tile as TileType } from "../utils/api.types";
import { TERRAIN_BASE, TerrainArt, type Terrain } from "./TileArt";
import "./Tile.scss";

export const HEX_POINTS = "0,-100 86.6,-50 86.6,50 0,100 -86.6,50 -86.6,-50";

// Tiles are drawn slightly small so the sand shows between them.
const TILE_SCALE = 0.945;

function NumberToken({
  number,
  rolled,
  blocked,
  target,
}: {
  number: number;
  rolled?: boolean;
  blocked?: boolean;
  target?: boolean;
}) {
  const dots = pips(number);
  const hot = number === 6 || number === 8;
  return (
    <g className={cn("number-token", { rolled, blocked, target, hot })}>
      {target && <circle className="token-target" r="36" />}
      {rolled && <circle className="token-rolled" r="38" />}
      <circle cy="3.5" r="31" className="token-shadow" />
      <circle r="31" className="token-face" />
      <text y="-1" textAnchor="middle" dominantBaseline="central" className="token-number">
        {number}
      </text>
      <g className="token-pips">
        {Array.from({ length: dots }, (_, i) => (
          <circle key={i} cx={(i - (dots - 1) / 2) * 6.6} cy="20" r="2.5" />
        ))}
      </g>
    </g>
  );
}

type TileProps = {
  center: [number, number];
  coordinate: CubeCoordinate;
  tile: TileType;
  size: number;
  onClick?: () => void;
  rolled?: boolean;
  blocked?: boolean;
};

export default function Tile({ center, coordinate, tile, size, onClick, rolled, blocked }: TileProps) {
  if (tile.type !== "RESOURCE_TILE" && tile.type !== "DESERT") return null;
  const [x, y] = tilePixelVector(coordinate, size, center[0], center[1]);
  const terrain: Terrain = tile.type === "DESERT" ? "DESERT" : tile.resource;
  const { fill, edge } = TERRAIN_BASE[terrain];
  return (
    <g
      className={cn("tile", `tile--${terrain.toLowerCase()}`, { rolled, targetable: !!onClick })}
      transform={`translate(${x} ${y}) scale(${(size / 100) * TILE_SCALE})`}
      role={onClick ? "button" : undefined}
      aria-label={onClick ? "Move the robber here" : undefined}
      onClick={onClick}
    >
      <polygon points={HEX_POINTS} transform="translate(0 5)" fill={edge} stroke={edge} className="tile-body" />
      <polygon points={HEX_POINTS} fill={fill} stroke={fill} className="tile-body tile-face" />
      <g clipPath="url(#hex-clip)">
        <TerrainArt terrain={terrain} flip={tile.id % 2 === 1} />
      </g>
      {tile.type === "RESOURCE_TILE" && (
        <NumberToken number={tile.number} rolled={rolled} blocked={blocked} target={!!onClick} />
      )}
      {tile.type === "DESERT" && onClick && <circle className="token-target" r="36" />}
    </g>
  );
}

// Which way a port's land edge faces from the centre of its water hex, in degrees clockwise from east.
const PORT_ANGLE: Partial<Record<Direction, number>> = {
  EAST: 0,
  SOUTHEAST: 60,
  SOUTHWEST: 120,
  WEST: 180,
  NORTHWEST: 240,
  NORTHEAST: 300,
};

const PORT_RESOURCES: ResourceCard[] = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];

/** A harbour: two piers from the port's coastal spots out to a sign in the water. */
export function Port({
  center,
  coordinate,
  size,
  direction,
  resource,
}: {
  center: [number, number];
  coordinate: CubeCoordinate;
  size: number;
  direction: Direction;
  resource: ResourceCard | null;
}) {
  const [wx, wy] = tilePixelVector(coordinate, size, center[0], center[1]);
  const angle = PORT_ANGLE[direction] ?? 0;
  const at = (degrees: number, radius: number) => {
    const radians = (degrees * Math.PI) / 180;
    return [wx + radius * Math.cos(radians), wy + radius * Math.sin(radians)];
  };
  const [signX, signY] = at(angle, size * 0.2);
  const corners = [at(angle - 30, size), at(angle + 30, size)];
  const specific = !!resource && PORT_RESOURCES.includes(resource);
  return (
    <g className="port" aria-label={specific ? `2 to 1 ${resource} port` : "3 to 1 port"}>
      {corners.map(([x, y]) => (
        <g key={`${x},${y}`} className="pier" style={{ strokeWidth: size * 0.13 }}>
          <line x1={x} y1={y} x2={signX} y2={signY} className="pier-edge" />
          <line x1={x} y1={y} x2={signX} y2={signY} className="pier-deck" style={{ strokeWidth: size * 0.09 }} />
          <line
            x1={x}
            y1={y}
            x2={signX}
            y2={signY}
            className="pier-planks"
            style={{ strokeWidth: size * 0.09, strokeDasharray: `${size * 0.02} ${size * 0.06}` }}
          />
        </g>
      ))}
      <g transform={`translate(${signX} ${signY}) scale(${size / 100})`}>
        <circle cy="4" r="35" className="port-shadow" />
        <circle r="35" className="port-rim" />
        <circle r="29" className="port-face" />
        {specific ? (
          <>
            <g transform="translate(-14 -25)">
              <ResourceIcon resource={resource} size={28} />
            </g>
            <text y="17" textAnchor="middle" dominantBaseline="central" className="port-rate">
              2:1
            </text>
          </>
        ) : (
          <>
            <path d="M0-23V-3L14-3ZM-3-19V-3H-14Z" className="port-sail" />
            <path d="M-17 0H17L11 7H-11Z" className="port-hull" />
            <text y="17" textAnchor="middle" dominantBaseline="central" className="port-rate">
              3:1
            </text>
          </>
        )}
      </g>
    </g>
  );
}
