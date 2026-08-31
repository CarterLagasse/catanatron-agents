import cn from "classnames";
import Paper from "@mui/material/Paper";

import "./Tile.scss";
import { SQRT3, tilePixelVector, calculatePortPosition, APOTHEM_FACTOR } from "../utils/coordinates";
import {
  type Direction,
  type ResourceCard,
  type Tile,
} from "../utils/api.types";

type NumberTokenProps = {
  number: number;
  className?: string;
  style?: Partial<React.CSSProperties>;
  flashing?: boolean;
};

export function NumberToken({
  number,
  className,
  style,
  flashing,
}: NumberTokenProps) {
  return (
    <Paper
      elevation={3}
      className={cn("number-token", className, {
        flashing: flashing,
        "number-token-red": number === 6 || number === 8,
      })}
      style={style}
    >
      <div>{number}</div>
      <div className="pips">{numberToPips(number)}</div>
    </Paper>
  );
}

const numberToPips = (number: number) => {
  switch (number) {
    case 2:
    case 12:
      return "•";
    case 3:
    case 11:
      return "••";
    case 4:
    case 10:
      return "•••";
    case 5:
    case 9:
      return "••••";
    case 6:
    case 8:
      return "•••••";
    default:
      return "";
  }
};

// true Catan terrain colors matched to goal.png reference image
export const TERRAIN_COLORS: Record<string, string> = {
  WOOD: "#1e5d29",      // forest dark green (from #1e5d29, #004405)
  BRICK: "#e26427",     // hills orange-red (from #e26427, #e46829)
  SHEEP: "#8bb60e",     // pasture bright yellow-green (from #8bb60e, #8eb60c)
  WHEAT: "#f0b615",     // fields golden yellow (from #f0b615, #f6a03f)
  ORE: "#9ba09c",       // mountains light gray (from #9ba09c, #a1a7a3)
  DESERT: "#dcb163",    // desert golden sand (from #dcb163, #fbd6c2)
  WATER: "#1564b4",     // sea dark blue (from #1564b4)
  PORT_WATER: "#53cdfe", // port/water hex bright blue (from #53cdfe)
};

const PORT_RESOURCES: ResourceCard[] = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];

// kept for board overlay (bridges/ships sit outside clipped hex)

export const PortBoat = ({
  resource,
  style,
  size,
}: {
  resource: ResourceCard;
  style?: Partial<React.CSSProperties>;
  size: number;
}) => {
  const ratio = PORT_RESOURCES.includes(resource) ? "2:1" : "3:1";
  return (
    <div className="port-boat" style={style} aria-label={`Port ${ratio}`}>
      <svg width={Math.round(size * 0.55)} height={Math.round(size * 0.38)} viewBox="0 0 64 36" aria-hidden>
        <path d="M6 22 Q32 30 58 22 L52 27 Q32 34 12 27 Z" fill="#5d4037" stroke="#3e2723" strokeWidth="1.2" />
        <path d="M8 22 Q32 28 56 22" fill="none" stroke="#8d6e63" strokeWidth="1" opacity="0.7" />
        <rect x="14" y="15" width="36" height="7" rx="1.5" fill="#d7ccc8" stroke="#3e2723" strokeWidth="1" />
        <rect x="30.8" y="4" width="2.4" height="12" fill="#3e2723" />
        <path d="M33.2 4.5 L50 10 L33.2 15 Z" fill="#fff8e1" stroke="#3e2723" strokeWidth="0.9" />
        <path d="M12 11 L12 6 L18 8.5 Z" fill={PORT_RESOURCES.includes(resource) ? TERRAIN_COLORS[resource as string] : "#fff"} stroke="#3e2723" strokeWidth="0.7" />
      </svg>
      <span className="port-badge">{ratio}</span>
    </div>
  );
};

type TileProps = {
  center: any;
  coordinate: any;
  tile: Tile;
  size: any;
  onClick?: React.MouseEventHandler<HTMLDivElement>;
  flashing: boolean;
};

export default function Tile({
  center,
  coordinate,
  tile,
  size,
  onClick,
  flashing,
}: TileProps) {
  const w = SQRT3 * size;
  const h = 2 * size;
  const GAP = 0.975; // sand gap 2.5% - chill intersections, not floating
  const displayW = w * GAP;
  const displayH = h * GAP;
  const [centerX, centerY] = center;
  const [x, y] = tilePixelVector(coordinate, size, centerX, centerY);

  if (tile.type === "WATER") return null;

  let contents: React.ReactNode = null;
  let hexFill: string | undefined;
  let showMotif: ResourceCard | null = null;

  if (tile.type === "RESOURCE_TILE") {
    contents = <NumberToken number={tile.number} flashing={flashing} />;
    hexFill = TERRAIN_COLORS[tile.resource];
    showMotif = tile.resource as ResourceCard;
  } else if (tile.type === "DESERT") {
    hexFill = TERRAIN_COLORS.DESERT;
    showMotif = "DESERT" as ResourceCard;
  } else if (tile.type === "PORT") {
    // Harbor boat in ocean — no hex, just boat centered around board
    const [wx, wy] = tilePixelVector(coordinate, size, centerX, centerY);
    const vx = centerX - wx;
    const vy = centerY - wy;
    const len = Math.hypot(vx, vy) || 1;
    const apothem = size * APOTHEM_FACTOR;
    const dist = apothem * 0.45;
    const bx = wx + (vx / len) * dist;
    const by = wy + (vy / len) * dist;
    return (
      <div
        key={`${coordinate}`}
        className="tile--port-boat"
        style={{ position: "absolute", left: bx, top: by, transform: "translate(-50%,-50%)", zIndex: 5 }}
        aria-hidden
      >
        <PortBoat resource={tile.resource as ResourceCard} size={size} style={{ left: 0, top: 0 }} />
      </div>
    );
  }

  const terrainClass =
    tile.type === "RESOURCE_TILE"
      ? `tile--${tile.resource.toLowerCase()}`
      : tile.type === "DESERT"
        ? "tile--desert"
        : tile.type === "PORT"
          ? `tile--port`
          : "";

  return (
    <div
      key={coordinate}
      className={cn("tile", terrainClass)}
      style={{
        left: x - displayW / 2,
        top: y - displayH / 2,
        width: displayW,
        height: displayH,
        background: hexFill,
      }}
      onClick={onClick}
    >
      {showMotif && (
        <span className="tile-motif" aria-hidden>
          {showMotif === "WOOD" && <i className="motif motif--wood" />}
          {showMotif === "BRICK" && <i className="motif motif--brick" />}
          {showMotif === "SHEEP" && <i className="motif motif--sheep" />}
          {showMotif === "WHEAT" && <i className="motif motif--wheat" />}
          {showMotif === "ORE" && <i className="motif motif--ore" />}
          {showMotif === "DESERT" && <i className="motif motif--desert" />}
        </span>
      )}
      {contents}
    </div>
  );
}
