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
  const is3to1 = !PORT_RESOURCES.includes(resource);
  const ratio = is3to1 ? "3:1" : "2:1";
  const sailH = Math.round(size * 0.28);
  const sailW = Math.round(size * 0.22);
  const hullW = Math.round(size * 0.44);
  const hullH = Math.round(size * 0.12);
  const mastH = Math.round(size * 0.03);
  const iconScale = is3to1 ? 0.30 : 0.26;
  const ratioScale = is3to1 ? 0.28 : 0.22;
  const mastExposed = Math.round(size * 0.06);
  const totalH = sailH + mastExposed + hullH;
  return (
    <div className="port-boat" style={style} aria-label={`Port ${ratio}`}>
      <div style={{ position: "relative", width: Math.max(sailW, hullW), height: totalH }}>
        {/* sail at top */}
        <div style={{
          position: "absolute",
          top: 4,
          left: "50%",
          transform: "translateX(-50%)",
          width: sailW,
          height: sailH,
          background: "#ffffff",
          border: "1.2px solid #3e2723",
          borderRadius: 2,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 1,
          filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.25))",
        }}>
          <svg width={Math.round(sailH * iconScale)} height={Math.round(sailH * iconScale)} viewBox="0 0 24 24" aria-hidden>
            {resource === "WOOD" && <path d="M12 3 L6 14 H8 L5 19 H19 L16 14 H18 Z" fill="#1e5d29" stroke="#004405" strokeWidth="1.8" />}
            {resource === "BRICK" && <rect x="3" y="5" width="18" height="14" rx="1.5" fill="#e26427" stroke="#b84d1e" strokeWidth="1.8" />}
            {resource === "SHEEP" && <ellipse cx="12" cy="13" rx="8" ry="5.5" fill="#8bb60e" stroke="#6a8a0b" strokeWidth="1.8" />}
            {resource === "WHEAT" && <path d="M12 19 V5" stroke="#a86a0a" strokeWidth="2.5" strokeLinecap="round" />}
            {resource === "ORE" && <path d="M4 18 L12 4 L20 18 Z" fill="#9ba09c" stroke="#7d827f" strokeWidth="1.8" />}
          </svg>
          <span style={{
            marginTop: -2,
            fontSize: Math.round(sailH * ratioScale),
            fontWeight: 900,
            color: "#1e3d1e",
            lineHeight: 1,
            fontFamily: "Nunito, sans-serif",
          }}>{ratio}</span>
        </div>
        {/* mast: from bottom of sail down into hull */}
        <div style={{
          position: "absolute",
          left: "50%",
          top: sailH + 4,
          width: 2,
          height: mastExposed + hullH - 8,
          background: "#3e2723",
          transform: "translateX(-50%)",
        }} />
        {/* hull at bottom */}
        <svg width={hullW} height={hullH} viewBox="0 0 64 14" style={{ position: "absolute", bottom: 0, left: "50%", transform: "translateX(-50%)" }} aria-hidden>
          <path d="M4 1 Q32 12 60 1 L54 8 Q32 14 10 8 Z" fill="#5d4037" stroke="#3e2723" strokeWidth="1.2" />
        </svg>
      </div>
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
  const GAP = 0.93; // tiles touch exactly, sand visible only on outer edge
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
