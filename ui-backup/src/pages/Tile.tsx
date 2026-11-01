import cn from "classnames";
import Paper from "@mui/material/Paper";

import "./Tile.scss";
import { SQRT3, tilePixelVector, calculatePortPosition, APOTHEM_FACTOR } from "../utils/coordinates";
import {
  type Direction,
  type ResourceCard,
  type Tile,
} from "../utils/api.types";

// Extended type for tile motifs (includes DESERT which is not a ResourceCard)
type TileMotif = ResourceCard | "DESERT";

type NumberTokenProps = {
  number: number;
  className?: string;
  style?: Partial<React.CSSProperties>;
  flashing?: boolean;
  size?: number;
};

export function NumberToken({
  number,
  className,
  style,
  flashing,
  size,
}: NumberTokenProps) {
  const s = size || 50;
  const tokenSize = Math.round(s * 0.62);
  const numFontSize = s * 0.3;
  const pipFontSize = s * 0.15;
  return (
    <Paper
      elevation={3}
      className={cn("number-token", className, {
        flashing: flashing,
        "number-token-red": number === 6 || number === 8,
      })}
      style={{ width: tokenSize, height: tokenSize, fontSize: numFontSize, ...style }}
    >
      <div>{number}</div>
      <div className="pips" style={{ fontSize: pipFontSize }}>{numberToPips(number)}</div>
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
  const sailH = Math.round(size * 0.36);
  const sailW = Math.round(size * 0.28);
  const hullW = Math.round(size * 0.56);
  const hullH = Math.round(size * 0.16);
  const mastH = Math.round(size * 0.04);
  const iconScale = is3to1 ? 0.44 : 0.6;
  const ratioScale = is3to1 ? 0.40 : 0.4;
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
            {resource === "SHEEP" && <ellipse cx="12" cy="13" rx="8" ry="5.5" fill="#ffffff" stroke="#999" strokeWidth="1.8" />}
            {resource === "WHEAT" && <g><path d="M12 20 V8" stroke="#a86a0a" strokeWidth="2.2" strokeLinecap="round" /><ellipse cx="12" cy="5.5" rx="3" ry="4" fill="#f0b615" stroke="#a86a0a" strokeWidth="1.4" /></g>}
            {resource === "ORE" && <path d="M4 18 L12 4 L20 18 Z" fill="#9ba09c" stroke="#7d827f" strokeWidth="1.8" />}
          </svg>
          <span style={{
            marginTop: is3to1 ? -12 : -2,
            marginLeft: is3to1 ? 1 : 1,
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

  let contents: React.ReactNode = null;
  let hexFill: string | undefined;
  let showMotif: TileMotif | null = null;
  let terrainClass = "";

  switch (tile.type) {
    case "WATER": {
      return null;
    }
    case "RESOURCE_TILE": {
      contents = <NumberToken number={tile.number} flashing={flashing} size={size} />;
      hexFill = TERRAIN_COLORS[tile.resource];
      showMotif = tile.resource as ResourceCard;
      terrainClass = `tile--${tile.resource.toLowerCase()}`;
      break;
    }
    case "DESERT": {
      hexFill = TERRAIN_COLORS.DESERT;
      showMotif = "DESERT";
      terrainClass = "tile--desert";
      break;
    }
    case "PORT": {
      // Harbor boat in ocean — no hex, just boat centered around board
      const [wx, wy] = tilePixelVector(coordinate, size, centerX, centerY);
      const vx = centerX - wx;
      const vy = centerY - wy;
      const len = Math.hypot(vx, vy) || 1;
      const nx = vx / len;
      const ny = vy / len;
      const apothem = size * APOTHEM_FACTOR;
      const dist = apothem * 0.04;
      const bx = wx - nx * dist;
      const by = wy - ny * dist;
      // Find the two hex corners closest to the direction toward center
      const centerAngle = Math.atan2(ny, nx);
      const vertexAngles = [-Math.PI/2, -Math.PI/6, Math.PI/6, Math.PI/2, 5*Math.PI/6, 7*Math.PI/6];
      const verts = vertexAngles.map((a) => {
        const da = ((a - centerAngle + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
        return { x: wx + size * Math.cos(a), y: wy + size * Math.sin(a), da: Math.abs(da) };
      }).sort((a, b) => a.da - b.da);
      const shoreOffset = 4;
      // Assign left/right based on cross product
      const sorted = verts.slice(0, 2);
      const cross0 = (sorted[0].x - wx) * ny - (sorted[0].y - wy) * nx;
      const left = cross0 > 0 ? sorted[0] : sorted[1];
      const right = cross0 > 0 ? sorted[1] : sorted[0];
      const cLeft = { x: left.x - nx * shoreOffset, y: left.y - ny * shoreOffset };
      const cRight = { x: right.x - nx * shoreOffset, y: right.y - ny * shoreOffset };
      // End 20px closer to shore than before, left corner goes to left side, right to right
      const endOffset = 20;
      const perpX = -ny;
      const perpY = nx;
      const hullW = Math.round(size * 0.56);
      const plankSpread = Math.round(hullW * 0.42);
      const endLeftX = bx + nx * endOffset - perpX * plankSpread;
      const endLeftY = by + ny * endOffset - perpY * plankSpread;
      const endRightX = bx + nx * endOffset + perpX * plankSpread;
      const endRightY = by + ny * endOffset + perpY * plankSpread;
      const dockW = Math.max(4, size * 0.06 + 6);
      // Center boat on hull, not bounding box
      const sailH = Math.round(size * 0.36);
      const mastExposed = Math.round(size * 0.06);
      const hullH = Math.round(size * 0.16);
      const totalH = sailH + mastExposed + hullH;
      const hullCenterOffset = (totalH - hullH) / 2;
      return (
        <div key={`${coordinate}`} style={{ position: "absolute", left: 0, top: 0, width: 0, height: 0, zIndex: 2 }} aria-hidden>
          {/* dock plank left: left corner → left side of boat */}
          <svg style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }}>
            <line x1={cLeft.x} y1={cLeft.y} x2={endLeftX} y2={endLeftY} stroke="#3e2723" strokeWidth={dockW + 3} strokeLinecap="butt" />
            <line x1={cLeft.x} y1={cLeft.y} x2={endLeftX} y2={endLeftY} stroke="#5d4037" strokeWidth={dockW} strokeLinecap="butt" />
          </svg>
          {/* dock plank right: right corner → right side of boat */}
          <svg style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }}>
            <line x1={cRight.x} y1={cRight.y} x2={endRightX} y2={endRightY} stroke="#3e2723" strokeWidth={dockW + 3} strokeLinecap="butt" />
            <line x1={cRight.x} y1={cRight.y} x2={endRightX} y2={endRightY} stroke="#5d4037" strokeWidth={dockW} strokeLinecap="butt" />
          </svg>
          <div
            className="tile--port-boat"
            style={{ position: "absolute", left: bx, top: by, transform: `translate(-50%, calc(-50% - ${hullCenterOffset}px))` }}
          >
            <PortBoat resource={tile.resource as ResourceCard} size={size} style={{ left: 0, top: 0 }} />
          </div>
        </div>
      );
    }
  }

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
