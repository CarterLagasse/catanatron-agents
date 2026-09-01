import cn from "classnames";

import { tilePixelVector, getNodeDelta, SQRT3, type CubeCoordinate } from "../utils/coordinates";
import type { Building, Color, Direction } from "../utils/api.types";

type BuildingProps<T = Color | null> = {
  color: T;
  building: T extends Color ? Building : null;
};

const FILL: Record<string, string> = {
  RED: "#fe0400",
  BLUE: "#2278ff",
  ORANGE: "#ffa500",
  WHITE: "#f2f2f2",
};

function Building({ building, color }: BuildingProps) {
  const fill = color ? (FILL[color] ?? "#999") : "#999";
  if (building === "CITY") {
    // city: slightly taller block with shallow roof, no outline
    return (
      <svg viewBox="0 0 24 18" className={cn(color, "city")} aria-hidden style={{ filter: "drop-shadow(0 2px 3px rgba(0,0,0,0.38))" }}>
        <rect x="4" y="8" width="16" height="9" fill={fill} />
        <path d="M12 1.5 L2 8 L22 8 Z" fill={fill} />
        <rect x="6" y="3.2" width="2.8" height="3" fill={fill} />
        <rect x="15.2" y="3.2" width="2.8" height="3" fill={fill} />
      </svg>
    );
  }
  // settlement: small square + triangle on top, no outline — deliberately compact
  return (
    <svg viewBox="0 0 24 24" className={cn(color, "settlement")} aria-hidden style={{ filter: "drop-shadow(0 1.6px 2px rgba(0,0,0,0.35))", overflow: "visible" }}>
      <rect x="7" y="14" width="10" height="8" fill={fill} />
      <path d="M12 4 L5 14 L19 14 Z" fill={fill} />
    </svg>
  );
}

type NodeProps = {
  center: [number, number];
  size: number;
  coordinate: CubeCoordinate;
  direction: Direction;
  building: Building | null;
  color: Color | null;
  flashing: boolean;
  onClick: React.MouseEventHandler<HTMLDivElement>;
};

export default function Node({
  center,
  size,
  coordinate,
  direction,
  building,
  color,
  flashing,
  onClick,
}: NodeProps) {
  const [centerX, centerY] = center;
  const w = SQRT3 * size;
  const h = 2 * size;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = getNodeDelta(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  const isCity = building === "CITY";
  return (
    <div
      className="node"
      style={{
        width: size * (isCity ? 0.48 : 0.36),
        height: size * (isCity ? 0.48 : 0.36),
        left: x,
        top: y,
        transform: `translateY(-50%) translateX(-50%)`,
      }}
      onClick={onClick}
    >
      {color && <Building building={building} color={color} />}
      {flashing && <div className="pulse"></div>}
      {/* {id} */}
    </div>
  );
}
