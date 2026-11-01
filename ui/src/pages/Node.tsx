import cn from "classnames";

import { tilePixelVector, getNodeDelta, SQRT3, type CubeCoordinate } from "../utils/coordinates";
import type { Building, Color, Direction } from "../utils/api.types";

type BuildingProps<T = Color | null> = {
  color: T;
  building: T extends Color ? Building : null;
};

function Building({ building, color }: BuildingProps) {
  const className = cn(color?.toLowerCase(), "piece-colors", building === "CITY" ? "city" : "settlement");
  if (building === "CITY") {
    return (
      <svg viewBox="0 0 36 34" className={className} aria-hidden>
        <path className="piece-outline" d="M3 31V12L8 2l5 10v7l7-8h8l6 8v12Z" />
        <path className="piece-side" d="M27 19h7v12h-7Z" />
        <path className="piece-face" d="M13 19l7-8 7 8v12H13Z" />
        <path className="piece-roof" d="M20 11h8l6 8h-7Z" />
        <path className="piece-face" d="M3 12h10v19H3Z" />
        <path className="piece-side" d="M9.5 12H13v19H9.5Z" />
        <path className="piece-roof" d="M3 12 8 2v10Z" />
        <path className="piece-side" d="M8 2l5 10H8Z" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 28 28" className={className} aria-hidden>
      <path className="piece-outline" d="M3.5 25V14L9.5 6.5h9L24.5 14v11Z" />
      <path className="piece-side" d="M15.5 14h9v11h-9Z" />
      <path className="piece-face" d="M3.5 14 9.5 6.5 15.5 14v11h-12Z" />
      <path className="piece-roof" d="M9.5 6.5h9l6 7.5H15.5Z" />
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
  recent: boolean;
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
  recent,
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
      className={cn("node", { clickable: flashing, recent })}
      style={{
        width: size * (isCity ? 0.74 : color ? 0.54 : 0.34),
        height: size * (isCity ? 0.74 : color ? 0.54 : 0.34),
        left: x,
        top: y,
        transform: `translateY(-50%) translateX(-50%)`,
      }}
      onClick={onClick}
      role={flashing ? "button" : undefined}
      aria-label={flashing ? "Build here" : undefined}
    >
      {color && <Building building={building} color={color} />}
      {flashing && <div className="pulse"></div>}
    </div>
  );
}
