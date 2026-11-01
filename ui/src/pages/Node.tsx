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
      <svg viewBox="0 0 36 32" className={className} aria-hidden>
        <path className="piece-outline" d="M2.5 29V11.5L8.5 3h5l6 8.5V13H29l4.5 6v10Z" />
        <path className="piece-face" d="M19.5 19h14v10h-14Z" />
        <path className="piece-roof" d="M19.5 19V13H29l4.5 6Z" />
        <path className="piece-side" d="M14.5 11.5h5V29h-5Z" />
        <path className="piece-face" d="M2.5 11.5 8.5 3l6 8.5V29h-12Z" />
        <path className="piece-roof" d="M8.5 3h5l6 8.5h-5Z" />
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
