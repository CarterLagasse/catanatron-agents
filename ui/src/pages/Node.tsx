import cn from "classnames";

import { CityArt, SettlementArt } from "../components/PieceArt";
import { tilePixelVector, getNodeDelta, SQRT3, type CubeCoordinate } from "../utils/coordinates";
import type { Building, Color, Direction } from "../utils/api.types";

type BuildingProps<T = Color | null> = {
  color: T;
  building: T extends Color ? Building : null;
};

function Building({ building, color }: BuildingProps) {
  const colors = cn(color?.toLowerCase(), "piece-colors");
  return building === "CITY" ? (
    <CityArt className={cn(colors, "city")} />
  ) : (
    <SettlementArt className={cn(colors, "settlement")} />
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
