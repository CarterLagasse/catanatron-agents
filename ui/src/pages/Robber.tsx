import { tilePixelVector, type CubeCoordinate } from "../utils/coordinates";

type RobberProps = {
  center: [number, number];
  size: number;
  coordinate: CubeCoordinate;
  isDesert?: boolean;
};

export default function Robber({ center, size, coordinate, isDesert }: RobberProps) {
  const [centerX, centerY] = center;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const offsetX = isDesert ? 0 : -size * 0.49;
  const offsetY = isDesert ? 0 : size * 0.08;

  return (
    <div
      className="robber"
      style={{
        left: tileX + offsetX,
        top: tileY + offsetY,
        width: size * 0.6,
        height: size * 0.84,
      }}
      title="Robber"
      aria-label="Robber"
    >
      <svg viewBox="0 0 40 56" width="100%" height="100%" aria-hidden className="robber-art">
        <ellipse cx="20" cy="51" rx="16" ry="4.4" fill="rgba(0,0,0,0.38)" />
        <g stroke="#14181d" strokeWidth="1.5" strokeLinejoin="round">
          <ellipse cx="20" cy="46" rx="13.5" ry="5.2" fill="#2c323a" />
          <path d="M12 45C13.5 35 16 29 17 24h6c1 5 3.5 11 5 21 0 3-16 3-16 0Z" fill="#4c555f" />
          <circle cx="20" cy="14" r="9.4" fill="#4c555f" />
        </g>
        <path d="M23 25c1 5 3.6 11 5 20 0 1.6-4.4 2.5-8 2.5C24 40 23.4 31 23 25Z" fill="#343b44" />
        <path d="M14.2 8.6A8 8 0 0 1 20 6a3 3 0 0 1 0 2.4 6 6 0 0 0-3.8 2.2 2 2 0 0 1-2-2Z" fill="#7f8a96" />
      </svg>
    </div>
  );
}
