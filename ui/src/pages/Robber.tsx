import {
  SQRT3,
  tilePixelVector,
  type CubeCoordinate,
} from "../utils/coordinates";
import { Paper } from "@mui/material";

type RobberProps = {
  center: [number, number];
  size: number;
  coordinate: CubeCoordinate;
};

export default function Robber({ center, size, coordinate }: RobberProps) {
  const [centerX, centerY] = center;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);

  return (
    <div
      className="robber"
      style={{
        left: tileX,
        top: tileY,
        width: size * 0.52,
        height: size * 0.62,
      }}
      title="Robber"
      aria-label="Robber"
    >
      <svg viewBox="0 0 24 32" width="100%" height="100%" aria-hidden style={{ display: "block", filter: "drop-shadow(0 2px 3px rgba(0,0,0,0.35))" }}>
        {/* minimalist pawn - flat, single tone, no belt/highlight/base ring */}
        <path d="M12 3.5 A5.5 5.5 0 1 0 12 14.5 A5.5 5.5 0 1 0 12 3.5 Z M9.4 14.8 L7.6 24.4 L12 26.2 L16.4 24.4 L14.6 14.8 Z M7.8 26.2 Q12 28.2 16.2 26.2 L15.6 27.6 Q12 29 8.4 27.6 Z" fill="#3a3f45" stroke="#1a1d22" strokeWidth="1.1" strokeLinejoin="round" />
      </svg>
    </div>
  );
}
