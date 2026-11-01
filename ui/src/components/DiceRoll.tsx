import { motion } from "framer-motion";

import type { Color, GameState } from "../utils/api.types";
import { lastRoll } from "../utils/gameLogic";
import "./DiceRoll.scss";

// Pip positions on a 3x3 grid, numbered left to right, top to bottom.
const FACES: Record<number, number[]> = {
  1: [5],
  2: [1, 9],
  3: [1, 5, 9],
  4: [1, 3, 7, 9],
  5: [1, 3, 5, 7, 9],
  6: [1, 3, 4, 6, 7, 9],
};

export function Die({ value, size = 34 }: { value: number; size?: number }) {
  return (
    <span className="die" style={{ width: size, height: size }}>
      {Array.from({ length: 9 }, (_, i) => (
        <span key={i} className={FACES[value]?.includes(i + 1) ? "pip" : undefined} />
      ))}
    </span>
  );
}

/** The latest roll. Stays up so it can be checked at any time, and shakes when it changes. */
export default function DiceRoll({
  gameState,
  name,
}: {
  gameState: GameState;
  name: (color: Color) => string;
}) {
  const roll = lastRoll(gameState);
  if (!roll) return null;
  const total = roll.dice[0] + roll.dice[1];
  return (
    <motion.div
      key={roll.index}
      className="dice-tray"
      role="status"
      aria-label={`${name(roll.color)} rolled ${total}`}
      initial={{ scale: 0.8, rotate: -6, opacity: 0 }}
      animate={{ scale: 1, rotate: 0, opacity: 1 }}
      transition={{ type: "spring", stiffness: 420, damping: 16 }}
    >
      <span className="dice-pair">
        <Die value={roll.dice[0]} />
        <Die value={roll.dice[1]} />
      </span>
      <span className="dice-text">
        <strong className={total === 7 ? "seven" : undefined}>{total}</strong>
        <small>{name(roll.color)}</small>
      </span>
    </motion.div>
  );
}
