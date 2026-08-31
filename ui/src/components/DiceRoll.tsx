import { useEffect, useState, useContext } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { store } from "../store";
import "./DiceRoll.scss";

function diceFaces(value: number | null) {
  if (value === null || value === undefined) return null;
  const dots = Array.from({ length: value }, (_, i) => i);
  return dots;
}

export default function DiceRoll() {
  const { state } = useContext(store);
  const gs = state.gameState;
  const [rolling, setRolling] = useState(false);
  const [dice, setDice] = useState<[number, number] | null>(null);
  const [visible, setVisible] = useState(false);

  // Detect last roll from action_records
  useEffect(() => {
    if (!gs) return;
    const last = gs.action_records[gs.action_records.length - 1];
    if (!last || !Array.isArray(last) || last[1] !== "ROLL") return;
    const sum: number = last[2] as number;
    // Derive two dice that sum to `sum` (approx, for visual only)
    // 7 is 3+4 most common colonist split; otherwise random split.
    let d1 = Math.floor(sum / 2);
    let d2 = sum - d1;
    if (d1 < 1) d1 = 1;
    if (d2 < 1) d2 = 1;
    if (d1 > 6) { d1 = 6; d2 = sum - 6; }
    if (d2 > 6) { d2 = 6; d1 = sum - 6; }

    setDice([d1, d2]);
    setVisible(true);
    setRolling(true);
    const t1 = setTimeout(() => setRolling(false), 650);
    const t2 = setTimeout(() => setVisible(false), 2800);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [gs?.action_records?.length]);

  // Also watch for non-roll to keep hidden
  if (!gs || !dice || !visible) return null;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="dice-overlay"
          initial={{ opacity: 0, y: 8, scale: 0.92 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.92 }}
          transition={{ duration: 0.22 }}
        >
          <div className="dice-row">
            {[dice[0], dice[1]].map((val, idx) => (
              <motion.div
                key={`${val}-${idx}-${rolling ? "r" : "s"}`}
                className={`die ${rolling ? "rolling" : ""}`}
                animate={rolling ? { rotate: 360 * (idx === 0 ? 1 : -1), scale: [1, 1.08, 1] } : { rotate: 0 }}
                transition={rolling ? { duration: 0.6, ease: "easeInOut" } : { duration: 0.2 }}
              >
                <div className={`die-face die-${val}`}>
                  {Array.from({ length: val }).map((_, i) => (
                    <span key={i} className="pip" />
                  ))}
                </div>
              </motion.div>
            ))}
            <motion.div className="dice-sum" initial={{ scale: 0.8 }} animate={{ scale: 1 }}>
              {dice[0] + dice[1]}
            </motion.div>
          </div>
          <div className="dice-label">{dice[0] + dice[1] === 7 ? "Robber!" : `Rolled ${dice[0] + dice[1]}`}</div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
