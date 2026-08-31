import { Paper } from "@mui/material";
import { motion, AnimatePresence } from "framer-motion";
import { type PlayerState } from "../utils/api.types";
import { type Card } from "../utils/api.types";

// TODO - do we need to split the SCSS for this component?
import "./PlayerStateBox.scss";

const CARD_META: Record<string, { cls: string; short: string }> = {
  WOOD: { cls: "wood-cards", short: "Wo" },
  BRICK: { cls: "brick-cards", short: "Br" },
  SHEEP: { cls: "sheep-cards", short: "Sh" },
  WHEAT: { cls: "wheat-cards", short: "Wh" },
  ORE: { cls: "ore-cards", short: "Or" },
};

function AnimatedCard({ amount, cls, label, title }: { amount: number; cls: string; label: string; title?: string }) {
  return (
    <motion.div
      layout
      initial={{ scale: 0.85, y: 6, opacity: 0 }}
      animate={{ scale: 1, y: 0, opacity: 1 }}
      exit={{ scale: 0.85, opacity: 0 }}
      transition={{ type: "spring", stiffness: 420, damping: 22 }}
      className={`${cls} center-text card`}
      title={title}
    >
      <Paper>
        <motion.span key={amount} initial={{ scale: 1.35 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 500, damping: 18 }}>
          {amount}
        </motion.span>
        <small style={{ fontSize: "0.52rem", opacity: 0.9, lineHeight: 1 }}>{label}</small>
      </Paper>
    </motion.div>
  );
}

export default function ResourceCards({ playerState, playerKey }: { playerState: PlayerState; playerKey: string }) {
  const amount = (card: Card) => playerState[`${playerKey}_${card}_IN_HAND`] ?? 0;
  return (
    <div className="resource-cards" title="Resource Cards">
      <AnimatePresence initial={false}>
        {amount("WOOD") !== 0 && <AnimatedCard key="WOOD" amount={amount("WOOD")} cls={CARD_META.WOOD.cls} label={CARD_META.WOOD.short} title="Wood" />}
        {amount("BRICK") !== 0 && <AnimatedCard key="BRICK" amount={amount("BRICK")} cls={CARD_META.BRICK.cls} label={CARD_META.BRICK.short} title="Brick" />}
        {amount("SHEEP") !== 0 && <AnimatedCard key="SHEEP" amount={amount("SHEEP")} cls={CARD_META.SHEEP.cls} label={CARD_META.SHEEP.short} title="Sheep" />}
        {amount("WHEAT") !== 0 && <AnimatedCard key="WHEAT" amount={amount("WHEAT")} cls={CARD_META.WHEAT.cls} label={CARD_META.WHEAT.short} title="Wheat" />}
        {amount("ORE") !== 0 && <AnimatedCard key="ORE" amount={amount("ORE")} cls={CARD_META.ORE.cls} label={CARD_META.ORE.short} title="Ore" />}
      </AnimatePresence>
      <div className="separator" />
      <AnimatePresence initial={false}>
        {amount("VICTORY_POINT") !== 0 && (
          <AnimatedCard key="VP" amount={amount("VICTORY_POINT")} cls="dev-cards" label="VP" title={`${amount("VICTORY_POINT")} Victory Point`} />
        )}
        {amount("KNIGHT") !== 0 && (
          <AnimatedCard key="KN" amount={amount("KNIGHT")} cls="dev-cards" label="KN" title={`${amount("KNIGHT")} Knight`} />
        )}
        {amount("MONOPOLY") !== 0 && (
          <AnimatedCard key="MO" amount={amount("MONOPOLY")} cls="dev-cards" label="MO" title={`${amount("MONOPOLY")} Monopoly`} />
        )}
        {amount("YEAR_OF_PLENTY") !== 0 && (
          <AnimatedCard key="YP" amount={amount("YEAR_OF_PLENTY")} cls="dev-cards" label="YP" title={`${amount("YEAR_OF_PLENTY")} Year of Plenty`} />
        )}
        {amount("ROAD_BUILDING") !== 0 && (
          <AnimatedCard key="RB" amount={amount("ROAD_BUILDING")} cls="dev-cards" label="RB" title={`${amount("ROAD_BUILDING")} Road Building`} />
        )}
      </AnimatePresence>
    </div>
  );
}
