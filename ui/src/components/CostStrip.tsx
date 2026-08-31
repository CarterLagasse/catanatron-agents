import { useContext } from "react";
import { store } from "../store";
import ResourceIcon, { type Resource } from "./ResourceIcon";
import HomeIcon from "@mui/icons-material/Home";
import LocationCityIcon from "@mui/icons-material/LocationCity";
import SignpostIcon from "@mui/icons-material/Signpost";
import StyleIcon from "@mui/icons-material/Style";
import "./CostStrip.scss";

type Cost = { label: string; cost: Resource[]; action: string; Icon: React.ElementType };

const COSTS: Cost[] = [
  { label: "Road", cost: ["WOOD", "BRICK"], action: "BUILD_ROAD", Icon: SignpostIcon },
  { label: "Settlement", cost: ["WOOD", "BRICK", "SHEEP", "WHEAT"], action: "BUILD_SETTLEMENT", Icon: HomeIcon },
  { label: "City", cost: ["WHEAT", "WHEAT", "ORE", "ORE", "ORE"], action: "BUILD_CITY", Icon: LocationCityIcon },
  { label: "Dev Card", cost: ["SHEEP", "WHEAT", "ORE"], action: "BUY_DEVELOPMENT_CARD", Icon: StyleIcon },
];

export default function CostStrip() {
  const { state } = useContext(store);
  const gs = state.gameState;
  if (!gs) return null;

  const playable = new Set(
    gs.is_initial_build_phase
      ? []
      : gs.current_playable_actions.map((a: any) => a[1] as string)
  );

  return (
    <div className="cost-strip" aria-label="Building costs">
      {COSTS.map((c) => {
        const affordable = playable.has(c.action);
        return (
          <div key={c.action} className={`cost-item ${affordable ? "affordable" : "poor"}`} title={`${c.label}: ${c.cost.join(" + ")}`}>
            <c.Icon className="cost-icon" style={{ fontSize: 16 }} />
            <span className="cost-label">{c.label}</span>
            <span className="cost-resources">
              {c.cost.map((r, i) => (
                <ResourceIcon key={`${c.action}-${r}-${i}`} resource={r} size={14} />
              ))}
            </span>
            <span className={`cost-dot ${affordable ? "yes" : "no"}`} />
          </div>
        );
      })}
    </div>
  );
}
