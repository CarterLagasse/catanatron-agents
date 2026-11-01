import cn from "classnames";
import { Tooltip } from "@mui/material";

import type { GameState } from "../utils/api.types";
import { RESOURCES, RESOURCE_NAMES } from "../utils/gameLogic";
import { DevCardArt } from "./PieceArt";
import ResourceIcon from "./ResourceIcon";
import "./BankTray.scss";

const LOW_STOCK = 3;

/** How many of each card the bank still holds, kept on screen beside the board. */
export default function BankTray({ gameState: gs }: { gameState: GameState }) {
  return (
    <section className="bank-tray" aria-label="Cards left in the bank">
      <h4>Bank</h4>
      <ul>
        {RESOURCES.map((resource) => {
          const left = gs.bank[resource];
          return (
            <Tooltip key={resource} arrow title={`${left} ${RESOURCE_NAMES[resource]} left in the bank`}>
              <li className={cn({ low: left <= LOW_STOCK, out: left === 0 })}>
                <ResourceIcon resource={resource} size={24} />
                <b>{left}</b>
              </li>
            </Tooltip>
          );
        })}
        <Tooltip arrow title={`${gs.dev_cards_left} development cards left to buy`}>
          <li className={cn("bank-dev", { out: gs.dev_cards_left === 0 })}>
            <DevCardArt />
            <b>{gs.dev_cards_left}</b>
          </li>
        </Tooltip>
      </ul>
    </section>
  );
}
