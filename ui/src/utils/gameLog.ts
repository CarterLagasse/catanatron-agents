import type {
  Color,
  GameActionRecord,
  GameState,
  MaritimeTradeAction,
  ResourceCard,
} from "./api.types";
import { RESOURCE_NAMES, type RollGains } from "./gameLogic";
import { getTileString, findTileByCoordinate } from "./promptUtils";

export type LogEntry = {
  index: number;
  color: Color;
  text: string;
  dice?: [number, number];
  resources?: ResourceCard[];
  gains?: { color: Color; resources: ResourceCard[] }[];
};

export type LogTurn = {
  key: string;
  label: string;
  color: Color;
  entries: LogEntry[];
};

const lower = (resource: ResourceCard) => RESOURCE_NAMES[resource].toLowerCase();

function describe(
  gs: GameState,
  record: GameActionRecord,
  name: (color: Color) => string,
): Pick<LogEntry, "text" | "resources"> {
  const [action, result] = record;
  const who = name(action[0]);
  const humanInvolved = (color?: Color | null) =>
    !gs.bot_colors.includes(action[0]) || (!!color && !gs.bot_colors.includes(color));
  switch (action[1]) {
    case "DISCARD_RESOURCE": {
      const card = (action[2] ?? result) as ResourceCard | null;
      return { text: `${who} discarded`, resources: card ? [card] : undefined };
    }
    case "PLAY_KNIGHT_CARD":
      return { text: `${who} played a knight` };
    case "PLAY_ROAD_BUILDING":
      return { text: `${who} played Road Building` };
    case "PLAY_MONOPOLY":
      return { text: `${who} played Monopoly on ${lower(action[2])}` };
    case "PLAY_YEAR_OF_PLENTY":
      return { text: `${who} played Year of Plenty for`, resources: [...action[2]] };
    case "MOVE_ROBBER": {
      const tile = findTileByCoordinate(gs, action[2][0]);
      const where = getTileString(tile).toLowerCase();
      const victim = (action[2][1] ?? null) as Color | null;
      const stolen = result as ResourceCard | null;
      if (victim) {
        return stolen && humanInvolved(victim)
          ? { text: `${who} moved the robber to ${where}, stole from ${name(victim)}:`, resources: [stolen] }
          : { text: `${who} moved the robber to ${where}, stole from ${name(victim)}` };
      }
      return { text: `${who} moved the robber to ${where}` };
    }
    case "MARITIME_TRADE": {
      const value = (action as MaritimeTradeAction)[2];
      const given = value.slice(0, 4).filter((card) => card !== null) as ResourceCard[];
      return {
        text: `${who} traded ${given.length} ${lower(given[0])} for 1 ${lower(value[4] as ResourceCard)}`,
      };
    }
    default:
      return { text: `${who}: ${String(action[1]).replaceAll("_", " ").toLowerCase()}` };
  }
}

/** A move that joins the turn's list of things built or bought, if it is one. */
function listedMove(gs: GameState, [action, result]: GameActionRecord) {
  switch (action[1]) {
    case "BUILD_ROAD":
      return { verb: "built", one: "road", many: "roads" };
    case "BUILD_SETTLEMENT":
      return { verb: "built", one: "settlement", many: "settlements" };
    case "BUILD_CITY":
      return { verb: "built", one: "city", many: "cities" };
    case "BUY_DEVELOPMENT_CARD": {
      // The card drawn is in the record, but is secret unless it's the player's own.
      const card = !gs.bot_colors.includes(action[0]) && result
        ? `${String(result).replaceAll("_", " ").toLowerCase()} card`
        : "development card";
      return { verb: "bought", one: card, many: `${card}s` };
    }
    default:
      return null;
  }
}

function listText(items: Map<string, { many: string; count: number }>) {
  const parts = [...items].map(([one, { many, count }]) => (count === 1 ? `a ${one}` : `${count} ${many}`));
  return parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}` : parts[0];
}

function gainsList(gains: RollGains | undefined) {
  if (!gains) return undefined;
  const list = Object.entries(gains).map(([color, cards]) => ({
    color: color as Color,
    resources: Object.entries(cards ?? {}).flatMap(([resource, count]) =>
      Array<ResourceCard>(count as number).fill(resource as ResourceCard),
    ),
  }));
  return list.length > 0 ? list : undefined;
}

/** Turns, oldest first, each with its moves in the order they happened. */
export function buildLog(
  gs: GameState,
  name: (color: Color) => string,
  gainsByRecord: Record<number, RollGains> = {},
): LogTurn[] {
  const turns: LogTurn[] = [];
  let current: LogTurn | null = null;
  let turnNumber = 0;
  let lists = new Map<string, { entry: LogEntry; items: Map<string, { many: string; count: number }> }>();
  for (let index = 0; index < gs.action_records.length; index += 1) {
    const record = gs.action_records[index];
    const action = record[0];
    const color = action[0];
    // The setup phase has no END_TURN records, so its first roll ends it.
    if (current?.label === "Setup" && action[1] === "ROLL") {
      current = null;
    }
    if (!current) {
      const isSetup = turns.length === 0 && action[1] === "BUILD_SETTLEMENT";
      if (!isSetup) turnNumber += 1;
      current = {
        key: `${index}`,
        color,
        label: isSetup ? "Setup" : `Turn ${turnNumber}`,
        entries: [],
      };
      turns.push(current);
      lists = new Map();
    }
    if (action[1] === "END_TURN") {
      current = null;
    } else if (action[1] === "ROLL") {
      const dice = record[1] as [number, number];
      current.entries.push({
        index,
        color,
        text: `${name(color)} rolled ${dice[0] + dice[1]}`,
        dice,
        gains: gainsList(gainsByRecord[index]),
      });
    } else if (listedMove(gs, record)) {
      const { verb, one, many } = listedMove(gs, record)!;
      const key = `${color}:${verb}`;
      let list = lists.get(key);
      if (!list) {
        list = { entry: { index, color, text: "" }, items: new Map() };
        lists.set(key, list);
        current.entries.push(list.entry);
      }
      const item = list.items.get(one) ?? { many, count: 0 };
      item.count += 1;
      list.items.set(one, item);
      list.entry.text = `${name(color)} ${verb} ${listText(list.items)}`;
    } else {
      const entry = describe(gs, record, name);
      const previous = current.entries[current.entries.length - 1];
      if (
        action[1] === "DISCARD_RESOURCE" &&
        previous?.color === color &&
        gs.action_records[previous.index][0][1] === "DISCARD_RESOURCE"
      ) {
        previous.resources = [...(previous.resources ?? []), ...(entry.resources ?? [])];
      } else {
        current.entries.push({ index, color, ...entry });
      }
    }
  }
  return turns;
}
