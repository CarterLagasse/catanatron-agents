import type {
  Color,
  DevelopmentCard,
  GameAction,
  GameState,
  MaritimeTradeAction,
  MoveRobberAction,
  ResourceCard,
  TileCoordinate,
} from "./api.types";
import { getHumanColor, playerKey } from "./stateUtils";

export const RESOURCES: ResourceCard[] = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];

export const RESOURCE_NAMES: Record<ResourceCard, string> = {
  WOOD: "Wood",
  BRICK: "Brick",
  SHEEP: "Sheep",
  WHEAT: "Wheat",
  ORE: "Ore",
};

export const DEV_CARDS: { card: DevelopmentCard | "VICTORY_POINT"; name: string; short: string; hint: string }[] = [
  { card: "KNIGHT", name: "Knight", short: "Knight", hint: "Move the robber and steal a card" },
  { card: "YEAR_OF_PLENTY", name: "Year of Plenty", short: "Plenty", hint: "Take any 2 cards from the bank" },
  { card: "MONOPOLY", name: "Monopoly", short: "Monopoly", hint: "Take every card of one resource" },
  { card: "ROAD_BUILDING", name: "Road Building", short: "Roads", hint: "Build 2 roads for free" },
  { card: "VICTORY_POINT", name: "Victory Point", short: "Point", hint: "Worth 1 point, counted automatically" },
];

export const BUILD_COSTS: Record<string, ResourceCard[]> = {
  BUILD_ROAD: ["WOOD", "BRICK"],
  BUILD_SETTLEMENT: ["WOOD", "BRICK", "SHEEP", "WHEAT"],
  BUILD_CITY: ["WHEAT", "WHEAT", "ORE", "ORE", "ORE"],
  BUY_DEVELOPMENT_CARD: ["SHEEP", "WHEAT", "ORE"],
};

export type RollGains = Partial<Record<Color, Partial<Record<ResourceCard, number>>>>;

export const isHumanTurn = (gs: GameState) =>
  !gs.winning_color && !gs.bot_colors.includes(gs.current_color);

export const hasRolled = (gs: GameState) =>
  !!gs.player_state[`${playerKey(gs, gs.current_color)}_HAS_ROLLED`];

export const actionsOfType = (gs: GameState, type: GameAction[1]) =>
  gs.current_playable_actions.filter((action) => action[1] === type);

/** After Road Building the only thing a player may do is place the free roads. */
export const isFreeRoadMode = (gs: GameState) =>
  isHumanTurn(gs) &&
  gs.current_prompt === "PLAY_TURN" &&
  gs.current_playable_actions.length > 0 &&
  gs.current_playable_actions.every((action) => action[1] === "BUILD_ROAD");

export function hand(gs: GameState, color: Color): Record<ResourceCard, number> {
  const key = playerKey(gs, color);
  return Object.fromEntries(
    RESOURCES.map((resource) => [resource, gs.player_state[`${key}_${resource}_IN_HAND`] ?? 0]),
  ) as Record<ResourceCard, number>;
}

export const cardCount = (gs: GameState, color: Color) =>
  Object.values(hand(gs, color)).reduce((sum, count) => sum + count, 0);

export const devCardCount = (gs: GameState, color: Color) =>
  DEV_CARDS.reduce(
    (sum, { card }) => sum + (gs.player_state[`${playerKey(gs, color)}_${card}_IN_HAND`] ?? 0),
    0,
  );

/** Which of a cost's cards the hand can't cover, keeping duplicates. */
export function missingCards(have: Record<ResourceCard, number>, cost: ResourceCard[]) {
  const left = { ...have };
  return cost.filter((resource) => {
    if (left[resource] > 0) {
      left[resource] -= 1;
      return false;
    }
    return true;
  });
}

/** Number of dice combinations out of 36 that roll a number. */
export const pips = (number: number) => 6 - Math.abs(7 - number);

/** For each resource, the rolls out of 36 that pay a player, with the robber applied. */
export function income(gs: GameState, color: Color): Record<ResourceCard, number> {
  const result = Object.fromEntries(RESOURCES.map((r) => [r, 0])) as Record<ResourceCard, number>;
  const robber = gs.tiles.find(
    ({ coordinate }) => `${coordinate}` === `${gs.robber_coordinate}`,
  )?.tile;
  Object.values(gs.nodes).forEach((node) => {
    if (node.color !== color) return;
    (gs.adjacent_tiles[node.id] ?? []).forEach((tile) => {
      if (tile.type !== "RESOURCE_TILE") return;
      if (robber?.type === "RESOURCE_TILE" && robber.id === tile.id) return;
      result[tile.resource] += pips(tile.number) * (node.building === "CITY" ? 2 : 1);
    });
  });
  return result;
}

export type TradeOption = {
  give: ResourceCard;
  rate: number;
  gets: Partial<Record<ResourceCard, MaritimeTradeAction>>;
};

/** Groups playable bank trades by the resource given away, at the best rate on offer. */
export function tradeOptions(gs: GameState): TradeOption[] {
  const options: Partial<Record<ResourceCard, TradeOption>> = {};
  (actionsOfType(gs, "MARITIME_TRADE") as MaritimeTradeAction[]).forEach((action) => {
    const offered = action[2].slice(0, 4).filter((card) => card !== null) as ResourceCard[];
    const give = offered[0];
    const get = action[2][4] as ResourceCard;
    const existing = options[give];
    if (!existing || offered.length < existing.rate) {
      options[give] = { give, rate: offered.length, gets: { [get]: action } };
    } else if (offered.length === existing.rate) {
      existing.gets[get] = action;
    }
  });
  return RESOURCES.flatMap((resource) => options[resource] ?? []);
}

/** Pieces other players placed since this player last acted, so they can be pointed out. */
export function recentPlacements(gs: GameState) {
  const nodes = new Set<number>();
  const edges = new Set<string>();
  for (let i = gs.action_records.length - 1; i >= 0; i -= 1) {
    const action = gs.action_records[i][0];
    if (!gs.bot_colors.includes(action[0])) break;
    if (action[1] === "BUILD_SETTLEMENT" || action[1] === "BUILD_CITY") {
      nodes.add(action[2]);
    } else if (action[1] === "BUILD_ROAD") {
      edges.add(`${action[2][0]},${action[2][1]}`);
    }
  }
  return { nodes, edges };
}

export function lastRoll(gs: GameState) {
  for (let i = gs.action_records.length - 1; i >= 0; i -= 1) {
    const record = gs.action_records[i];
    if (record[0][1] === "ROLL") {
      return { color: record[0][0], dice: record[1] as [number, number], index: i };
    }
  }
  return null;
}

/** The number rolled this turn, which stays lit on the board until the turn ends. */
export function currentRollNumber(gs: GameState) {
  for (let i = gs.action_records.length - 1; i >= 0; i -= 1) {
    const record = gs.action_records[i];
    if (record[0][1] === "END_TURN") return null;
    if (record[0][1] === "ROLL") {
      const [a, b] = record[1] as [number, number];
      return a + b;
    }
  }
  return null;
}

export function rollHistogram(gs: GameState) {
  const counts = Array.from({ length: 11 }, (_, i) => ({ total: i + 2, count: 0 }));
  gs.action_records.forEach((record) => {
    if (record[0][1] === "ROLL") {
      const [a, b] = record[1] as [number, number];
      counts[a + b - 2].count += 1;
    }
  });
  return counts;
}

export const turnsPlayed = (gs: GameState) =>
  gs.action_records.filter((record) => record[0][1] === "END_TURN").length;

/** One player's totals for the end-of-game table. */
export function playerStats(gs: GameState, color: Color) {
  const key = playerKey(gs, color);
  const state = gs.player_state;
  const records = gs.action_records.filter((record) => record[0][0] === color);
  const count = (type: GameAction[1]) => records.filter((record) => record[0][1] === type).length;
  const buildings = Object.values(gs.nodes).filter((node) => node.color === color);
  return {
    points: state[`${key}_ACTUAL_VICTORY_POINTS`] ?? 0,
    settlements: buildings.filter((node) => node.building === "SETTLEMENT").length,
    cities: buildings.filter((node) => node.building === "CITY").length,
    roads: count("BUILD_ROAD"),
    longestRoad: state[`${key}_LONGEST_ROAD_LENGTH`] ?? 0,
    knights: state[`${key}_PLAYED_KNIGHT`] ?? 0,
    devCards: count("BUY_DEVELOPMENT_CARD"),
    trades: count("MARITIME_TRADE"),
    stolen: records.filter((record) => record[0][1] === "MOVE_ROBBER" && record[1]).length,
    discarded: count("DISCARD_RESOURCE"),
    income: Object.values(income(gs, color)).reduce((sum, value) => sum + value, 0),
  };
}

/** What a roll paid each player, read from the change in their hands. */
export function rollGains(previous: GameState, next: GameState): RollGains | null {
  if (next.action_records.length !== previous.action_records.length + 1) return null;
  if (next.action_records[next.action_records.length - 1][0][1] !== "ROLL") return null;
  const gains: RollGains = {};
  next.colors.forEach((color) => {
    const before = hand(previous, color);
    const after = hand(next, color);
    RESOURCES.forEach((resource) => {
      const delta = after[resource] - before[resource];
      if (delta > 0) {
        gains[color] = { ...gains[color], [resource]: delta };
      }
    });
  });
  return gains;
}

export const robberActionsFor = (gs: GameState, coordinate: TileCoordinate) =>
  (actionsOfType(gs, "MOVE_ROBBER") as MoveRobberAction[]).filter(
    (action) => `${action[2][0]}` === `${coordinate}`,
  );

export type Status = { title: string; detail: string; tone: "you" | "wait" | "done" };

export function gameStatus(
  gs: GameState,
  name: (color: Color) => string,
  buildMode: string | null,
): Status {
  if (gs.winning_color) {
    const humanWon = gs.winning_color === getHumanColor(gs);
    return {
      title: humanWon ? "You won" : `${name(gs.winning_color)} won`,
      detail: `Game over after ${turnsPlayed(gs)} turns`,
      tone: "done",
    };
  }
  if (!isHumanTurn(gs)) {
    const waitingOn = gs.current_prompt === "DISCARD" ? "is discarding" : "is playing";
    return { title: `${name(gs.current_color)} ${waitingOn}`, detail: "Watch the board", tone: "wait" };
  }
  const you = (title: string, detail: string): Status => ({ title, detail, tone: "you" });
  if (isFreeRoadMode(gs)) return you("Place a free road", "Pick a glowing edge");
  switch (gs.current_prompt) {
    case "BUILD_INITIAL_SETTLEMENT":
      return you("Place a settlement", "Pick a glowing spot");
    case "BUILD_INITIAL_ROAD":
      return you("Place a road", "Pick a glowing edge next to your settlement");
    case "DISCARD":
      return you(
        `Discard ${gs.current_discard_count} card${gs.current_discard_count === 1 ? "" : "s"}`,
        `You held more than ${gs.discard_limit} when a 7 was rolled`,
      );
    case "MOVE_ROBBER":
      return you("Move the robber", "Pick a glowing number to block it and steal a card");
    default:
  }
  if (!hasRolled(gs)) return you("Your turn: roll the dice", "Press R or the Roll button");
  if (buildMode === "ROAD") return you("Place your road", "Pick a glowing edge, or press Esc");
  if (buildMode === "SETTLEMENT") return you("Place your settlement", "Pick a glowing spot, or press Esc");
  if (buildMode === "CITY") return you("Upgrade to a city", "Pick a glowing settlement, or press Esc");
  return you("Your turn: build, trade or end", "Press E to end your turn");
}
