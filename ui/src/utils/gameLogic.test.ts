import { describe, test, expect, beforeEach } from "vitest";
import type { GameState } from "./api.types";
import {
  currentRollNumber,
  gameStatus,
  income,
  isFreeRoadMode,
  missingCards,
  pips,
  recentPlacements,
  rollGains,
  rollHistogram,
  tradeOptions,
} from "./gameLogic";
import { buildLog } from "./gameLog";
import { getGame, playerLabel, recordAgainst, saveGame, saveResult, type GameRecord } from "./history";

function gameState(overrides: Partial<GameState> = {}): GameState {
  return {
    tiles: [
      { coordinate: [0, 0, 0], tile: { id: 1, type: "RESOURCE_TILE", resource: "ORE", number: 8 } },
      { coordinate: [1, -1, 0], tile: { id: 2, type: "RESOURCE_TILE", resource: "WOOD", number: 3 } },
      { coordinate: [0, 1, -1], tile: { id: 3, type: "DESERT" } },
    ],
    adjacent_tiles: {
      5: [
        { id: 1, type: "RESOURCE_TILE", resource: "ORE", number: 8 },
        { id: 2, type: "RESOURCE_TILE", resource: "WOOD", number: 3 },
        { id: 3, type: "DESERT" },
      ],
    },
    bot_colors: ["BLUE"],
    colors: ["RED", "BLUE"],
    current_color: "RED",
    current_prompt: "PLAY_TURN",
    player_state: {},
    action_records: [],
    robber_coordinate: [0, 1, -1],
    current_discard_count: 0,
    nodes: [{ id: 5, tile_coordinate: [0, 0, 0], direction: "NORTH", building: "SETTLEMENT", color: "RED" }],
    edges: [],
    current_playable_actions: [],
    is_initial_build_phase: false,
    state_index: 0,
    vps_to_win: 10,
    discard_limit: 7,
    bank: { WOOD: 19, BRICK: 19, SHEEP: 19, WHEAT: 19, ORE: 19 },
    dev_cards_left: 25,
    ...overrides,
  };
}

const name = (color: string) => (color === "RED" ? "You" : "PSRO - Hard");

test("pips counts dice combinations", () => {
  expect([2, 3, 6, 7, 8, 12].map(pips)).toEqual([1, 2, 5, 6, 5, 1]);
});

describe("income", () => {
  test("counts a settlement once and a city twice", () => {
    expect(income(gameState(), "RED")).toMatchObject({ ORE: 5, WOOD: 2, WHEAT: 0 });
    const city = gameState({
      nodes: [{ id: 5, tile_coordinate: [0, 0, 0], direction: "NORTH", building: "CITY", color: "RED" }],
    });
    expect(income(city, "RED")).toMatchObject({ ORE: 10, WOOD: 4 });
  });
  test("the robber stops a tile paying", () => {
    expect(income(gameState({ robber_coordinate: [0, 0, 0] }), "RED")).toMatchObject({ ORE: 0, WOOD: 2 });
  });
  test("other players' buildings don't count", () => {
    expect(income(gameState(), "BLUE")).toMatchObject({ ORE: 0, WOOD: 0 });
  });
});

test("missingCards keeps duplicates in a cost", () => {
  const have = { WOOD: 0, BRICK: 0, SHEEP: 0, WHEAT: 1, ORE: 2 };
  expect(missingCards(have, ["WHEAT", "WHEAT", "ORE", "ORE", "ORE"])).toEqual(["WHEAT", "ORE"]);
});

describe("tradeOptions", () => {
  const trade = (...cards: (string | null)[]) => ["RED", "MARITIME_TRADE", cards] as any;

  test("groups by the card given and reads the rate from the action", () => {
    const options = tradeOptions(
      gameState({
        current_playable_actions: [
          trade("WOOD", "WOOD", "WOOD", "WOOD", "ORE"),
          trade("WOOD", "WOOD", "WOOD", "WOOD", "BRICK"),
          trade("ORE", "ORE", null, null, "WOOD"),
        ],
      }),
    );
    expect(options.map(({ give, rate }) => [give, rate])).toEqual([["WOOD", 4], ["ORE", 2]]);
    expect(Object.keys(options[0].gets).sort()).toEqual(["BRICK", "ORE"]);
  });

  test("offers only the best rate when a port and the bank both apply", () => {
    const options = tradeOptions(
      gameState({
        current_playable_actions: [
          trade("WOOD", "WOOD", "WOOD", "WOOD", "ORE"),
          trade("WOOD", "WOOD", "WOOD", null, "ORE"),
        ],
      }),
    );
    expect(options[0].rate).toBe(3);
    expect(options[0].gets.ORE?.[2].filter(Boolean)).toHaveLength(4);
  });
});

test("isFreeRoadMode only when roads are the sole option", () => {
  const road = ["RED", "BUILD_ROAD", [1, 2]] as any;
  expect(isFreeRoadMode(gameState({ current_playable_actions: [road] }))).toBe(true);
  expect(
    isFreeRoadMode(gameState({ current_playable_actions: [road, ["RED", "END_TURN", null]] })),
  ).toBe(false);
  expect(isFreeRoadMode(gameState({ current_color: "BLUE", current_playable_actions: [road] }))).toBe(false);
});

test("recentPlacements stops at the human's last move", () => {
  const gs = gameState({
    action_records: [
      [["BLUE", "BUILD_SETTLEMENT", 1], null],
      [["RED", "END_TURN", null], null],
      [["BLUE", "BUILD_SETTLEMENT", 2], null],
      [["BLUE", "BUILD_ROAD", [2, 3]], null],
    ],
  });
  const { nodes, edges } = recentPlacements(gs);
  expect([...nodes]).toEqual([2]);
  expect([...edges]).toEqual(["2,3"]);
});

test("currentRollNumber clears when the turn ends", () => {
  const roll: any = [["RED", "ROLL", null], [3, 5]];
  const end: any = [["RED", "END_TURN", null], null];
  expect(currentRollNumber(gameState({ action_records: [roll] }))).toBe(8);
  expect(currentRollNumber(gameState({ action_records: [roll, end] }))).toBeNull();
});

test("rollHistogram counts totals from the dice", () => {
  const histogram = rollHistogram(
    gameState({
      action_records: [
        [["RED", "ROLL", null], [3, 5]],
        [["BLUE", "ROLL", null], [4, 4]],
        [["BLUE", "ROLL", null], [1, 1]],
      ] as any,
    }),
  );
  expect(histogram.find(({ total }) => total === 8)?.count).toBe(2);
  expect(histogram[0]).toEqual({ total: 2, count: 1 });
  expect(histogram).toHaveLength(11);
});

test("rollGains reads what a roll paid, and ignores other moves", () => {
  const before = gameState({ player_state: { P0_ORE_IN_HAND: 1 } });
  const after = gameState({
    player_state: { P0_ORE_IN_HAND: 3, P1_WOOD_IN_HAND: 1 },
    action_records: [[["BLUE", "ROLL", null], [4, 4]]] as any,
  });
  expect(rollGains(before, after)).toEqual({ RED: { ORE: 2 }, BLUE: { WOOD: 1 } });
  const notRoll = gameState({ action_records: [[["RED", "END_TURN", null], null]] as any });
  expect(rollGains(before, notRoll)).toBeNull();
});

describe("gameStatus", () => {
  test("names who is playing, and who won", () => {
    expect(gameStatus(gameState({ current_color: "BLUE" }), name, null).title).toBe("PSRO - Hard is playing");
    expect(gameStatus(gameState({ winning_color: "RED" }), name, null).title).toBe("You won");
    expect(gameStatus(gameState({ winning_color: "BLUE" }), name, null).title).toBe("PSRO - Hard won");
  });
  test("tells the human what to do", () => {
    expect(gameStatus(gameState(), name, null).title).toMatch(/roll/);
    const rolled = gameState({ player_state: { P0_HAS_ROLLED: true } });
    expect(gameStatus(rolled, name, null).title).toMatch(/build/);
    expect(gameStatus(rolled, name, "CITY").title).toMatch(/city/);
    expect(gameStatus(gameState({ current_prompt: "DISCARD", current_discard_count: 1 }), name, null).title).toBe(
      "Discard 1 card",
    );
  });
});

describe("buildLog", () => {
  test("groups moves into turns, oldest first, and shows dice and gains", () => {
    const gs = gameState({
      action_records: [
        [["RED", "ROLL", null], [3, 5]],
        [["RED", "BUILD_ROAD", [1, 2]], null],
        [["RED", "END_TURN", null], null],
        [["BLUE", "ROLL", null], [1, 1]],
      ] as any,
    });
    const turns = buildLog(gs, name, { 0: { RED: { ORE: 2 } } });
    expect(turns.map((turn) => turn.label)).toEqual(["Turn 1", "Turn 2"]);
    expect(turns[0].entries[0].text).toBe("You rolled 8");
    expect(turns[0].entries[0].gains).toEqual([{ color: "RED", resources: ["ORE", "ORE"] }]);
    expect(turns[0].entries[1].text).toBe("You built a road");
    expect(turns[0].entries).toHaveLength(2);
  });

  test("puts one player's discards on a single line", () => {
    const gs = gameState({
      action_records: [
        [["RED", "ROLL", null], [3, 4]],
        [["BLUE", "DISCARD_RESOURCE", "ORE"], null],
        [["BLUE", "DISCARD_RESOURCE", "ORE"], null],
        [["RED", "DISCARD_RESOURCE", "WOOD"], null],
        [["RED", "DISCARD_RESOURCE", "BRICK"], null],
      ] as any,
    });
    const entries = buildLog(gs, name)[0].entries;
    expect(entries.map((entry) => entry.resources)).toEqual([undefined, ["ORE", "ORE"], ["WOOD", "BRICK"]]);
  });

  test("doesn't reveal which development card a bot drew", () => {
    const gs = gameState({
      action_records: [
        [["BLUE", "BUY_DEVELOPMENT_CARD", null], "VICTORY_POINT"],
        [["RED", "BUY_DEVELOPMENT_CARD", null], "KNIGHT"],
      ] as any,
    });
    const entries = buildLog(gs, name)[0].entries;
    expect(entries[0].text).toBe("PSRO - Hard bought a development card");
    expect(entries[1].text).toBe("You bought a knight card");
  });
});

describe("history", () => {
  const record = (id: string, opponent: string, humanWon?: boolean): GameRecord => ({
    id,
    createdAt: 1,
    players: ["HUMAN", opponent],
    labels: ["You", "PSRO - Hard"],
    settings: { mapTemplate: "BASE", vpsToWin: 10, discardLimit: 7, friendlyRobber: false },
    result: humanWon === undefined ? undefined : { winner: humanWon ? "RED" : "BLUE", humanWon, turns: 20 },
  });

  beforeEach(() => localStorage.clear());

  test("keeps the first result of a game", () => {
    saveGame(record("a", "JAX:psro-hard"));
    saveResult("a", { winner: "RED", humanWon: true, turns: 30 });
    saveResult("a", { winner: "BLUE", humanWon: false, turns: 31 });
    expect(getGame("a")?.result?.humanWon).toBe(true);
  });

  test("counts wins and losses per opponent, skipping unfinished games", () => {
    const games = [
      record("a", "X", true),
      record("b", "X", false),
      record("c", "X", false),
      record("d", "Y", true),
      record("e", "X"),
    ];
    expect(recordAgainst(games, "X")).toEqual({ wins: 1, losses: 2 });
  });

  test("labels players by seat, with a fallback for unknown games", () => {
    expect(playerLabel(record("a", "X"), "BLUE")).toBe("PSRO - Hard");
    expect(playerLabel(undefined, "BLUE")).toBe("Blue");
  });

  test("survives corrupted storage", () => {
    localStorage.setItem("catanatron.games", "not json");
    expect(getGame("a")).toBeUndefined();
  });
});
