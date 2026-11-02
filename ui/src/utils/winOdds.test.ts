import { describe, expect, it } from "vitest";

import type { GameState } from "./api.types";
import { oddsSeries, saveOdds, turnEndIndexes } from "./winOdds";

const record = (color: string, type: string) => [[color, type, null], null];

const game = (types: string[], winning_color?: string) =>
  ({
    colors: ["RED", "BLUE"],
    winning_color,
    action_records: types.map((type) => record("RED", type)),
  }) as unknown as GameState;

const even = { RED: 50, BLUE: 50 } as never;

describe("turnEndIndexes", () => {
  it("gives the state right after each END_TURN", () => {
    expect(turnEndIndexes(game(["BUILD_ROAD", "ROLL", "END_TURN", "ROLL", "BUILD_ROAD", "END_TURN"]))).toEqual([3, 6]);
  });

  it("has nothing during setup", () => {
    expect(turnEndIndexes(game(["BUILD_SETTLEMENT", "BUILD_ROAD"]))).toEqual([]);
  });
});

describe("oddsSeries", () => {
  const types = ["ROLL", "END_TURN", "ROLL", "END_TURN", "ROLL", "END_TURN"];

  it("numbers points by turn and skips turns not worked out yet", () => {
    expect(oddsSeries(game(types), { 2: even, 6: even }).map((point) => point.turn)).toEqual([1, 3]);
  });

  it("ignores odds from moves that aren't the end of a turn", () => {
    expect(oddsSeries(game(types), { 1: even })).toEqual([]);
  });

  it("stops at the move being replayed", () => {
    expect(oddsSeries(game(types.slice(0, 2)), { 2: even, 4: even }).map((point) => point.turn)).toEqual([1]);
  });

  it("ends a finished game on the result, in the turn it was won", () => {
    const series = oddsSeries(game([...types, "ROLL", "BUILD_CITY"], "BLUE"), { 2: even });
    expect(series[series.length - 1]).toEqual({ turn: 4, odds: { RED: 0, BLUE: 100 }, final: true });
  });
});

describe("saveOdds", () => {
  it("keeps odds per game and survives being called again", () => {
    saveOdds("a", 2, even);
    saveOdds("a", 4, even);
    saveOdds("b", 2, even);
    const stored = JSON.parse(localStorage.getItem("catanatron.winOdds") ?? "{}");
    expect(Object.keys(stored.a)).toEqual(["2", "4"]);
    expect(Object.keys(stored.b)).toEqual(["2"]);
  });

  it("forgets the oldest games past the limit, not the one being played", () => {
    for (let i = 0; i < 35; i += 1) saveOdds(`game-${i}`, 2, even);
    saveOdds("game-1", 4, even);
    const stored = JSON.parse(localStorage.getItem("catanatron.winOdds") ?? "{}");
    expect(Object.keys(stored)).toHaveLength(30);
    expect(Object.keys(stored)).toContain("game-34");
    expect(Object.keys(stored)).not.toContain("a");
    expect(Object.keys(stored["game-1"])).toEqual(["4"]);
  });
});
