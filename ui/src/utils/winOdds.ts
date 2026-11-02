import { useEffect, useState, useSyncExternalStore } from "react";

import type { Color, GameState } from "./api.types";
import { getMctsAnalysis, type MCTSProbabilities } from "./apiClient";

/** Win odds by state index. Kept in this browser, like the list of games. */
export type SavedOdds = Record<number, MCTSProbabilities>;

export type OddsPoint = { turn: number; odds: MCTSProbabilities; final?: boolean };

const KEY = "catanatron.winOdds";
const MAX_GAMES = 30;
const RETRY_MS = 30000;
const NONE: SavedOdds = {};

function read(): Record<string, SavedOdds> {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

let games = read();
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function saveOdds(gameId: string, stateIndex: number, odds: MCTSProbabilities) {
  const { [gameId]: saved, ...others } = games;
  const kept = Object.entries(others).slice(-(MAX_GAMES - 1));
  games = { ...Object.fromEntries(kept), [gameId]: { ...saved, [stateIndex]: odds } };
  try {
    localStorage.setItem(KEY, JSON.stringify(games));
  } catch {
    // Storage can be full or disabled. The graph then lasts until the page reloads.
  }
  listeners.forEach((listener) => listener());
}

const pending = new Map<string, ReturnType<typeof getMctsAnalysis>>();

/** Asks for the odds at a move and saves them. A request already running for that move is shared. */
export function requestOdds(gameId: string, stateIndex: number) {
  const key = `${gameId}/${stateIndex}`;
  const running = pending.get(key);
  if (running) return running;
  const request = getMctsAnalysis(gameId, stateIndex)
    .then((result) => {
      if (result.success) saveOdds(gameId, stateIndex, result.probabilities);
      return result;
    })
    .finally(() => pending.delete(key));
  pending.set(key, request);
  return request;
}

/** The move right after each END_TURN. These states are saved, so any of them can be worked out later. */
export function turnEndIndexes(gs: GameState) {
  return gs.action_records.flatMap((record, i) => (record[0][1] === "END_TURN" ? [i + 1] : []));
}

/** The graph's points: one for each finished turn with known odds, then the result once there is one. */
export function oddsSeries(gs: GameState, saved: SavedOdds): OddsPoint[] {
  const ends = turnEndIndexes(gs);
  const points = ends.flatMap((index, i) => (saved[index] ? [{ turn: i + 1, odds: saved[index] }] : []));
  if (!gs.winning_color) return points;
  const result = Object.fromEntries(
    gs.colors.map((color: Color) => [color, color === gs.winning_color ? 100 : 0]),
  ) as MCTSProbabilities;
  return [...points, { turn: ends.length + 1, odds: result, final: true }];
}

export function useWinOdds(gameId?: string) {
  return useSyncExternalStore(subscribe, () => (gameId && games[gameId]) || NONE);
}

/**
 * Works out the odds for every finished turn, newest first and one at a time, so a slow
 * server isn't asked for several at once. Stays out of the way while a request fails.
 */
export function useWinOddsTracker(gameId: string | undefined, gs: GameState | null, enabled: boolean) {
  const saved = useWinOdds(gameId);
  const [busy, setBusy] = useState(false);
  const next = enabled && gs ? turnEndIndexes(gs).filter((index) => !saved[index]).pop() : undefined;
  useEffect(() => {
    if (busy || next === undefined || !gameId) return;
    setBusy(true);
    (async () => {
      let worked = false;
      try {
        worked = (await requestOdds(gameId, next)).success;
      } catch {
        // Logged by the API client. Tried again below.
      }
      if (!worked) await new Promise((resolve) => setTimeout(resolve, RETRY_MS));
      setBusy(false);
    })();
  }, [busy, gameId, next]);
}
