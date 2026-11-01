import type { Color, MapTemplate } from "./api.types";

/** Games played in this browser. The server has no player names or game list. */
export type GameRecord = {
  id: string;
  createdAt: number;
  players: string[];
  labels: string[];
  settings: {
    mapTemplate: MapTemplate;
    vpsToWin: number;
    discardLimit: number;
    friendlyRobber: boolean;
  };
  result?: { winner: Color; humanWon: boolean | null; turns: number };
};

const KEY = "catanatron.games";
const MAX_GAMES = 200;
// Colors are handed out in this order when a game is created.
const SEAT_COLORS: Color[] = ["RED", "BLUE", "ORANGE", "WHITE"];

function read(): GameRecord[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function write(games: GameRecord[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(games.slice(0, MAX_GAMES)));
  } catch {
    // Storage can be full or disabled. History is a convenience, so play on without it.
  }
}

export const listGames = () => read();

export const getGame = (id?: string) => read().find((game) => game.id === id);

export function saveGame(game: GameRecord) {
  write([game, ...read().filter((other) => other.id !== game.id)]);
}

export function saveResult(id: string, result: NonNullable<GameRecord["result"]>) {
  write(read().map((game) => (game.id === id && !game.result ? { ...game, result } : game)));
}

export function playerLabel(game: GameRecord | undefined, color: Color) {
  const seat = SEAT_COLORS.indexOf(color);
  return game?.labels[seat] ?? color[0] + color.slice(1).toLowerCase();
}

/** Wins and losses for a human against one opponent, in 2-player games. */
export function recordAgainst(games: GameRecord[], opponent: string) {
  const played = games.filter(
    (game) =>
      game.result &&
      game.result.humanWon !== null &&
      game.players.length === 2 &&
      game.players[0] === "HUMAN" &&
      game.players[1] === opponent,
  );
  const wins = played.filter((game) => game.result?.humanWon).length;
  return { wins, losses: played.length - wins };
}

export function loadSetting<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(`catanatron.${key}`) as T) ?? fallback;
  } catch {
    return fallback;
  }
}

export function saveSetting(key: string, value: string) {
  try {
    localStorage.setItem(`catanatron.${key}`, value);
  } catch {
    // Same as above: settings just won't persist.
  }
}
