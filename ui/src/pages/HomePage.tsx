import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import cn from "classnames";
import { Alert, Button, Checkbox, MenuItem, Select, Slider, Tooltip } from "@mui/material";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import { useSnackbar } from "notistack";

import { CLASSIC_UI_URL } from "../configuration";

import {
  createGame,
  getJaxAgents,
  type JaxAgent,
  type MapTemplate,
  type PlayerArchetype,
} from "../utils/apiClient";
import { listGames, recordAgainst, saveGame } from "../utils/history";
import "./HomePage.scss";

type Opponent = {
  key: string;
  label: string;
  group: string;
  level: string;
  note: string;
};

const CLASSIC_BOTS: Opponent[] = [
  { key: "RANDOM", label: "Random", group: "Classic bots", level: "Easy", note: "Plays any legal move" },
  { key: "WEIGHTED_RANDOM", label: "Weighted Random", group: "Classic bots", level: "Easy", note: "Random, but prefers building" },
  { key: "CATANATRON", label: "Catanatron", group: "Classic bots", level: "Medium", note: "Searches a couple of moves ahead" },
];

const LEVEL_DOTS: Record<string, number> = { Easy: 1, Medium: 2, Hard: 3 };
const MAP_TEMPLATES: { value: MapTemplate; description: string }[] = [
  { value: "BASE", description: "Classic 19 hexes" },
  { value: "MINI", description: "Compact 7 hexes" },
  { value: "TOURNAMENT", description: "Balanced for pros" },
];
const PLAYER_COLORS = ["RED", "BLUE", "ORANGE", "WHITE"] as const;

const toOpponent = (agent: JaxAgent): Opponent => ({
  key: agent.key,
  label: agent.label,
  group: agent.group ?? "Trained agents",
  level: agent.level ?? "Medium",
  note:
    agent.win_rate !== undefined
      ? `Wins ${Math.round(agent.win_rate)}% against a solid hand-written bot`
      : "Trained agent",
});

function displayName(key: PlayerArchetype, opponents: Opponent[]) {
  if (key === "HUMAN") return "You";
  return opponents.find((opponent) => opponent.key === key)?.label ?? key;
}

function timeAgo(timestamp: number) {
  const minutes = Math.round((Date.now() - timestamp) / 60000);
  if (minutes < 60) return `${Math.max(minutes, 1)} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  return `${Math.round(hours / 24)} d ago`;
}

export default function HomePage() {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);
  const [serverDown, setServerDown] = useState(false);
  const [agents, setAgents] = useState<JaxAgent[]>([]);
  const [mapTemplate, setMapTemplate] = useState<MapTemplate>("BASE");
  const [vpsToWin, setVpsToWin] = useState(10);
  const [discardLimit, setDiscardLimit] = useState(7);
  const [friendlyRobber, setFriendlyRobber] = useState(false);
  const [players, setPlayers] = useState<PlayerArchetype[]>(["HUMAN", "CATANATRON"]);
  const choiceMade = useRef(false);

  const games = useMemo(() => listGames(), []);
  const opponents = useMemo(() => [...agents.map(toOpponent), ...CLASSIC_BOTS], [agents]);
  const groups = useMemo(() => [...new Set(opponents.map((opponent) => opponent.group))], [opponents]);
  const unfinished = games.find((game) => !game.result && game.players.includes("HUMAN"));
  const recent = games.filter((game) => game.result).slice(0, 5);

  const humanCount = players.filter((player) => player === "HUMAN").length;
  const hasJaxAgent = players.some((player) => player.startsWith("JAX:"));
  const jaxSetupInvalid = hasJaxAgent && (players.length !== 2 || mapTemplate !== "BASE");
  const isDuel = players.length === 2 && players[0] === "HUMAN";

  const useTrainedSettings = () => {
    // The settings these agents trained on, which their measured strength assumes.
    setMapTemplate("BASE");
    setVpsToWin(10);
    setDiscardLimit(7);
    setFriendlyRobber(false);
  };

  const fetchAgents = useCallback(async () => {
    const result = await getJaxAgents();
    setServerDown(result === null);
    if (result) setAgents(result);
    return result;
  }, []);

  useEffect(() => {
    fetchAgents().then((result) => {
      // Start on a fair fight, unless the player already picked someone.
      const preferred = result?.find((agent) => agent.level === "Medium" && agent.group === "PSRO");
      if (preferred && !choiceMade.current) {
        setPlayers(["HUMAN", preferred.key]);
        useTrainedSettings();
      }
    });
  }, [fetchAgents]);

  const pickOpponent = (key: string) => {
    choiceMade.current = true;
    setPlayers(["HUMAN", key]);
    if (key.startsWith("JAX:")) useTrainedSettings();
  };

  const handleSeatChange = (index: number, value: PlayerArchetype) => {
    if (value === "HUMAN" && players[index] !== "HUMAN" && humanCount >= 1) return;
    choiceMade.current = true;
    if (value.startsWith("JAX:")) useTrainedSettings();
    setPlayers((current) => current.map((player, i) => (i === index ? value : player)));
  };

  const start = async () => {
    if (humanCount > 1 || jaxSetupInvalid) return;
    setLoading(true);
    try {
      const settings = { mapTemplate, vpsToWin, discardLimit, friendlyRobber };
      const id = await createGame({ players, ...settings });
      saveGame({
        id,
        createdAt: Date.now(),
        players,
        labels: players.map((player) => displayName(player, opponents)),
        settings,
      });
      navigate(`/games/${id}`);
    } catch (error) {
      console.error(error);
      enqueueSnackbar("Couldn't start the game. Is the game server running?", {
        variant: "error",
        autoHideDuration: 5000,
      });
      setLoading(false);
    }
  };

  const selected = isDuel ? players[1] : null;
  const selectedOpponent = opponents.find((opponent) => opponent.key === selected);
  const playLabel = loading
    ? "Setting up the board"
    : selectedOpponent
      ? `Play ${selectedOpponent.label}`
      : "Start game";

  return (
    <div className="home-page">
      <div className="home-inner">
        <header className="home-hero">
          <span className="home-mark" aria-hidden />
          <div>
            <h1>Catanatron</h1>
            <p>Play Catan against agents trained by self-play.</p>
          </div>
          {CLASSIC_UI_URL && (
            <nav className="ui-switch" aria-label="Choose interface">
              <span aria-current="page">New</span>
              <a href={CLASSIC_UI_URL}>Classic</a>
            </nav>
          )}
        </header>

        {serverDown && (
          <Alert
            severity="warning"
            className="home-alert"
            action={
              <Button color="inherit" size="small" onClick={fetchAgents}>
                Retry
              </Button>
            }
          >
            Can't reach the game server, so games can't start yet. Is it running on port 5001?
          </Alert>
        )}

        {unfinished && (
          <Link to={`/games/${unfinished.id}`} className="continue-card">
            <span>
              <strong>Continue your game</strong>
              <small>
                vs {unfinished.labels.filter((label) => label !== "You").join(", ")} ·{" "}
                {timeAgo(unfinished.createdAt)}
              </small>
            </span>
            <em>Resume</em>
          </Link>
        )}

        <h2>Choose your opponent</h2>
        <div className="opponent-groups">
          {groups.map((group) => (
            <section key={group} className="opponent-group" aria-label={group}>
              <h3>{group}</h3>
              <div className="opponent-grid">
                {opponents
                  .filter((opponent) => opponent.group === group)
                  .map((opponent) => {
                    const { wins, losses } = recordAgainst(games, opponent.key);
                    return (
                      <button
                        key={opponent.key}
                        className={cn("opponent-card", { selected: selected === opponent.key })}
                        aria-pressed={selected === opponent.key}
                        onClick={() => pickOpponent(opponent.key)}
                      >
                        <span className="opponent-top">
                          <strong>{group === "Classic bots" ? opponent.label : opponent.level}</strong>
                          <span className="dots" aria-label={`Difficulty: ${opponent.level}`}>
                            {[1, 2, 3].map((dot) => (
                              <i key={dot} className={cn({ on: dot <= (LEVEL_DOTS[opponent.level] ?? 2) })} />
                            ))}
                          </span>
                        </span>
                        <small>{opponent.note}</small>
                        {wins + losses > 0 && (
                          <span className="your-record">
                            You: {wins} won, {losses} lost
                          </span>
                        )}
                      </button>
                    );
                  })}
              </div>
            </section>
          ))}
        </div>

        <details className="advanced">
          <summary>Game settings and custom tables</summary>
          <div className="advanced-body">
            <div className="setting">
              <span className="setting-label">Map</span>
              <div className="map-choices">
                {MAP_TEMPLATES.map(({ value, description }) => (
                  <button
                    key={value}
                    className={cn("map-choice", { selected: mapTemplate === value })}
                    aria-pressed={mapTemplate === value}
                    onClick={() => setMapTemplate(value)}
                  >
                    <strong>{value[0] + value.slice(1).toLowerCase()}</strong>
                    <small>{description}</small>
                  </button>
                ))}
              </div>
            </div>
            <div className="setting-row">
              <label className="setting">
                <span className="setting-label">
                  Points to win <b>{vpsToWin}</b>
                </span>
                <Slider
                  aria-label="Points to win"
                  value={vpsToWin}
                  min={3}
                  max={20}
                  onChange={(_, value) => setVpsToWin(value as number)}
                />
              </label>
              <label className="setting">
                <span className="setting-label">
                  Card discard limit <b>{discardLimit}</b>
                </span>
                <Slider
                  aria-label="Card discard limit"
                  value={discardLimit}
                  min={5}
                  max={20}
                  onChange={(_, value) => setDiscardLimit(value as number)}
                />
              </label>
              <label className="setting robber-setting">
                <Checkbox
                  checked={friendlyRobber}
                  onChange={(event) => setFriendlyRobber(event.target.checked)}
                />
                <span className="setting-label">Friendly robber</span>
                <Tooltip title="The robber can't block players who have only 2 victory points." arrow enterTouchDelay={0}>
                  <HelpOutlineRoundedIcon fontSize="inherit" aria-label="About the friendly robber" />
                </Tooltip>
              </label>
            </div>

            {hasJaxAgent && !jaxSetupInvalid && (
              <Alert severity="info" className="home-alert">
                PSRO and PPO agents trained on 10 points, discard limit 7, and friendly robber off. Other
                settings work but are untested.
              </Alert>
            )}

            <div className="setting">
              <span className="setting-label">Seats ({players.length} of 4)</span>
              {players.map((player, index) => (
                <div className="seat" key={`${player}-${index}`}>
                  <span className={cn("seat-color", PLAYER_COLORS[index].toLowerCase())}>{PLAYER_COLORS[index]}</span>
                  <Select
                    size="small"
                    fullWidth
                    value={opponents.some((opponent) => opponent.key === player) || player === "HUMAN" ? player : ""}
                    inputProps={{ "aria-label": `Seat ${index + 1}` }}
                    onChange={(event) => handleSeatChange(index, event.target.value as PlayerArchetype)}
                  >
                    <MenuItem value="HUMAN" disabled={humanCount >= 1 && player !== "HUMAN"}>
                      You
                    </MenuItem>
                    {opponents.map((opponent) => (
                      <MenuItem key={opponent.key} value={opponent.key}>
                        {opponent.label}
                      </MenuItem>
                    ))}
                  </Select>
                  <Button
                    color="inherit"
                    disabled={players.length <= 2}
                    onClick={() => setPlayers((current) => current.filter((_, i) => i !== index))}
                  >
                    Remove
                  </Button>
                </div>
              ))}
              <Button
                className="add-seat"
                variant="outlined"
                color="inherit"
                disabled={players.length >= 4}
                onClick={() => setPlayers((current) => [...current, "WEIGHTED_RANDOM"])}
              >
                Add a seat
              </Button>
              <small className="seat-hint">Set no seat to You to watch the bots play each other.</small>
            </div>
          </div>
        </details>

        {jaxSetupInvalid && (
          <Alert severity="error" className="home-alert">
            PSRO and PPO agents only play 2-player games on the Base map.
          </Alert>
        )}

        <div className="play-bar">
          <Button
            variant="contained"
            className="play-button"
            disabled={loading || jaxSetupInvalid || serverDown}
            onClick={start}
          >
            {playLabel}
          </Button>
        </div>

        {recent.length > 0 && (
          <section className="recent-games" aria-label="Recent games">
            <h2>Recent games</h2>
            <ul>
              {recent.map((game) => (
                <li key={game.id}>
                  <span className={cn("result", { won: game.result?.humanWon, lost: game.result?.humanWon === false })}>
                    {game.result?.humanWon === null ? "Watched" : game.result?.humanWon ? "Won" : "Lost"}
                  </span>
                  <span className="recent-name">vs {game.labels.filter((label) => label !== "You").join(", ")}</span>
                  <small>
                    {game.result?.turns} turns · {timeAgo(game.createdAt)}
                  </small>
                  <Link to={`/replays/${game.id}`}>Replay</Link>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
