import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import cn from "classnames";
import {
  Dialog,
  IconButton,
  ListItemIcon,
  Menu,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import LeaderboardIcon from "@mui/icons-material/Leaderboard";
import SettingsIcon from "@mui/icons-material/Settings";
import ReplayIcon from "@mui/icons-material/Replay";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import CheckIcon from "@mui/icons-material/Check";

import type { Status } from "../utils/gameLogic";
import "./TopBar.scss";

export type BotSpeed = "slow" | "normal" | "fast";

const SHORTCUTS: [string, string][] = [
  ["R", "Roll the dice"],
  ["E", "End your turn"],
  ["1", "Build a road"],
  ["2", "Build a settlement"],
  ["3", "Build a city"],
  ["4", "Buy a development card"],
  ["T", "Trade with the bank"],
  ["Esc", "Cancel building"],
];

function HelpDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth slotProps={{ paper: { className: "help-dialog" } }}>
      <h2>How to play</h2>
      <p>
        First to the winning points wins. Settlements are worth 1 point and cities 2. Longest Road (5+
        roads) and Largest Army (3+ knights) are worth 2 each.
      </p>
      <p>
        Each turn: roll, collect cards from tiles next to your buildings, then build and trade. Rolling a 7
        moves the robber, and anyone over the discard limit loses half their cards.
      </p>
      <p>Drag to move the board, and scroll or use the buttons to zoom. The numbers on glowing spots count the rolls out of 36 that pay them.</p>
      <h3>Keyboard</h3>
      <dl>
        {SHORTCUTS.map(([key, meaning]) => (
          <div key={key}>
            <dt>
              <kbd>{key}</kbd>
            </dt>
            <dd>{meaning}</dd>
          </div>
        ))}
      </dl>
    </Dialog>
  );
}

export default function TopBar({
  status,
  matchup,
  speed,
  onSpeedChange,
  revealHidden,
  onRevealChange,
  replayHref,
  classicHref,
  onOpenPanel,
  children,
}: {
  status: Status;
  matchup?: string;
  speed?: BotSpeed;
  onSpeedChange?: (speed: BotSpeed) => void;
  revealHidden?: boolean;
  onRevealChange?: (reveal: boolean) => void;
  replayHref?: string;
  classicHref?: string;
  onOpenPanel: () => void;
  children?: ReactNode;
}) {
  const [helpOpen, setHelpOpen] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  return (
    <header className="top-bar">
      <Link to="/" className="brand" aria-label="Catanatron home">
        <span className="brand-mark" aria-hidden />
        <span className="brand-name">Catanatron</span>
      </Link>
      {matchup && <span className="matchup">{matchup}</span>}
      <div className={cn("status-pill", status.tone)} role="status" aria-live="polite">
        <span className="status-dot" aria-hidden />
        <strong>{status.title}</strong>
        <small>{status.detail}</small>
      </div>
      <div className="top-actions">
        {children}
        <Tooltip title="Scores, win odds and log" arrow>
          <IconButton className="panel-toggle" onClick={onOpenPanel} aria-label="Open scores, win odds and log">
            <LeaderboardIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="How to play" arrow>
          <IconButton onClick={() => setHelpOpen(true)} aria-label="How to play">
            <HelpOutlineIcon />
          </IconButton>
        </Tooltip>
        {onSpeedChange && (
          <Tooltip title="Game options" arrow>
            <IconButton onClick={(event) => setMenuAnchor(event.currentTarget)} aria-label="Game options">
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        )}
        <Link to="/" className="new-game">
          <AddIcon fontSize="small" />
          <span>New game</span>
        </Link>
      </div>
      <Menu anchorEl={menuAnchor} open={!!menuAnchor} onClose={() => setMenuAnchor(null)}>
        <div className="speed-row">
          <span>Bot speed</span>
          <ToggleButtonGroup
            size="small"
            exclusive
            value={speed}
            onChange={(_, value) => value && onSpeedChange?.(value)}
          >
            <ToggleButton value="slow">Slow</ToggleButton>
            <ToggleButton value="normal">Normal</ToggleButton>
            <ToggleButton value="fast">Fast</ToggleButton>
          </ToggleButtonGroup>
        </div>
        <MenuItem onClick={() => onRevealChange?.(!revealHidden)}>
          <ListItemIcon>{revealHidden && <CheckIcon fontSize="small" />}</ListItemIcon>
          Show opponents' cards
        </MenuItem>
        {replayHref && (
          <MenuItem component={Link} to={replayHref}>
            <ListItemIcon>
              <ReplayIcon fontSize="small" />
            </ListItemIcon>
            Replay this game
          </MenuItem>
        )}
        {classicHref && (
          <MenuItem component="a" href={classicHref}>
            <ListItemIcon>
              <SwapHorizIcon fontSize="small" />
            </ListItemIcon>
            Open in the classic UI
          </MenuItem>
        )}
      </Menu>
      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </header>
  );
}
