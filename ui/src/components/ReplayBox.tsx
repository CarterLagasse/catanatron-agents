import { IconButton, Slider, Tooltip } from "@mui/material";
import NavigateBeforeIcon from "@mui/icons-material/NavigateBefore";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";

import "./ReplayBox.scss";

type ReplayBoxProps = {
  stateIndex: number;
  latestStateIndex: number;
  playing: boolean;
  onTogglePlay: () => void;
  onPrevMove: () => void;
  onNextMove: () => void;
  onSeekMove: (value: number) => void;
};

export default function ReplayBox({
  stateIndex,
  latestStateIndex,
  playing,
  onTogglePlay,
  onPrevMove,
  onNextMove,
  onSeekMove,
}: ReplayBoxProps) {
  return (
    <div className="replay-box">
      <Tooltip title="Previous move" arrow>
        <span>
          <IconButton onClick={onPrevMove} disabled={stateIndex === 0} aria-label="Previous move">
            <NavigateBeforeIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title={playing ? "Pause" : "Play"} arrow>
        <IconButton className="play" onClick={onTogglePlay} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <PauseIcon /> : <PlayArrowIcon />}
        </IconButton>
      </Tooltip>
      <Tooltip title="Next move" arrow>
        <span>
          <IconButton onClick={onNextMove} disabled={stateIndex === latestStateIndex} aria-label="Next move">
            <NavigateNextIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Slider
        className="move-slider"
        aria-label="Move"
        min={0}
        max={latestStateIndex}
        step={1}
        value={stateIndex}
        onChange={(_, value) => onSeekMove(value as number)}
      />
      <span className="move-count">
        {stateIndex} / {latestStateIndex}
      </span>
    </div>
  );
}
