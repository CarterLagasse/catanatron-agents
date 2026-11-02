import { useLayoutEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import cn from "classnames";

import type { Color, GameState } from "../utils/api.types";
import { oddsSeries, turnEndIndexes, useWinOdds, type OddsPoint } from "../utils/winOdds";
import "./WinOddsChart.scss";

const MARGIN = { left: 36, right: 44, top: 10, bottom: 22 };
const LABEL_GAP = 13;
const STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500];

function Readout({ point, colors, name, x, width }: { point: OddsPoint; colors: Color[]; name: (color: Color) => string; x: number; width: number }) {
  return (
    <div className={cn("odds-readout", { flip: x > width / 2 })} style={{ left: x }}>
      <strong>{point.final ? `Result, turn ${point.turn}` : `After turn ${point.turn}`}</strong>
      {[...colors]
        .sort((a, b) => point.odds[b] - point.odds[a])
        .map((color) => (
          <span key={color} className={color.toLowerCase()}>
            <i aria-hidden />
            <b>{Math.round(point.odds[color])}%</b>
            {name(color)}
          </span>
        ))}
    </div>
  );
}

export default function WinOddsChart({
  gameState: gs,
  name,
  live = false,
}: {
  gameState: GameState;
  name: (color: Color) => string;
  live?: boolean;
}) {
  const { gameId } = useParams();
  const points = oddsSeries(gs, useWinOdds(gameId));
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(280);
  const [hover, setHover] = useState<number | null>(null);

  const hasPoints = points.length > 0;
  useLayoutEffect(() => {
    if (!box.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(box.current);
    return () => observer.disconnect();
  }, [hasPoints]);

  const waiting = turnEndIndexes(gs).length - points.filter((point) => !point.final).length;
  const caption = `Chance of winning after each turn.${live && waiting > 0 ? ` Working out ${waiting} more.` : ""}`;
  const heading = (
    <figcaption>
      <strong>Win odds by turn</strong>
      <span>{caption}</span>
    </figcaption>
  );
  if (!hasPoints) {
    return (
      <figure className="win-odds-chart">
        {heading}
        <p className="empty-note">
          {live ? "The graph starts once the first turn ends." : "No win odds were saved for this game in this browser."}
        </p>
      </figure>
    );
  }

  const height = Math.round(Math.min(260, Math.max(150, width * 0.5)));
  const plotWidth = width - MARGIN.left - MARGIN.right;
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  const lastTurn = Math.max(2, points[points.length - 1].turn);
  const xAt = (turn: number) => MARGIN.left + ((turn - 1) / (lastTurn - 1)) * plotWidth;
  const yAt = (percent: number) => MARGIN.top + (1 - percent / 100) * plotHeight;

  const maxTicks = Math.max(2, Math.floor(plotWidth / 34));
  const step = STEPS.find((candidate) => lastTurn / candidate <= maxTicks) ?? 1000;
  const turnTicks = Array.from({ length: Math.floor(lastTurn / step) }, (_, i) => (i + 1) * step);

  const newest = points[points.length - 1];
  // Close end labels aren't pushed apart, which would detach them. The readout has those values.
  const endLabels = gs.colors.filter((color) =>
    gs.colors.every(
      (other) => other === color || Math.abs(yAt(newest.odds[color]) - yAt(newest.odds[other])) >= LABEL_GAP,
    ),
  );

  const pick = (clientX: number, element: Element) => {
    const x = clientX - element.getBoundingClientRect().left;
    let nearest = 0;
    points.forEach((point, i) => {
      if (Math.abs(xAt(point.turn) - x) < Math.abs(xAt(points[nearest].turn) - x)) nearest = i;
    });
    setHover(nearest);
  };
  const shown = hover !== null ? points[hover] : null;

  return (
    <figure className="win-odds-chart">
      {heading}
      <ul className="odds-legend">
        {gs.colors.map((color) => (
          <li key={color} className={color.toLowerCase()}>
            <i aria-hidden />
            {name(color)}
          </li>
        ))}
      </ul>
      <div className="odds-plot" ref={box}>
        <svg
          width={width}
          height={height}
          role="img"
          tabIndex={0}
          aria-label={`Win odds over ${newest.turn} turns. Latest: ${gs.colors
            .map((color) => `${name(color)} ${Math.round(newest.odds[color])}%`)
            .join(", ")}. Use the arrow keys to read each turn.`}
          onPointerMove={(event) => pick(event.clientX, event.currentTarget)}
          onPointerDown={(event) => pick(event.clientX, event.currentTarget)}
          onPointerLeave={() => setHover(null)}
          onFocus={() => setHover(points.length - 1)}
          onBlur={() => setHover(null)}
          onKeyDown={(event) => {
            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
            event.preventDefault();
            const move = event.key === "ArrowLeft" ? -1 : 1;
            setHover((index) => Math.min(points.length - 1, Math.max(0, (index ?? points.length - 1) + move)));
          }}
        >
          {[0, 50, 100].map((percent) => (
            <g key={percent}>
              <line className="grid" x1={MARGIN.left} x2={width - MARGIN.right} y1={yAt(percent)} y2={yAt(percent)} />
              <text className="axis" x={MARGIN.left - 6} y={yAt(percent) + 3.5} textAnchor="end">
                {percent}%
              </text>
            </g>
          ))}
          {turnTicks.map((turn) => (
            <text key={turn} className="axis" x={xAt(turn)} y={height - 6} textAnchor="middle">
              {turn}
            </text>
          ))}
          {shown && <line className="crosshair" x1={xAt(shown.turn)} x2={xAt(shown.turn)} y1={MARGIN.top} y2={MARGIN.top + plotHeight} />}
          {gs.colors.map((color) => (
            <g key={color} className={color.toLowerCase()}>
              <polyline className="series" points={points.map((point) => `${xAt(point.turn)},${yAt(point.odds[color])}`).join(" ")} />
              <circle className="dot" cx={xAt((shown ?? newest).turn)} cy={yAt((shown ?? newest).odds[color])} r={4} />
            </g>
          ))}
          {!shown &&
            endLabels.map((color) => (
              <text key={color} className="end-label" x={xAt(newest.turn) + 8} y={yAt(newest.odds[color]) + 3.5}>
                {Math.round(newest.odds[color])}%
              </text>
            ))}
        </svg>
        {shown && <Readout point={shown} colors={gs.colors} name={name} x={xAt(shown.turn)} width={width} />}
      </div>
      <details className="odds-table">
        <summary>Show as a table</summary>
        <div>
          <table>
            <thead>
              <tr>
                <th scope="col">Turn</th>
                {gs.colors.map((color) => (
                  <th key={color} scope="col">
                    {name(color)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {points.map((point) => (
                <tr key={point.turn}>
                  <th scope="row">{point.final ? `${point.turn} (result)` : point.turn}</th>
                  {gs.colors.map((color) => (
                    <td key={color}>{Math.round(point.odds[color])}%</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
