"""Draws the figures in IMPROVING.md from the training logs in runs/.

    python -m catanatron.jax_rl.plot_training --runs runs \
        --out catanatron/catanatron/jax_rl/figures

Each figure is written twice, as NAME-light.svg and NAME-dark.svg, so the markdown can
pick one per page theme. Only the standard library is used, so this runs anywhere the
logs are.
"""

import argparse
import json
import math
from pathlib import Path

STEPS_PER_ITERATION = 256 * 128  # num_envs x rollout_ticks in both PPO runs
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834"],
        "negative": "#e34948",
        "middle": "#f0efec",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926"],
        "negative": "#e66767",
        "middle": "#383835",
    },
}


# ===== Colour


def _to_oklab(hex_color):
    def linear(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(int(hex_color[i : i + 2], 16)) for i in (1, 3, 5))
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _from_oklab(lab):
    lightness, a, b = lab
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )

    def encoded(c):
        c = min(max(c, 0.0), 1.0)
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(c * 255)

    return "#" + "".join(f"{encoded(c):02x}" for c in rgb)


def diverging(theme, value, limit):
    """Gray at 0, blue for positive, red for negative, mixed in OKLab."""
    t = max(-1.0, min(1.0, value / limit))
    pole = theme["series"][0] if t >= 0 else theme["negative"]
    a, b = _to_oklab(theme["middle"]), _to_oklab(pole)
    return _from_oklab(tuple(x + abs(t) * (y - x) for x, y in zip(a, b)))


# ===== SVG


def _escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width, height, theme, title, description):
        self.width, self.height, self.theme = width, height, theme
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" font-family="{FONT}">',
            f"<title>{_escape(title)}</title><desc>{_escape(description)}</desc>",
            f'<rect width="{width}" height="{height}" rx="8" fill="{theme["surface"]}"/>',
        ]

    def add(self, element):
        self.parts.append(element)

    def line(self, x1, y1, x2, y2, color, width=1.0, opacity=1.0):
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" '
            'stroke-linecap="round"/>'
        )

    def path(self, points, color, width=2.0):
        if len(points) < 2:
            return
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.add(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def dot(self, x, y, color, radius=4.0):
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            f'stroke="{self.theme["surface"]}" stroke-width="2"/>'
        )

    def rect(self, x, y, width, height, color):
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'fill="{color}"/>'
        )

    def text(self, x, y, content, color, size=12, anchor="start", weight=400):
        numeric = ' font-variant-numeric="tabular-nums"' if anchor != "start" else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{numeric}>'
            f"{_escape(content)}</text>"
        )

    def save(self, path):
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def nice_ticks(low, high, count=5):
    raw = (high - low) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _number(value):
    return f"{value:g}" if abs(value) < 1000 else f"{value:,.0f}"


class Plot:
    """One panel: linear axes, solid hairline grid, axis titles."""

    def __init__(
        self,
        svg,
        box,
        x_range,
        y_range,
        x_title="",
        y_title="",
        y_format=None,
        x_ticks=None,
    ):
        self.svg = svg
        self.left, self.top, self.right, self.bottom = box
        self.x_range, self.y_range = x_range, y_range
        theme = svg.theme
        y_format = y_format or _number
        for tick in nice_ticks(*y_range, 4):
            y = self.y(tick)
            svg.line(self.left, y, self.right, y, theme["grid"])
            svg.text(self.left - 6, y + 4, y_format(tick), theme["muted"], 11, "end")
        svg.line(self.left, self.bottom, self.right, self.bottom, theme["axis"])
        for tick in x_ticks if x_ticks is not None else nice_ticks(*x_range, 5):
            svg.text(
                self.x(tick),
                self.bottom + 16,
                _number(tick),
                theme["muted"],
                11,
                "middle",
            )
        if x_title:
            svg.text(
                (self.left + self.right) / 2,
                self.bottom + 34,
                x_title,
                theme["ink2"],
                12,
                "middle",
            )
        if y_title:
            svg.text(self.left - 44, self.top - 12, y_title, theme["ink2"], 12)

    def x(self, value):
        low, high = self.x_range
        return self.left + (value - low) / (high - low) * (self.right - self.left)

    def y(self, value):
        low, high = self.y_range
        return self.bottom - (value - low) / (high - low) * (self.bottom - self.top)


def legend(svg, x, y, entries):
    """Line keys with names, left to right. Entries are (color, name, kind)."""
    theme = svg.theme
    for color, name, kind in entries:
        if kind == "dot":
            svg.dot(x + 7, y - 4, color)
        else:
            svg.line(x, y - 4, x + 16, y - 4, color, 2)
        svg.text(x + 22, y, name, theme["ink2"], 12)
        x += 30 + 6.6 * len(name)


# ===== Data


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def moving_average(values, window):
    half = window // 2
    result = []
    for i in range(len(values)):
        chunk = values[max(0, i - half) : i + half + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def snapshot_strength(runs):
    """Engine win rate vs F of GPU PPO snapshots, from two tournament runs."""
    points = []
    for folder in ("levels/calibration", "gpu/selection"):
        for name, row in json.loads(
            (runs / folder / "summary.json").read_text()
        ).items():
            label = name.split(" vs ")[0]
            if not label.startswith("it_"):
                continue
            points.append(
                {
                    "iteration": int(label[3:]),
                    "games": row["games"],
                    "rate": 100 * row["win_rate"],
                    "low": 100 * row["ci_low"],
                    "high": 100 * row["ci_high"],
                }
            )
    return sorted(points, key=lambda p: p["iteration"])


# ===== Figures


def figure_strength(runs, theme, out):
    points = snapshot_strength(runs)
    retest = json.loads((runs / "eval/summary.json").read_text())["PPO-GPU vs F"]
    svg = Svg(
        760,
        380,
        theme,
        "GPU PPO strength against F over training",
        "Engine win rate against ValueFunctionPlayer for snapshots of the 200M-step GPU "
        "PPO run, with 95% intervals, rising from 0% at 1.6M steps to 75% at 200M.",
    )
    plot = Plot(
        svg,
        (64, 64, 720, 320),
        (0, 205),
        (0, 100),
        "Training steps (millions)",
        "Win rate vs F (%)",
    )
    blue, orange = theme["series"]
    y50 = plot.y(50)
    svg.line(plot.left, y50, plot.right, y50, theme["axis"], 1.5)
    svg.text(plot.right, y50 - 6, "As strong as F", theme["muted"], 11, "end")
    xy = [(plot.x(p["iteration"] * STEPS_PER_ITERATION / 1e6), p) for p in points]
    for x, p in xy:
        svg.line(x, plot.y(p["low"]), x, plot.y(p["high"]), blue, 1.5, 0.55)
    svg.path([(x, plot.y(p["rate"])) for x, p in xy], blue)
    for x, p in xy:
        svg.dot(x, plot.y(p["rate"]), blue)
    labels = {200: "PPO Easy", 400: "PPO Medium", 6103: "PPO Hard (selected)"}
    for x, p in xy:
        if p["iteration"] in labels:
            anchor = "end" if p["iteration"] == 6103 else "start"
            dx = -10 if anchor == "end" else 10
            svg.text(
                x + dx,
                plot.y(p["rate"]) + 4,
                labels[p["iteration"]],
                theme["ink"],
                12,
                anchor,
            )
    final_x = plot.x(6103 * STEPS_PER_ITERATION / 1e6)
    svg.line(
        final_x + 9,
        plot.y(100 * retest["ci_low"]),
        final_x + 9,
        plot.y(100 * retest["ci_high"]),
        orange,
        1.5,
    )
    svg.dot(final_x + 9, plot.y(100 * retest["win_rate"]), orange)
    legend(
        svg,
        64,
        30,
        [
            (blue, "Snapshot, 400 to 600 games", "line"),
            (orange, "Selected snapshot on 4,000 new seeds", "dot"),
        ],
    )
    svg.save(out)


def figure_simulator(runs, theme, out):
    svg = Svg(
        760,
        360,
        theme,
        "In-simulator win rate against the scripted heuristic bot",
        "Five-evaluation moving average of the PPO evaluation against the JAX heuristic "
        "bot. The CPU and GPU runs follow the same curve per step, flattening near 80%.",
    )
    plot = Plot(
        svg,
        (64, 64, 720, 300),
        (0, 205),
        (0, 100),
        "Training steps (millions)",
        "Win rate (%)",
    )
    entries = []
    for color, folder, name in zip(
        theme["series"],
        ("gpu/ppo", "ppo"),
        ("GPU run, 200M steps", "CPU run, 30M steps"),
    ):
        rows = [
            r
            for r in _read_jsonl(runs / folder / "metrics.jsonl")
            if "eval_heuristic_win" in r
        ]
        smooth = moving_average([100 * r["eval_heuristic_win"] for r in rows], 5)
        svg.path(
            [(plot.x(r["steps"] / 1e6), plot.y(v)) for r, v in zip(rows, smooth)], color
        )
        last = rows[-1]
        svg.dot(plot.x(last["steps"] / 1e6), plot.y(smooth[-1]), color)
        entries.append((color, name, "line"))
    legend(svg, 64, 30, entries)
    svg.save(out)


def figure_diagnostics(runs, theme, out):
    rows = _read_jsonl(runs / "gpu/ppo/metrics.jsonl")
    window = 50
    # Whole windows only, so the last point isn't an average of a few iterations.
    starts = range(0, len(rows) - window + 1, window)

    def rate(name):
        values = []
        for start in starts:
            chunk = rows[start : start + window]
            games = sum(r[f"{name}_games"] for r in chunk)
            values.append(100 * sum(r[f"{name}_wins"] for r in chunk) / max(games, 1))
        return values

    def smoothed(key):
        return [sum(r[key] for r in rows[s : s + window]) / window for s in starts]

    steps = [rows[s + window // 2]["steps"] / 1e6 for s in starts]
    panels = [
        ("Policy entropy (nats)", smoothed("entropy"), (0, 0.6)),
        ("Approximate KL per update", smoothed("approx_kl"), (0, 0.004)),
        ("Explained variance", smoothed("explained_variance"), (0, 1)),
        ("Value loss", smoothed("value_loss"), (0, 0.8)),
        ("Training wins vs heuristic bot (%)", rate("heuristic"), (0, 100)),
        ("Training wins vs league snapshots (%)", rate("pool"), (0, 100)),
    ]
    svg = Svg(
        760,
        640,
        theme,
        "GPU PPO training diagnostics",
        "Six panels over 200M steps, each averaged over 50 iterations: entropy, KL, "
        "explained variance, value loss, and win rates against the heuristic bot and "
        "the league.",
    )
    color = theme["series"][0]
    for i, (title, values, y_range) in enumerate(panels):
        column, row = i % 2, i // 2
        left = 70 + column * 370
        top = 40 + row * 205
        svg.text(left - 50, top - 14, title, theme["ink"], 12, weight=600)
        y_format = (lambda v: f"{v:.3f}") if "KL" in title else None
        plot = Plot(
            svg,
            (left, top, left + 290, top + 140),
            (0, 205),
            y_range,
            y_format=y_format,
        )
        if "league" in title:
            y50 = plot.y(50)
            svg.line(plot.left, y50, plot.right, y50, theme["axis"], 1.5)
        svg.path([(plot.x(s), plot.y(v)) for s, v in zip(steps, values)], color)
        svg.dot(plot.x(steps[-1]), plot.y(values[-1]), color)
    svg.text(380, 628, "Training steps (millions)", theme["ink2"], 12, "middle")
    svg.save(out)


def figure_psro_gains(runs, theme, out):
    svg = Svg(
        760,
        340,
        theme,
        "How much each PSRO response beat the mixture before it",
        "Win-rate points above 50% for each new best response against the previous "
        "Nash mixture, for the CPU and GPU PSRO runs. All stay between -3 and +3.",
    )
    plot = Plot(
        svg,
        (64, 64, 720, 280),
        (-0.5, 11.5),
        (-4, 4),
        "PSRO iteration",
        "Win-rate points above 50%",
        y_format=lambda v: f"{v:+g}" if v else "0",
        x_ticks=range(12),
    )
    # One standard error of a single pairing measured over the GPU run's 512 games.
    noise = 100 * math.sqrt(0.25 / 512)
    band_top, band_bottom = plot.y(noise), plot.y(-noise)
    svg.add(
        f'<rect x="{plot.left}" y="{band_top:.1f}" width="{plot.right - plot.left}" '
        f'height="{band_bottom - band_top:.1f}" fill="{theme["muted"]}" fill-opacity="0.12"/>'
    )
    svg.text(
        plot.right - 4,
        band_bottom + 15,
        f"Shaded: ±{noise:.1f}, one standard error of a 512-game pairing",
        theme["muted"],
        11,
        "end",
    )
    zero = plot.y(0)
    svg.line(plot.left, zero, plot.right, zero, theme["axis"], 1.5)
    entries = []
    for color, folder, name in zip(
        theme["series"],
        ("gpu/psro", "psro"),
        ("GPU, 16M steps per response", "CPU, 4M steps per response"),
    ):
        history = json.loads((runs / folder / "population.json").read_text())["history"]
        points = [
            (plot.x(h["iteration"]), plot.y(50 * h["response_vs_previous_mixture"]))
            for h in history
        ]
        svg.path(points, color)
        for x, y in points:
            svg.dot(x, y, color)
        entries.append((color, name, "line"))
    legend(svg, 64, 30, entries)
    svg.save(out)


def figure_psro_payoffs(runs, theme, out):
    data = json.loads((runs / "gpu/psro/population.json").read_text())
    keep = [i for i, kind in enumerate(data["kinds"]) if kind < 0]
    names = [data["names"][i] for i in keep]
    points = [[50 * data["payoffs"][i][j] for j in keep] for i in keep]
    limit = 10.0
    cell = 34
    left, top = 90, 54
    size = cell * len(names)
    svg = Svg(
        left + size + 150,
        top + size + 70,
        theme,
        "Payoffs between GPU PSRO networks",
        "Win-rate points above 50% for the row network against the column network. "
        "Later responses mostly beat earlier ones by a few points.",
    )
    for r, row in enumerate(points):
        for c, value in enumerate(row):
            x, y = left + c * cell, top + r * cell
            svg.rect(x + 1, y + 1, cell - 2, cell - 2, diverging(theme, value, limit))
    for i, name in enumerate(names):
        short = name.replace("br_", "")
        svg.text(
            left - 8, top + i * cell + cell / 2 + 4, short, theme["ink2"], 11, "end"
        )
        svg.text(
            left + i * cell + cell / 2, top - 8, short, theme["ink2"], 11, "middle"
        )
    svg.text(
        left, top + size + 22, "Row network against column network", theme["ink2"], 12
    )
    svg.text(
        left - 80, 24, "Responses are numbered by PSRO iteration", theme["muted"], 11
    )
    bar_x, bar_top, bar_height = left + size + 28, top, size
    steps = 40
    for k in range(steps):
        value = limit - (k + 0.5) * 2 * limit / steps
        svg.rect(
            bar_x,
            bar_top + k * bar_height / steps,
            14,
            bar_height / steps + 0.5,
            diverging(theme, value, limit),
        )
    for value, label in (
        (limit, f"+{limit:g} or more"),
        (0, "Even"),
        (-limit, f"-{limit:g} or less"),
    ):
        y = bar_top + (limit - value) / (2 * limit) * bar_height
        svg.text(bar_x + 20, y + 4, label, theme["ink2"], 11)
    svg.save(out)


def figure_final(runs, theme, out):
    evals = json.loads((runs / "eval/summary.json").read_text())
    levels = json.loads((runs / "levels/confirm/summary.json").read_text())
    rows = [
        ("PSRO Hard: GPU PSRO, greedy", evals["PSRO-GPU vs F"]),
        ("GPU PSRO, sampled", evals["PSRO-GPU-sampled vs F"]),
        ("PPO Hard: GPU PPO, 200M", evals["PPO-GPU vs F"]),
        ("PSRO Medium: CPU PSRO", evals["PSRO-CPU vs F"]),
        ("CPU PPO, 30M", evals["PPO-CPU vs F"]),
        ("PPO Medium: 13M steps", levels["PPO-Medium vs F"]),
        ("F against itself (control)", evals["F-control vs F"]),
        ("PSRO Easy", levels["PSRO-Easy vs F"]),
        ("PPO Easy: 6.5M steps", levels["PPO-Easy vs F"]),
    ]
    row_height = 30
    top = 50
    svg = Svg(
        760,
        top + row_height * len(rows) + 50,
        theme,
        "Win rate against F for every agent",
        "Engine win rates against ValueFunctionPlayer with 95% intervals, from 32% for "
        "PPO Easy to 75% for PSRO Hard.",
    )
    plot = Plot(
        svg,
        (250, top, 700, top + row_height * len(rows)),
        (0, 100),
        (0, len(rows)),
        "Win rate vs F (%), 95% interval",
    )
    # Plot draws horizontal grid for a y scale; this chart reads along x, so redraw.
    svg.parts = svg.parts[:3]
    for tick in nice_ticks(0, 100, 5):
        x = plot.x(tick)
        svg.line(x, top - 6, x, plot.bottom, theme["grid"])
        svg.text(x, plot.bottom + 16, f"{tick:g}", theme["muted"], 11, "middle")
    svg.text(
        (plot.left + plot.right) / 2,
        plot.bottom + 36,
        "Win rate vs F (%), with 95% interval",
        theme["ink2"],
        12,
        "middle",
    )
    x50 = plot.x(50)
    svg.line(x50, top - 6, x50, plot.bottom, theme["axis"], 1.5)
    svg.text(x50, top - 14, "As strong as F", theme["muted"], 11, "middle")
    blue = theme["series"][0]
    for i, (label, row) in enumerate(rows):
        y = top + row_height * (i + 0.5)
        color = theme["muted"] if "control" in label else blue
        svg.text(plot.left - 14, y + 4, label, theme["ink"], 12, "end")
        low, high = 100 * row["ci_low"], 100 * row["ci_high"]
        svg.line(plot.x(low), y, plot.x(high), y, color, 2)
        svg.dot(plot.x(100 * row["win_rate"]), y, color)
        svg.text(
            plot.x(high) + 8, y + 4, f"{100 * row['win_rate']:.1f}%", theme["ink2"], 11
        )
    svg.save(out)


FIGURES = {
    "strength-over-training": figure_strength,
    "simulator-eval": figure_simulator,
    "training-diagnostics": figure_diagnostics,
    "psro-response-gains": figure_psro_gains,
    "psro-payoffs": figure_psro_payoffs,
    "final-strength": figure_final,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="catanatron/catanatron/jax_rl/figures")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        for mode, theme in THEMES.items():
            draw(runs, theme, out / f"{name}-{mode}.svg")
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
