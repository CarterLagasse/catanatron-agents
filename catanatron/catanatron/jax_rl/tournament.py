"""Engine matches sized and reported so the outcome can actually be read.

Every agent plays the same seeds against an opponent, so comparisons are paired.
Finished games are saved as they end: rerun the same command after a crash and only
the missing games are played.

    python -m catanatron.jax_rl.tournament --run-dir runs/eval \
        --agent PPO=runs/ppo/best.npz:1 --agent PSRO=runs/psro:1 \
        --agent F-control=bot:F \
        --vs F=4000 --vs AB:2=1000

An agent is NAME=PATH, with :1 for greedy play, or NAME=bot:CODE for an engine bot.
An opponent is SPEC=GAMES, using CLI player codes, e.g. JAX:runs/ppo/best.npz:1=2000.
A control such as F against F must come out near 50%: if not, the harness is biased.
"""

import argparse
import ctypes
import json
import math
import multiprocessing
import re
import sys
from pathlib import Path

from catanatron.jax_rl.jax_player import play_match, summarize

Z_95 = 1.96


def paired_difference(records_a, records_b):
    """Win-rate difference of A minus B over the seeds both played."""
    b_by_seed = {r["seed"]: r for r in records_b}
    deltas = [
        (r["result"] == "win") - (b_by_seed[r["seed"]]["result"] == "win")
        for r in records_a
        if r["seed"] in b_by_seed
    ]
    count = len(deltas)
    if count < 2:
        return None
    mean = sum(deltas) / count
    variance = sum((d - mean) ** 2 for d in deltas) / (count - 1)
    spread = Z_95 * math.sqrt(variance / count)
    return {"games": count, "difference": mean, "ci_low": mean - spread, "ci_high": mean + spread}


def verdict(low, high, even=0.5):
    if low > even:
        return "better"
    if high < even:
        return "worse"
    return "not separated"


def write_report(run_dir, agents, opponents, results):
    lines = [
        "# Tournament results",
        "",
        "Real Python engine. Intervals are 95%: Wilson for win rates, paired over shared "
        "seeds for differences. *not separated* means the interval includes even, so "
        "the games played cannot tell the two sides apart.",
        "",
        "| Agent | Opponent | Games | Wins | Draws | Win rate | 95% CI | Against 50% "
        "| Seat 0 / 1 | Mean VPs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (agent, opponent), records in results.items():
        r = summarize(records)
        lines.append(
            f"| {agent} | {opponent} | {r['games']} | {r['wins']} | {r['draws']} "
            f"| {r['win_rate']:.1%} | {r['ci_low']:.1%} to {r['ci_high']:.1%} "
            f"| {verdict(r['ci_low'], r['ci_high'])} "
            f"| {r['seat0_win_rate']:.1%} / {r['seat1_win_rate']:.1%} | {r['mean_vps']:.2f} |"
        )

    names = list(agents)
    compared = []
    for opponent, _ in opponents:
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                if (first, opponent) in results and (second, opponent) in results:
                    difference = paired_difference(
                        results[first, opponent], results[second, opponent]
                    )
                    if difference is not None:
                        compared.append((first, second, opponent, difference))
    if compared:
        lines += [
            "",
            "## Paired differences",
            "",
            "| A | B | Opponent | Games | A minus B | 95% CI | Result |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for first, second, opponent, d in compared:
            result = {"better": "A higher", "worse": "B higher"}.get(
                verdict(d["ci_low"], d["ci_high"], even=0.0), "not separated"
            )
            lines.append(
                f"| {first} | {second} | {opponent} | {d['games']} | {d['difference']:+.1%} "
                f"| {d['ci_low']:+.1%} to {d['ci_high']:+.1%} | {result} |"
            )
    (Path(run_dir) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {f"{a} vs {o}": summarize(records) for (a, o), records in results.items()}
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2))


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def _keep_awake():
    if sys.platform == "win32":
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this process lives.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default="runs/eval")
    parser.add_argument(
        "--agent", action="append", required=True, help="NAME=PATH[:1] or NAME=bot:CODE"
    )
    parser.add_argument("--vs", action="append", required=True, help="SPEC=GAMES")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--map-type", default="BASE")
    parser.add_argument("--seed", type=int, default=100_000)
    args = parser.parse_args()

    _keep_awake()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    agents = dict(spec.split("=", 1) for spec in args.agent)
    opponents = [(s.rsplit("=", 1)[0], int(s.rsplit("=", 1)[1])) for s in args.vs]

    results = {}
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for opponent, games in opponents:
            for name, spec in agents.items():
                greedy = spec.endswith(":1")
                path = spec[:-2] if greedy else spec
                records_path = run_dir / f"{_slug(name)}__{_slug(opponent)}.jsonl"
                results[name, opponent] = play_match(
                    path, opponent, games, pool, greedy, args.map_type, args.seed, records_path
                )
                r = summarize(results[name, opponent])
                print(
                    f"{name} vs {opponent}: {r['wins']}/{r['games']} = {r['win_rate']:.1%} "
                    f"({r['ci_low']:.1%} to {r['ci_high']:.1%})",
                    flush=True,
                )
                write_report(run_dir, agents, opponents, results)


if __name__ == "__main__":
    main()
