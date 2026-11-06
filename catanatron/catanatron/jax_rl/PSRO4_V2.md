# 4-player PSRO v2: how the agents developed

This run retrained the 4-player PSRO agent after the rule fixes of November 2026 (Longest
Road after a broken road, roads ending at an opponent's settlement, winning only on your
own turn, and bank shortages). It ran 12 rounds on 4 November 2026. Each round trained one
new agent, a best response to the current mix of agents.

## Setup

- **Warm start:** the final agent of the first 4-player run (`runs/gpu/psro4/br_11.npz`).
  It joins the population as `init`.
- **Population:** the scripted heuristic bot and `init`, plus one new agent per round.
- **Each round:** 24M training steps against the current mix, then each new agent plays
  every member of the population in the simulator, with 512 games in parallel for 1,600
  steps.
- **Hardware:** RTX 4050 Laptop GPU through WSL, at about 48k steps/s. In total, 288M steps
  took 2.6 hours.

## Round by round

Payoff is +1 for a win and −1/3 for a loss, so 0 means even with the other players at the
table. "vs heuristic" puts the agent in one seat and the heuristic bot in the other three.
The win rate next to it ignores draws. "vs init" is the payoff the meta-solver uses,
(A − Aᵀ)/2. "Gain" is the new agent's payoff against the mix it was trained against.

| Round | Agent | Trained against | vs heuristic | vs init | Gain | Time |
|---|---|---|---|---|---|---|
| – | init | – | +0.544 (≈66%) | – | – | – |
| 0 | br_00 | init | +0.562 (≈67%) | −0.004 | −0.009 | 9.8 min |
| 1 | br_01 | init | +0.554 (≈67%) | −0.007 | +0.001 | 10.0 min |
| 2 | br_02 | init | +0.552 (≈66%) | +0.022 | +0.021 | 9.6 min |
| 3 | br_03 | br_02 | +0.548 (≈66%) | +0.010 | −0.014 | 9.6 min |
| 4 | br_04 | br_02 | +0.543 (≈66%) | +0.035 | +0.003 | 9.8 min |
| 5 | br_05 | br_04 | +0.566 (≈68%) | +0.014 | −0.010 | 10.1 min |
| 6 | br_06 | br_04 | +0.575 (≈68%) | +0.024 | +0.014 | 10.4 min |
| 7 | br_07 | br_06 | **+0.606 (≈71%)** | +0.029 | −0.016 | 11.0 min |
| 8 | br_08 | br_06 | +0.601 (≈70%) | +0.035 | +0.038 | 10.9 min |
| 9 | br_09 | br_06/07/08 | +0.601 (≈70%) | **+0.041** | −0.010 | 11.6 min |
| 10 | br_10 | br_06/07/08 | +0.577 (≈68%) | +0.030 | +0.010 | 39.8 min* |
| 11 | br_11 | br_10 | +0.587 (≈69%) | +0.020 | +0.010 | 15.3 min* |

\*Round 10 includes 27 minutes of laptop standby. Round 11 ran on battery, at about 39k
steps/s.

The final meta-strategy mixes three agents: br_09 at 18%, br_10 at 41% and br_11 at 42%.
Loading `runs/gpu/psro4_v2` as a player uses that mix.

## What happened

- **Rounds 0–4: flat.** Against the heuristic bot the agents stayed at the warm start's
  level, +0.54 to +0.56. The meta-strategy stayed on `init` for the first three rounds,
  so the first new agents couldn't beat it. The v1 agent was already close to an
  equilibrium under the corrected rules.
- **Rounds 5–7: the climb.** The score against the heuristic bot rose from +0.57 to
  +0.61 at br_07, about 66% to 71% wins at a table with three heuristic bots.
- **Rounds 8–11: plateau.** The score settled between +0.58 and +0.60, and the
  meta-strategy moved on to the newest agents.
- **Small, steady gain over the warm start.** From round 4 on, every new agent beat
  `init` head to head, by +0.01 to +0.04.
- **No round clearly beat the mix it trained against.** Each gain was between −0.016 and
  +0.038, and an evaluation's noise is about ±0.02. During training the new agents won
  24–30% of games against their mix, close to the 25% that means even at a 4-player
  table. By this measure PSRO had converged at this budget per round.

## Takeaways

- **About 8 rounds were enough for a warm-started run.** Rounds 9–12 cost about 45
  minutes, not counting the standby, and added nothing measurable. A cold start (the v1
  run went from +0.43 to +0.56) keeps improving for longer.
- **To get a stronger agent, change the training, not the round count.** Each round
  mostly continues training the previous agent. A larger budget per round, a bigger
  network or more varied opponents in the population, such as AlphaBeta or the v1
  agent, are more likely to help than more rounds.
- **Outside the simulator, v2 is as strong as v1, not stronger.** Both were tested
  under the corrected rules. The differences are paired over the same seeds; neither
  rules out zero.

  | Opponent | Games | v2 | v1 | v2 − v1 |
  |---|---|---|---|---|
  | `F` | 1,000 | 59.5% (56.4–62.5%) | 60.5% (57.4–63.5%) | −1.0% (−4.9 to +2.9) |
  | `AB:2` | 300 | 57.7% (52.0–63.1%) | 53.3% (47.7–58.9%) | +4.3% (−2.7 to +11.3) |
  | 3 JSettlers robots | 300 | 42.3% (36.9–48.0%) | 43.4% (37.9–49.1%)* | – |

  An equally strong player wins 25% of 4-player games. \*v1's JSettlers result is from
  its own earlier run of 295 games, not paired with v2's. All 300 JSettlers games
  finished, with no declined moves and no abandoned games.
