# Pure-JAX agent training

Everything needed to train runs inside XLA: the game rules, observations, network,
PPO and PSRO. PyTorch, Stable-Baselines3 and Gymnasium are not imported. The Python
engine remains the source of truth for the rules and still powers the UI and CLI.

```bash
pip install -e ".[jax]"
```

Scope: 2 to 4 players, on the BASE, TOURNAMENT and MINI maps. Domestic trades are
not modelled, the same as in the gym environment. 1v1 is the default; see
[More than two players](#more-than-two-players) for 3 and 4.

[RESULTS.md](RESULTS.md) has measured agent strength and how to reproduce it.
[IMPROVING.md](IMPROVING.md) has the training curves and what to try next. To play
against the agents, start the web server from the repository root in an environment
with JAX, then pick a PSRO or PPO level as Player 2. They are listed only if their
files exist under `runs/`.

## Train

```bash
# PPO: self-play, a league of past selves, and scripted bots
python -m catanatron.jax_rl.ppo --run-dir runs/ppo --timesteps 50000000

# PSRO: best responses against the Nash mixture of a growing population
python -m catanatron.jax_rl.psro --run-dir runs/psro --iterations 12
```

PPO writes `best.npz`, `latest.npz`, `snapshots/` and `metrics.jsonl`. PSRO writes one
`br_XX.npz` per iteration and `population.json`, which holds the payoff matrix and
the Nash mixture.

## Evaluate in the real engine

```bash
# Comparisons that can be trusted: shared seeds, saved games, controls, intervals
python -m catanatron.jax_rl.tournament --run-dir runs/eval \
    --agent PPO=runs/ppo/best.npz:1 --agent PSRO=runs/psro:1 --agent F-control=bot:F \
    --vs F=4000 --vs AB:2=1000

# Quick looks
python -m catanatron.jax_rl.jax_player --agent runs/ppo/best.npz --opponents R,W,F,AB:2 --games 200
catanatron-play --players=JAX:runs/ppo/best.npz,F --num=100
catanatron-play --players=JAX:runs/psro,F --num=100   # plays the PSRO mixture
```

`tournament` writes `report.md`. Rerun the same command to finish an interrupted
run. `F-control` plays the engine's bot against itself and must land near 50%. For a
95% interval of about +/- 1.5 points at these win rates, play about 4000 games.

Use relative paths in `--players`: a Windows drive colon would split the player spec.

## More than two players

`--num-players 3` or `4` trains agents for bigger tables, such as the JSettlers
benchmark (one agent against 3 JSettlers robots, see `catanatron/sota/`). Training
stays in the JAX simulator, about as fast per move as 1v1; JSettlers games are far
too slow to train on, so they are used only to measure.

```bash
# Start from a 1v1 agent: --init-from widens a 1v1 checkpoint automatically
python -m catanatron.jax_rl.psro --run-dir runs/psro4 --num-players 4     --init-from runs/gpu/psro/br_11.npz --scripted-population heuristic     --iterations 12 --first-response-timesteps 32000000 --response-timesteps 24000000     --eval-envs 512 --eval-ticks 1600 --resume
python -m catanatron.jax_rl.ppo --run-dir runs/ppo4 --num-players 4     --init-from runs/gpu/ppo/selected.npz

# A 1v1 agent turned into a 4-player one without training, as a baseline
python -m catanatron.jax_rl.widen runs/gpu/psro runs/widened/psro4 --num-players 4

# Measure: engine games against a table of 3 bots, and the JSettlers robots
python -m catanatron.jax_rl.tournament --run-dir runs/eval4 --num-players 4     --agent PSRO4=runs/psro4:1 --vs F=1000 --vs AB:2=300
catanatron-jsettlers --player=JAX:runs/psro4:1 --num=300 --parallel=4
```

An equally strong player wins 1 game in P. How it works:

- **Seats and actions.** Seat 0 moves first and play goes up the seats. The robber has
  one action per tile for each possible victim, named by how many seats after the
  mover the victim sits; that is the only change to the action space.
- **Observations.** The 1v1 features stay first. Board features in them ("an
  opponent's road") cover all opponents, and global ones describe the next player.
  One block per opponent is appended to every node, edge, tile and the globals.
- **Rewards.** +1 for a win and -1/(P-1) for a loss, so rewards sum to zero and an
  equal player expects 0. VP shaping uses each seat's lead over the average opponent.
- **PPO.** In pool and scripted games the learner takes one random seat and the
  opponent fills all the others; in self-play games every seat learns.
- **PSRO.** `payoffs[i, j]` is i's reward with j in every other seat, measured in both
  directions. The meta-solver works on `(payoffs - payoffs.T) / 2`, which in 1v1 is
  the payoff matrix itself. Responses train against tables drawn from the mixture.
- **Warm starts.** `network.adapt_params` gives the new inputs zero weights and copies
  the 1v1 steal logit to every victim, so a widened agent first plays exactly as the
  1v1 one would, treating every opponent as "the opponent". The widened GPU PSRO
  agent wins 99.8% against tables of 3 random bots and 48% against 3 heuristic bots.

A checkpoint records its player count, and an agent only plays games of that size.
Older checkpoints, saved without a player count, are 1v1.

[PSRO4_V2.md](PSRO4_V2.md) follows the latest 4-player run round by round, with its
engine and JSettlers results.

## Layout

| File | Purpose |
| --- | --- |
| `topology.py` | Board tables, derived from the Python map templates, and the action layout |
| `env.py` | Rules as pure functions; `batch_step` is the training entry point |
| `obs.py` | Ego-centric observations with hidden information removed |
| `network.py` | Graph actor-critic with per-node, per-edge and per-tile action logits |
| `bots.py` | Scripted opponents written in JAX |
| `ppo.py` | Rollout, per-seat GAE, updates, league training loop |
| `psro.py` | Meta-solver, payoff matrix, best-response loop |
| `evaluate.py` | Fast JAX-vs-JAX matches |
| `bridge.py`, `jax_player.py` | Run trained agents in the Python engine |
| `tournament.py` | Large engine matches with a readable report |
| `benchmark.py` | Measures simulation and training throughput |
| `widen.py` | Turns a 1v1 agent into a 3- or 4-player one, as a baseline or warm start |

## Guarantees and how they are checked

- `tests/jax_rl/test_parity.py` replays engine games through JAX without re-syncing,
  comparing full state and legal moves after every action, with 2, 3 and 4 players.
  It also checks the rare rule cases of `tests/test_rules.py` (Longest Road after a
  broken road, winning only on your own turn, bank shortages) position by position.
- `tests/jax_rl/test_env.py` checks longest road against brute force, card
  conservation, and that observations do not leak the opponent's hidden cards.
- `tests/jax_rl/test_training.py` checks GAE by hand-computed cases, action-head
  wiring, a real learning test, and the meta-solver on known games.

## Performance notes

- On native Windows JAX is CPU-only. To use an NVIDIA GPU, run under WSL2 or Linux
  with `pip install "jax[cuda12]"`; no code changes are needed. An RTX 4050 Laptop
  trains at about 43,000 ticks/s and 1 GB of VRAM with the defaults, against about
  3,000 ticks/s on a Core Ultra 7 155H. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Keep `--num-envs` at 256 on that GPU: 1024 was slower (38,000 ticks/s), because
  each sample still costs a gradient, which takes about two thirds of the time.
- Gradients are the bottleneck on CPU, so width (`--hidden`) sets training speed.
- Engine evaluation stays on the CPU: it decides one move at a time, which a GPU
  would slow down.
- Rollouts store packed game states and rebuild observations per minibatch, which
  keeps buffers small. If memory is tight, raise `num_minibatches` in `PPOConfig`.
