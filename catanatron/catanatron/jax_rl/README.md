# Pure-JAX agent training

Everything needed to train runs inside XLA: the game rules, observations, network,
PPO and PSRO. PyTorch, Stable-Baselines3 and Gymnasium are not imported. The Python
engine remains the source of truth for the rules and still powers the UI and CLI.

```bash
pip install -e ".[jax]"
```

Scope: 1v1 games, on the BASE, TOURNAMENT and MINI maps. Domestic trades are not
modelled, the same as in the gym environment.

[RESULTS.md](RESULTS.md) has measured agent strength and how to reproduce it. To play
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

## Guarantees and how they are checked

- `tests/jax_rl/test_parity.py` replays engine games through JAX without re-syncing,
  comparing full state and legal moves after every action.
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
