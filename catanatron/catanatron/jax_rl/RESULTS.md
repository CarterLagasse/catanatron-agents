# Agent results

Measured November 2026. All win rates come from the real Python engine, not the JAX
simulator, and all games are 1v1 on the BASE map. Trained agents and raw game records
live in `runs/`, which is gitignored, so they are not in this repository.

## Summary

**Use the GPU-trained PSRO agent with greedy moves** (`runs/gpu/psro`). It is the
strongest agent measured:

- It beats the engine's two strongest bots tested about 3 games in 4.
- It beats the other JAX agents head-to-head, though only narrowly against GPU PPO.
- It has not lost a game to the random or weighted random bots in 800 games.

```bash
catanatron-play --players=JAX:runs/gpu/psro:1,F --num=100
```

Most of the improvement came from training longer, which the GPU made affordable. PSRO
added a smaller gain on top. Strength against people has not been measured.

## Results

Numbers in brackets are 95% intervals. `F` is `ValueFunctionPlayer` and `AB:2` is
`AlphaBetaPlayer` at depth 2. Both are 50% matches for an equal opponent.

| Agent | vs `F` (4000 games) | vs `AB:2` (1000 games) | vs CPU PPO (2000 games) |
| --- | --- | --- | --- |
| **GPU PSRO, greedy** | **74.6%** (73.2–75.9) | **76.6%** (73.9–79.1) | **65.4%** (63.3–67.5) |
| GPU PSRO, sampled | 73.5% (72.1–74.8) | 72.6% (69.8–75.3) | 64.8% (62.7–66.9) |
| GPU PPO, greedy | 69.5% (68.1–70.9) | 70.5% (67.6–73.2) | 62.2% (60.1–64.3) |
| CPU PSRO, greedy | 61.5% (60.0–63.0) | 60.8% (57.7–63.8) | 52.3% (50.2–54.5) |
| CPU PPO, greedy | 59.7% (58.2–61.2) | 54.3% (51.2–57.4) | 50.6% (48.4–52.8), mirror |
| `F` itself | 48.1% (46.5–49.6), mirror | 49.9% (46.8–53.0) | 39.8% (37.6–41.9) |

The older Stable-Baselines3 agents won 0 of 30 games against `F` (interval 0–11%). They
were not part of this tournament, and used different seeds and far fewer games.

### Head-to-head

| Match | Games | Win rate for the first agent |
| --- | --- | --- |
| GPU PSRO vs GPU PPO | 4000 | 52.4% (50.8–53.9) |
| GPU PSRO vs CPU PSRO | 2000 | 64.2% (62.1–66.3) |
| GPU PPO vs CPU PSRO | 2000 | 61.0% (58.8–63.1) |
| GPU PSRO vs CPU PPO | 2000 | 65.4% (63.3–67.5) |
| GPU PPO vs itself (control) | 4000 | 49.8% (48.2–51.3) |

### Weak bots

GPU PSRO and GPU PPO each won 400 of 400 games against both the random bot and the
weighted random bot (interval 99.0–100%).

### Which agent is stronger?

Each row compares two agents on the same games. The numbers are how many more games out
of 100 the first agent wins against that opponent.

| Question | vs `F` | vs `AB:2` | vs CPU PPO | Answer |
| --- | --- | --- | --- | --- |
| Is GPU PSRO better than GPU PPO? | 5.0 more | 6.1 more | 3.2 more | Yes, by a modest margin |
| Is GPU PPO better than CPU PPO? | 9.8 more | 16.2 more | 11.6 more | Yes, by a lot |
| Is GPU PSRO better than CPU PSRO? | 13.0 more | 15.8 more | 13.1 more | Yes, by a lot |
| Is CPU PSRO better than CPU PPO? | 1.8 more | 6.5 more | 1.8 more | Probably; only clear vs `AB:2` |
| Is greedy better than sampled for GPU PSRO? | 1.1 more | 4.0 more | 0.5 more | At least as good; only clear vs `AB:2` |

### Exact ranges for those comparisons

These are the same comparisons with 95% ranges. If a range includes 0, the games cannot
show that either side is better.

| Comparison | vs `F` | vs `AB:2` | vs CPU PPO |
| --- | --- | --- | --- |
| GPU PSRO minus GPU PPO | +5.0 (+3.3 to +6.7) | +6.1 (+2.8 to +9.4) | +3.2 (+0.6 to +5.8) |
| GPU PPO minus CPU PPO | +9.8 (+8.0 to +11.7) | +16.2 (+12.3 to +20.1) | +11.6 (+8.9 to +14.3) |
| GPU PSRO minus CPU PSRO | +13.0 (+11.2 to +14.8) | +15.8 (+12.2 to +19.4) | +13.1 (+10.3 to +15.8) |
| CPU PSRO minus CPU PPO | +1.8 (+0.0 to +3.6) | +6.5 (+2.9 to +10.1) | +1.8 (-0.7 to +4.2) |
| Greedy minus sampled, GPU PSRO | +1.1 (-0.4 to +2.5) | +4.0 (+1.2 to +6.8) | +0.5 (-1.6 to +2.7) |

## What the results show

- **Longer training helped most.** PPO gained about 10 points against `F` going from 30M
  to 200M steps. The GPU only made that affordable: it does not change what a given
  number of steps learns. The two PPO runs also used different random seeds.
- **PSRO adds a real but smaller gain.** On GPU it adds 3 to 6 points depending on the
  opponent. Direct play gives the smallest figure, 52.4%.
- **Greedy is at least as good as sampled.** The gap is only clear against `AB:2`.
- **`F` and `AB:2` are about equal here.** `F` scored 49.9% against `AB:2`, so beating one
  by a given margin says similar things about the other.

## Difficulty levels in the UI

The game's player menu offers three levels of each algorithm. An opponent who wins 50%
against `F` is about as strong as that bot, so Easy is below it and Hard well above it.

| Menu option | Win rate vs `F` | Games | Agent |
| --- | --- | --- | --- |
| PSRO - Easy | 38.5% (35.5–41.6) | 1000 | `runs/levels/psro_easy_v2`, 4 short responses from PPO at 6.5M steps |
| PSRO - Medium | 61.5% (60.0–63.0) | 4000 | `runs/psro`, the CPU PSRO agent |
| PSRO - Hard | 74.6% (73.2–75.9) | 4000 | `runs/gpu/psro`, the GPU PSRO agent |
| PPO - Easy | 32.4% (29.6–35.4) | 1000 | `snapshots/it_00200.npz`, PPO at 6.5M steps |
| PPO - Medium | 50.6% (47.5–53.7) | 1000 | `snapshots/it_00400.npz`, PPO at 13M steps |
| PPO - Hard | 69.5% (68.1–70.9) | 4000 | `runs/gpu/ppo/selected.npz`, PPO at 200M steps |

The levels are not matched across algorithms: PSRO is the stronger of the two at each
level. These win rates are measured against a bot, so they do not say how hard each level
will feel to a person. The Easy agents still beat the weighted random bot at least 96.5% of
the time, so they play sensibly rather than randomly.

Easy PSRO keeps only the random and weighted bots in its population:

```bash
python -m catanatron.jax_rl.psro --run-dir runs/levels/psro_easy_v2 --iterations 4 \
    --first-response-timesteps 500000 --response-timesteps 500000 --eval-envs 512 --seed 3 \
    --scripted-population random,weighted --init-from runs/gpu/ppo/snapshots/it_00200.npz
```

Without that restriction, a short run from scratch ended with a mixture of 100%
`heuristic`, so it would have played a scripted bot rather than anything it trained.

## How to read the numbers

- **Sample sizes** were chosen so the intervals are small: about ±1.4 points at 4000
  games, ±2.2 at 2000 and ±3 at 1000.
- **Paired seeds.** Every agent facing a given opponent plays the same seeds, which set
  the board and all randomness at the start of each game. Games diverge once the agents
  choose differently, so the pairing removes board luck but not all luck.
- **Reproducible.** Agent sampling is seeded per game, and each game is saved to a
  `.jsonl` record, so an interrupted match continues where it stopped.
- **Controls pass.** Mirror matches should give 50%:
  - GPU PPO vs itself gave 49.8%, and CPU PPO vs itself gave 50.6%.
  - `F` vs `F` gave 48.1% on the first 4000 seeds, then 49.5% (48.4–50.6) on 8000 new ones.
    Combined, that is 49.05% (48.2–49.9) over 12,000 games.
  - So the agent's side may carry a bias of up to about 1–2 points in bot matches. That
    would make results against bots slightly conservative. Paired comparisons are not
    affected, because both agents sit on the same side.
- **Cross-check.** CPU PPO scored 59.7% against `F`, and `F` scored 39.8% against CPU PPO
  in a separate match. Those agree: 39.8% from the other side is 60.2%.
- **Seats.** Results by seat are in the tournament report. Seat 0 usually does a little
  better, by up to about 8 points in the closest matches, and seats were split about evenly.
- **Borderline results.** The report contains dozens of intervals, so a few near the edge
  would occur by chance. Treat CPU PSRO's +1.8 against `F` as weak evidence.
- **Selection inflates scores.** The chosen PPO checkpoint scored 75.0% in selection and
  69.5% on new seeds. Only the second number is reported as its strength.

## How the agents were trained

All agents use the same network: a graph actor-critic with 48 hidden units and 3 message
rounds, about 150 KB per checkpoint.

| | CPU run | GPU run |
| --- | --- | --- |
| Hardware | Core Ultra 7 155H | RTX 4050 Laptop, 6 GB |
| Speed | About 3,000 steps/s | About 40,000 steps/s |
| PPO steps | 30M | 200M |
| PPO time | About 3 hours | About 1.6 hours awake |
| PSRO | 12 responses of 4M steps | 12 responses of 16M steps |
| PSRO time | About 5 hours | About 1.3 hours |
| PSRO parallel games per payoff matchup | 128 | 512 |
| PSRO warm start | CPU PPO `best.npz` | GPU PPO selected checkpoint |

Final GPU PSRO mixture: `br_11` 48.5%, `br_08` 39.2%, `br_07` 12.0%, `br_09` 0.3%.

### Choosing the PPO checkpoint

PPO's own `best.npz` is picked from a few hundred simulated games against a scripted bot,
which is noisy. For the GPU run, six evenly spaced snapshots plus `best.npz` each played
600 engine games against `F`, on seeds not used in the results above:

| Checkpoint | Steps | Win rate vs `F` |
| --- | --- | --- |
| `it_06103` (selected, final) | 200M | 75.0% (71.4–78.3) |
| `best.npz` | | 71.5% (67.8–75.0) |
| `it_04150` | 136M | 70.7% (66.9–74.2) |
| `it_05150` | 169M | 70.2% (66.4–73.7) |
| `it_02150` | 70M | 67.3% (63.5–71.0) |
| `it_03150` | 103M | 65.3% (61.4–69.0) |
| `it_01150` | 38M | 58.8% (54.9–62.7) |

Strength was still rising late in training, so more steps may help further.

### Reproducing the GPU setup

JAX has no GPU support on native Windows, so training runs in WSL2. Evaluation stays on
the CPU because the engine decides one move at a time.

```bash
# In PowerShell
wsl --install -d Ubuntu-24.04

# In Ubuntu, using a virtual environment, from the repository root
pip install "jax[cuda12]" optax numpy networkx click rich
export PYTHONPATH=$PWD/catanatron XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m catanatron.jax_rl.ppo --run-dir runs/gpu/ppo --timesteps 200000000 --seed 1

# Pick a checkpoint by engine games, then copy it to runs/gpu/ppo/selected.npz
python -m catanatron.jax_rl.tournament --run-dir runs/gpu/selection --seed 200000 \
    --agent final=runs/gpu/ppo/latest.npz:1 --agent best=runs/gpu/ppo/best.npz:1 --vs F=600

python -m catanatron.jax_rl.psro --run-dir runs/gpu/psro --iterations 12 \
    --first-response-timesteps 16000000 --response-timesteps 16000000 \
    --eval-envs 512 --seed 1 --init-from runs/gpu/ppo/selected.npz --resume
```

The results above use this evaluation command, on a Windows CPU with these environment
variables set: `JAX_PLATFORMS=cpu` and
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.

```bash
python -m catanatron.jax_rl.tournament --run-dir runs/eval \
    --agent PPO-CPU=runs/ppo/best.npz:1 --agent PSRO-CPU=runs/psro:1 \
    --agent PPO-GPU=runs/gpu/ppo/selected.npz:1 --agent PSRO-GPU=runs/gpu/psro:1 \
    --agent PSRO-GPU-sampled=runs/gpu/psro --agent F-control=bot:F \
    --vs F=4000 --vs AB:2=1000 --vs JAX:runs/ppo/best.npz:1=2000
```

## Limits

- **Measured under the old rules.** These agents were trained and measured before six
  rules were corrected (see the main README's "Rule Changes in This Fork"). The old rules
  had two quirks they may exploit: cutting a road could hand Longest Road to a road under
  5 pieces, and a player could build onward from behind an enemy settlement that cut a
  loop. The 4-player agent was retrained under the new rules and was about as strong as
  before ([PSRO4_V2.md](PSRO4_V2.md)). The 1v1 agents here haven't been re-measured.
- **Reduced game.** Only 1v1 is covered, with no domestic trades.
- **Opponents are limited.** The agents have only been measured against these bots and
  each other, not people or stronger search such as deeper `AB` or MCTS.
- **Exploitability is unknown.** No opponent was trained specifically to beat the final
  agents. The `exploitability_within_population` value in `population.json` stays near 0
  by construction, so it says nothing about this.
- **One training seed per setup.** Differences between CPU and GPU runs mix the extra steps
  with seed luck, although their size makes the steps the likely cause.

## Tried or not done

- **Sparse game resets:** measured about 2x slower under load, so reverted. The cause was
  not isolated, and the GPU makes it matter less.
- **Multi-process CPU training:** dropped, because the GPU is about 13x faster.
- **More parallel games on GPU:** 1024 games at once gave 38,000 steps/s against 43,000 for
  256 in the throughput benchmark, because gradients take about two thirds of the time.
- **Not tried:** a wider network, more than 200M steps, and a JAX version of `F` to train
  against directly.
