# Benchmarking against published agents

Two outside opponents for testing new agents. Neither project's code is checked
in: each is downloaded on first use from a pinned version, checked against a
SHA-256 hash, and cached in `~/.cache/catanatron` (override with
`CATANATRON_CACHE_DIR`).

## HexMachina ("Agents of Change")

[Agents of Change: Self-Evolving LLM Agents for Strategic Planning](https://arxiv.org/abs/2506.04651)
(Belle et al., 2025) evolves a Python Catan player with a team of LLM agents
that rewrite its code between evaluation rounds against AlphaBeta. No LLM runs
at play time, so the evolved players are as fast as other search players.
The code comes from [nbelle1/strategy-game-agents](https://github.com/nbelle1/strategy-game-agents)
at commit `b1f2f1c`.

```bash
catanatron-play --players=HM,AB --num=100           # GPT-5-mini evolution (the paper's best)
catanatron-play --players=HM:mistral,AB --num=100
catanatron-play --players=HM:claude,AB --num=100
python -m catanatron.jax_rl.jax_player --agent runs/ppo/best.npz --opponents HM,AB:2
```

| Variant | Orchestrator | Paper win rate vs AlphaBeta |
| --- | --- | --- |
| `gpt` (default) | GPT-5-mini | 54.1% |
| `mistral` | Mistral-large | 49.2% |
| `claude` | Claude 3.7 | 38.4% |

The evolved files import an `adapters` module written for the older
catanatron fork in the authors' repo; `hexmachina/adapters.py` provides the same
names on top of this repo. The evolved code itself is unchanged except that the
player is told its real seat color (it hardcodes `Color.BLUE`) and its debug
prints are silenced (`HM:gpt:1` shows them).

**Results depend on the engine.** In 2-player games against AlphaBeta (depth 2),
`HM:gpt` won 102 of 200 (51%) in the authors' own catanatron fork, matching the
paper, but 80 of 200 (40%) in this repo (86 of 200 with
`--config-number-placement random`). The rules engine has changed since their
fork (for example, players now choose their discards instead of discarding at
random), and those changes favor AlphaBeta. Compare agents in the same engine.

## JSettlers2 robots

`catanatron-jsettlers` plays a catanatron player against the robots of
[JSettlers2](https://github.com/jdmonin/JSettlers2) 2.6.10, the long-standing
open-source Catan program whose robot AI is a common research baseline. Games
run on a real JSettlers server under its rules engine: the robots trade with
each other, the board is JSettlers' own, and information is hidden as in real
Catan. Requires Java 8 or newer on the PATH.

```bash
catanatron-jsettlers --player=AB:2 --num=100 --parallel=4
catanatron-jsettlers --player=HM:gpt --num=100 --parallel=4
catanatron-jsettlers --player=JAX:runs/gpu/psro4:1 --num=100 --parallel=4
catanatron-jsettlers --player=JAX:runs/gpu/psro:1 --players=2 --num=100 --parallel=4
catanatron-jsettlers --player=F --num=20 --check-tracking   # test the bridge itself
```

By default each game has 4 players: the catanatron player and 3 JSettlers
"smart" robots, so an equally strong player wins 25%. `--players=2` plays 1v1
against one robot (50% for an equal player). A JAX agent plays only the game size
it was trained for: 1v1 agents need `--players=2`, and 4-player agents come from
`--num-players 4` training (see `jax_rl/README.md`). Use `--robots=N` with N ≥ 4 to also include JSettlers'
weaker "fast" robots (30% of N, rounded down). `--player` accepts any
`catanatron-play` code, and `--code` loads custom players the same way.

How it works: the server is started with three built-in robots and a robot
connection cookie. The bridge (`jsettlers/client.py`) connects as a third-party
robot. A second connection (`jsettlers/starter.py`) logs in as the server's
`debug` user and starts robot-only games, and the server gives the bridge one
seat in each. The bridge tracks each game from the server's messages
(`jsettlers/view.py`). Whenever the server waits on it, the bridge builds a
catanatron `Game` matching the JSettlers position, asks the player to decide,
and sends the move. The board maps onto catanatron's base map, so node and edge
ids are the same as in normal catanatron games.

Differences from playing inside catanatron:

- **Hidden information.** The player sees its own hand exactly, as in
  JSettlers. Opponents' cards of unknown type (after steals between other
  players) are assigned to the resources they produce most, and their unplayed
  dev cards are sampled from the unseen cards, excluding victory points. With
  `--fully-observable`, JSettlers announces every card (option `PLAY_FO`),
  which matches catanatron's full information.
- **Trading.** catanatron players never trade with other players, so the
  bridge rejects all robot offers. The robots still trade among themselves.
- **Bank.** JSettlers' bank is unlimited; the snapshot uses catanatron's
  19-card bank, which only matters in rare situations.
- **Declined moves.** If the server refuses a move (it shouldn't), the bridge
  stops offering that move type for the rest of the turn and warns. The summary
  reports how many moves were declined.
- **Failed games.** Now and then the JSettlers robots leave a game partway
  through, or never join one, so it can't finish. A game with no activity for
  twice `--turn-timeout` (10 minutes by default), or one that ends without a
  winner, is abandoned: it isn't counted, a new game replaces it, and the
  summary lists it. If more than 5% of games fail, the run stops with an error.

`--check-tracking` makes the server send the bots everyone's true hands at the
end of each turn (`jsettlers.debug.bots.datacheck.rsrc`). The bridge compares
these with its tracked hands and, at game end, compares its tracked VPs with
the final scores, then reports mismatches. The player never sees these true
hands.

**Security note:** while it runs, the JSettlers server listens on all network
interfaces (it has no bind-address setting) and enables its passwordless
`debug` user. Keep the port (default 8880) firewalled from other machines.
