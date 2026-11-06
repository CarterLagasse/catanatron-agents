"""HexMachina: LLM-evolved Catan players from "Agents of Change" (arXiv:2506.04651).

HexMachina is a multi-agent LLM system that iteratively rewrites a Python
player (foo_player.py) and keeps the best version by its win rate against
AlphaBetaPlayer. No LLM runs at play time; the evolved players are ordinary
search + heuristic code. Table 2 of the paper, 2 players vs AlphaBeta(depth 2),
10 VPs:

    variant   orchestrator    paper win rate   source evolution
    gpt       GPT-5-mini      54.1%            GPT Evolution 2, evolution 10
    mistral   Mistral-large   49.2%            Mistral Evolution, evolution 1
    claude    Claude 3.7      38.4%            Claude Evolution, evolution 2

Usage:
    catanatron-play --players=HM,AB --num=100          # gpt variant
    catanatron-play --players=HM:claude,AB --num=100
    python -m catanatron.jax_rl.jax_player --agent runs/ppo/best.npz --opponents HM,AB:2

The evolved files are downloaded from the authors' repo at a pinned commit on
first use (that repo has no license file, so they aren't vendored here).
"""

from catanatron.sota.hexmachina.player import VARIANTS, HexMachinaPlayer

__all__ = ["HexMachinaPlayer", "VARIANTS"]
