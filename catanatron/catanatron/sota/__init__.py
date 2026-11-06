"""Published Catan agents from outside this repo, for benchmarking against.

- hexmachina: the LLM-evolved players from "Agents of Change: Self-Evolving
  LLM Agents for Strategic Planning" (Belle et al., arXiv:2506.04651).
  Plays inside catanatron, so it works anywhere a Player does:
      catanatron-play --players=HM,AB --num=100
- jsettlers: plays any catanatron Player against the JSettlers2 robots on a
  real JSettlers server, under JSettlers' own rules engine:
      catanatron-jsettlers --player=F --num=20

Neither project's code is checked in here. Both are downloaded on first use
from a pinned version and verified by SHA-256 (see _download.py).
"""
