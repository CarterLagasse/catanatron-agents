import contextlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path

from catanatron.models.player import Player
from catanatron.sota._download import cache_dir, fetch

REPO = "nbelle1/strategy-game-agents"
COMMIT = "b1f2f1c6d43afd8d5ce5ca71542b78d37173b2a1"

# variant -> (path in the authors' repo, sha256 of that file at COMMIT)
VARIANTS = {
    "gpt": (
        "supplementary/Evolutions/GPT Evolution 2/foo_player.py",
        "b01c69386a7a362365085e4b6866a003c93068e6911d79a958c999ce72c7c692",
    ),
    "mistral": (
        "supplementary/Evolutions/Mistral Evolution/foo_player.py",
        "b6c16c9cb80d4f360497a280f3fe460abe6988795efca8a99ba49c0a754bb1de",
    ),
    "claude": (
        "supplementary/Evolutions/Claude Evolution/foo_player.py",
        "835dd3212316c694bf2a2303df634b7ecc10ad0da58b01acc3c653ac9b4befd4",
    ),
}

_PACKAGE = __name__.rpartition(".")[0]


def variant_path(variant: str) -> Path:
    """Local path of the evolved player file, downloading it if needed."""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown HexMachina variant {variant!r}; try {list(VARIANTS)}"
        )
    repo_path, sha256 = VARIANTS[variant]
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{repo_path.replace(' ', '%20')}"
    dest = cache_dir() / "hexmachina" / COMMIT[:7] / f"{variant}_foo_player.py"
    return fetch(url, dest, sha256)


def load_foo_player_class(variant: str):
    # Loaded as a submodule of this package so its `from .adapters import ...`
    # resolves to our adapters.py.
    module_name = f"{_PACKAGE}._evolved_{variant}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, variant_path(variant)
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
    return sys.modules[module_name].FooPlayer


class HexMachinaPlayer(Player):
    """HexMachina player evolved by LLM agents ("Agents of Change", arXiv:2506.04651).
    Params: VARIANT (gpt/mistral/claude, default gpt), VERBOSE (1/0). E.g. HM:claude"""

    def __init__(self, color, variant="gpt", verbose=False):
        super().__init__(color)
        self.variant = variant or "gpt"
        self.verbose = str(verbose).lower() in ("1", "true", "yes")
        self._warned_invalid = False
        foo_player_class = load_foo_player_class(self.variant)
        with self._quiet():
            # The evolved __init__ hardcodes Color.BLUE (it always sat in seat 2
            # during evolution), so set the real color afterwards.
            self._impl = foo_player_class()
        self._impl.color = color

    def _quiet(self):
        # The evolved code prints debug traces on most decisions.
        if self.verbose:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(_devnull())

    def decide(self, game, playable_actions):
        self._impl.color = self.color  # callers may reassign color after construction
        with self._quiet():
            action = self._impl.decide(game, playable_actions)
        if action not in playable_actions:
            if not self._warned_invalid:
                warnings.warn(
                    f"{self} returned unplayable {action}; using first playable action"
                )
                self._warned_invalid = True
            action = playable_actions[0]
        return action

    def reset_state(self):
        self._impl.reset_state()

    def __repr__(self):
        return super().__repr__() + f"(variant={self.variant})"


_DEVNULL = None


def _devnull():
    global _DEVNULL
    if _DEVNULL is None:
        _DEVNULL = open(os.devnull, "w")
    return _DEVNULL
