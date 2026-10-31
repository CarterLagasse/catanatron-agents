"""Checkpoints as plain .npz files: flattened parameter arrays plus metadata."""

import json
import os
from pathlib import Path

import numpy as np


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for name, value in tree.items():
            yield from _flatten(value, f"{prefix}{name}/")
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            yield from _flatten(value, f"{prefix}#{i}/")
    else:
        yield prefix[:-1], np.asarray(tree)


def _insert(tree, parts, value):
    head, rest = parts[0], parts[1:]
    if not rest:
        tree[head] = value
    else:
        tree = tree.setdefault(head, {})
        _insert(tree, rest, value)


def _restore_lists(tree):
    if not isinstance(tree, dict):
        return tree
    if tree and all(k.startswith("#") for k in tree):
        ordered = sorted(tree, key=lambda k: int(k[1:]))
        return [_restore_lists(tree[k]) for k in ordered]
    return {k: _restore_lists(v) for k, v in tree.items()}


def save_params(path, params, hidden, rounds, map_type):
    """Writes atomically, so an interrupted run never leaves a corrupt checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(_flatten(params))
    meta = json.dumps({"hidden": hidden, "rounds": rounds, "map_type": map_type})
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, __meta__=np.array(meta), **arrays)
    os.replace(temporary, path)


def load_params(path):
    """Returns (params, meta)."""
    with np.load(path) as data:
        meta = json.loads(str(data["__meta__"]))
        tree = {}
        for name in data.files:
            if name != "__meta__":
                _insert(tree, name.split("/"), data[name])
    return _restore_lists(tree), meta
