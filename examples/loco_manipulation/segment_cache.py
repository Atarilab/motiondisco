"""Per-run cache of solved TO windows (incremental evaluation).

A plan evaluated window by window (``to_eval._solve_segmented`` with
``SHINKA_TO_INCREMENTAL=1``) stores every window that converged or was
physically valid, keyed by everything its solution depends on: the plan
up to the window's end (the window itself and, through the chain, the
start it was solved from), the start state's numbers, and the solver
settings. A later evaluation whose plan keeps that prefix finds the window
here and does not solve it again, so a long plan that changes only after
mode 50 costs only the modes after 50. Failed windows are not stored (TO
is not deterministic; a retry may pass).

One pickle per key, written to a temporary name and renamed, so
concurrent evaluators never read a half-written entry and need no lock.
The directory is ``<run>/segment_cache`` (``SHINKA_SEGMENT_CACHE_DIR``,
set by evaluate.py); without it the cache is off.

    key(plan, end_mode, start, settings)     -> hex digest
    Cache(dir).get(key) / .put(key, record)  -> record dict / None
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from typing import Any, Dict, Mapping, Optional

import numpy as np

CACHE_DIR_ENV = "SHINKA_SEGMENT_CACHE_DIR"
# Start-state numbers are rounded to this many decimals before hashing: a
# start reproduced from cached windows is bit-identical, a re-solved one
# differs by far more than this.
START_DECIMALS = 6


def _canonical_prefix(plan: Mapping[str, Any], end_mode: int) -> str:
    canon = {}
    for ch in sorted(plan):
        seq = plan[ch]
        canon[ch] = [list(e) if isinstance(e, (tuple, list)) else [e] for e in list(seq)[:end_mode]]
    return json.dumps(canon, sort_keys=True, separators=(",", ":"))


def _start_bytes(start: Any) -> bytes:
    if start is None:
        return b"initial"
    parts = [np.round(np.asarray(start.robot_x, float), START_DECIMALS).tobytes()]
    for name in sorted(start.obj_x):
        parts.append(name.encode("utf-8"))
        parts.append(np.round(np.asarray(start.obj_x[name], float), START_DECIMALS).tobytes())
    return b"|".join(parts)


def settings_digest(settings: Mapping[str, Any]) -> str:
    """Digest of the solver settings a window's solution depends on."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def key(plan: Mapping[str, Any], end_mode: int, start: Any, settings: Mapping[str, Any]) -> str:
    """Cache key of the window ending at ``end_mode`` (exclusive) solved
    from ``start`` (None: the task's initial state) under ``settings``."""
    h = hashlib.sha256()
    h.update(_canonical_prefix(plan, end_mode).encode("utf-8"))
    h.update(b"\x00")
    h.update(_start_bytes(start))
    h.update(b"\x00")
    h.update(settings_digest(settings).encode("utf-8"))
    return h.hexdigest()


class Cache:
    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def path(self, k: str) -> str:
        return os.path.join(self.dir, f"{k}.pkl")

    def get(self, k: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self.path(k), "rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            return None

    def put(self, k: str, record: Dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp_", suffix=".pkl")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(record, f)
            os.replace(tmp, self.path(k))
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def __len__(self) -> int:
        return sum(1 for name in os.listdir(self.dir) if name.endswith(".pkl") and not name.startswith(".tmp_"))


def open_cache(directory: Optional[str] = None) -> Optional[Cache]:
    """The run's cache, or None when no directory is configured."""
    directory = directory or os.environ.get(CACHE_DIR_ENV)
    if not directory:
        return None
    try:
        return Cache(directory)
    except OSError as e:
        print(f"[segment_cache] cannot use {directory}: {e}", flush=True)
        return None
