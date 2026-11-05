"""
Helpers shared by the evaluator and the post-run scripts (import-light:
stdlib + yaml only, so scripts that never touch a solver stay cheap).

    redirect_fds_to(log_path)   send fd 1/2 (CasADi / acados / moto
                                chatter) to a file for the duration
    canonical_plan_hash(plan)   SHA-256 of a plan, tuples and lists alike
    plan_to_tuples(plan)        JSON round-trip lists back to tuples
    read_hydra_task(run_dir)    (scene, ic, description) of a run
    TASK_ENV_VARS / NONE_STRS   task-selection env vars and the strings
                                that mean "unset" in YAML / CLI values
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

TASK_ENV_VARS = {
    "scene": "SHINKA_LOCOMANIP_SCENE",
    "initial_condition": "SHINKA_LOCOMANIP_IC",
    "description": "SHINKA_LOCOMANIP_DESCRIPTION",
}
NONE_STRS = {"", "none", "null", "None", "Null", "NONE"}


@contextlib.contextmanager
def redirect_fds_to(log_path: str, mode: str = "a") -> Iterator[None]:
    """Redirect OS-level fd 1/2 to *log_path* so C-library output
    (CasADi/acados substitution messages, solver iterations, etc.) is
    captured along with Python stdout/stderr instead of spamming the
    console. The Python streams are flushed before/after."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(log_path, mode) as log_f:
            os.dup2(log_f.fileno(), 1)
            os.dup2(log_f.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def canonical_plan_hash(plan: Dict[str, List[Any]]) -> str:
    """SHA-256 of a canonical JSON form of the plan.

    Sorted channels, tuples coerced to lists so `(1, 1)` and `[1, 1]`
    hash to the same value. Empty tuples/lists render as `[]`.
    """
    canon: Dict[str, List[List[Any]]] = {}
    for ch in sorted(plan):
        seq = plan[ch]
        if not isinstance(seq, list):
            seq = [seq]
        canon[ch] = [list(e) if isinstance(e, (tuple, list)) else [e] for e in seq]
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_to_tuples(plan: Dict[str, List[Any]]) -> Dict[str, List[Tuple]]:
    """JSON round-trips tuples as lists; convert back so the solver's
    tuple-equality checks behave."""
    return {ch: [tuple(t) for t in seq] for ch, seq in plan.items()}


def read_hydra_task(run_dir: str) -> Optional[Tuple[str, str, str]]:
    """``(scene, ic, description)`` from ``<run>/.hydra/config.yaml``, or
    None if the file is missing, malformed or has no scene."""
    import yaml

    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    task = (cfg or {}).get("task") or {}
    scene = task.get("scene")
    if not scene:
        return None
    ic = task.get("initial_condition") or "default"
    desc = task.get("description") or ""
    return scene, ic, desc
