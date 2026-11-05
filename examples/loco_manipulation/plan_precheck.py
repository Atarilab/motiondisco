"""
Plan-level proposal filter for the search (``EvolutionConfig.proposal_filter``).

Runs in the runner process right after a proposed program applied and
before the code embedding, novelty judge and evaluation. It executes the
proposed ``generate_contact_plan`` in a subprocess (scene construction
only, 1-3 s; no solver is built) and rejects, without spending a KSO /
TO evaluation (minutes):

  - an exact duplicate of a plan this run already evaluated
    (``seen_plans.json``; a provisional record from an unfinished
    evaluation does not count), which would otherwise score 0;
  - a plan whose strategy family (``plan_signature``: same contacts in
    the same order, only step counts differ) already has
    ``SHINKA_PRECHECK_MAX_FAMILY`` (default 3) valid plans in this run.

Anything else is accepted, including programs that crash or produce an
over-cap plan: the evaluator's own short-circuits give the LLM that
feedback. Rejections re-enter the runner's resample loop with a new
parent. Configure with::

    evo_config:
      proposal_filter: examples.loco_manipulation.plan_precheck.precheck
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from plan_signature import plan_patches, plan_signature  # noqa: E402

MAX_FAMILY_ENV = "SHINKA_PRECHECK_MAX_FAMILY"
_SCENE_ENV = {"scene": "SHINKA_LOCOMANIP_SCENE", "initial_condition": "SHINKA_LOCOMANIP_IC",
              "description": "SHINKA_LOCOMANIP_DESCRIPTION"}

_RUNNER = (
    "import json, runpy, sys\n"
    "ns = runpy.run_path(sys.argv[1])\n"
    "out = ns['run_planning']()\n"
    "plan = out[0] if isinstance(out, tuple) else out\n"
    "print('PLAN ' + json.dumps({ch: [list(e) for e in seq] for ch, seq in plan.items()}))\n"
)


def _run_root(results_dir: str) -> str:
    # <run_root>/gen_N/results, as in evaluate._find_run_root.
    return os.path.dirname(os.path.dirname(os.path.abspath(results_dir)))


def _scene_env(results_dir: str) -> Dict[str, str]:
    """SHINKA_LOCOMANIP_* for the subprocess: the runner's own environment
    (run_evo.py exports it) or the run's Hydra config (motiondisco_launch)."""
    env = dict(os.environ)
    if env.get(_SCENE_ENV["scene"]):
        return env
    cfg_path = os.path.join(_run_root(results_dir), ".hydra", "config.yaml")
    try:
        import yaml

        with open(cfg_path) as f:
            task = (yaml.safe_load(f) or {}).get("task") or {}
    except Exception:  # noqa: BLE001
        return env
    for field, var in _SCENE_ENV.items():
        value = task.get(field)
        if value not in (None, "", "None", "null"):
            env[var] = str(value)
    return env


def extract_plan(exec_fname: str, results_dir: str = "", timeout: float = 60.0) -> Optional[Dict[str, List[Any]]]:
    """The plan a proposed program produces, or None if it cannot be run."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, os.path.abspath(exec_fname)],
            cwd=_THIS_DIR, env=_scene_env(results_dir), capture_output=True, text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("PLAN "):
            try:
                raw = json.loads(line[5:])
            except json.JSONDecodeError:
                return None
            return {ch: [tuple(e) for e in seq] for ch, seq in raw.items()}
    return None


def _seen_plans(results_dir: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(_run_root(results_dir), "seen_plans.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def precheck(
    exec_fname: str,
    results_dir: str,
    parent: Any = None,
    generation: Any = None,
    *,
    plan_loader: Optional[Callable[[str, str], Optional[Dict[str, List[Any]]]]] = None,
    max_family: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Proposal filter entry point: ``(accept, info)``."""
    loader = plan_loader or extract_plan
    plan = loader(exec_fname, results_dir)
    if not isinstance(plan, dict) or not plan:
        return True, {"reason": "plan could not be extracted; left to the evaluator"}
    from evaluate import _canonical_plan_hash

    plan_hash = _canonical_plan_hash(plan)
    signature = plan_signature(plan)
    patches = plan_patches(plan)
    seen = _seen_plans(results_dir)
    info: Dict[str, Any] = {"plan_hash": plan_hash[:12], "signature": signature,
                            "plan_patches": patches}
    prior = seen.get(plan_hash)
    if prior is not None and not prior.get("provisional"):
        info["reason"] = (f"duplicate of {prior.get('first_seen_gen')} "
                          f"(score {float(prior.get('combined_score') or 0.0):.3f})")
        return False, info
    cap = max_family if max_family is not None else int(os.environ.get(MAX_FAMILY_ENV, "3"))
    valid = [r for r in seen.values()
             if r.get("signature") == signature and float(r.get("combined_score") or 0.0) > 0.0]
    info["family_valid_count"] = len(valid)
    if cap > 0 and len(valid) >= cap:
        info["reason"] = (f"strategy family already has {len(valid)} valid plans (cap {cap}); "
                          f"only step counts differ")
        return False, info
    info["reason"] = "accepted"
    return True, info
