#!/usr/bin/env python3
import argparse
import os
import sys

import yaml

# Make sibling modules importable regardless of caller cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

from prompts import build_task_sys_msg
from _run_common import TASK_ENV_VARS  # noqa: E402


_TASK_ENV_VARS = TASK_ENV_VARS

# Eval-side ablation toggles read by evaluate.py. Booleans in the YAML
# task: block (`disable_to: true`, `hide_ik_feedback: true`); exported
# as "1"/unset rather than passed to build_task_sys_msg.
_TASK_ABLATION_FLAGS = {
    "disable_to": "SHINKA_DISABLE_TO",
    "hide_ik_feedback": "SHINKA_HIDE_IK_FEEDBACK",
    # Opt-in plan-level diversity (plan_signature.py).
    "diversity": "SHINKA_DIVERSITY",
}
# Valued task: keys exported verbatim (scoring.py: legacy / quality).
_TASK_VALUE_VARS = {
    "score_mode": "SHINKA_SCORE_MODE",
}


def _export_task_env(task_cfg: dict) -> dict:
    """Set SHINKA_LOCOMANIP_* from the yaml task: block and return the
    normalized kwargs dict for build_task_sys_msg."""
    kwargs = {}
    for field, env_name in _TASK_ENV_VARS.items():
        val = task_cfg.get(field) if task_cfg else None
        if val is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = str(val)
            kwargs[field] = str(val)
    for field, env_name in _TASK_ABLATION_FLAGS.items():
        if task_cfg and task_cfg.get(field):
            os.environ[env_name] = "1"
        else:
            os.environ.pop(env_name, None)
    for field, env_name in _TASK_VALUE_VARS.items():
        val = task_cfg.get(field) if task_cfg else None
        if val is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = str(val)
    return kwargs


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Task selection: propagate yaml `task:` block to env vars so
    # child evaluator processes (env.py / ik_eval.py) see them, then
    # pass the same values to build_task_sys_msg directly.
    task_kwargs = _export_task_env(config.get("task") or {})
    config["evo_config"]["task_sys_msg"] = build_task_sys_msg(**task_kwargs)

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="01:15:00",  # KSO + diagnosis + TO; see task/loco_manipulation.yaml
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_small.yaml")
    args = parser.parse_args()
    main(args.config_path)
