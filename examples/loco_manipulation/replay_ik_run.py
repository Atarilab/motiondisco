"""
Replay a saved IK trajectory (the ``ik_run.pkl`` files written by
evaluate.py) in a mujoco viewer.

Usage:

    PYTHONPATH=submodules/faro/src \\
    python examples/loco_manipulation/replay_ik_run.py \\
        examples/loco_manipulation/results/<eval_id>/ik_run.pkl

Controls (mujoco passive viewer defaults): drag to rotate, scroll to
zoom, ESC to quit. The script loops the trajectory until you close
the window unless ``--once`` is passed.

The active scene is auto-detected from the run's ``.hydra/config.yaml``
(walking up from the pickle path), so no env vars are required when
replaying a pickle produced by ``motiondisco_launch``. For pickles outside a
Hydra run tree, the scripts fall back to ``SHINKA_LOCOMANIP_{SCENE,IC,
DESCRIPTION}`` env vars and finally to built-in defaults; see
``_task_loader.load_task``.

This script intentionally does NOT instantiate a KsoTask, so you
do not need ACADOS_SOURCE_DIR set or libhpipm.so on the loader path
just to view a saved run. It only loads the task to build the scene's
MuJoCo model with ``faro.configs.mujoco.build_mujoco``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# Make sibling modules importable regardless of caller cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shinka.utils import load_hydra_config  # noqa: E402

from _task_loader import load_task  # noqa: E402
from _replay_common import (  # noqa: E402
    build_qpos_trajectory,
    build_replay_model,
    interactive_replay,
)


def _task_kwargs_from_hydra(pickle_path: Path) -> dict:
    """Read ``task.{scene,initial_condition,description}`` from the nearest
    ``.hydra/config.yaml`` above ``pickle_path``. Returns ``{}`` when the
    pickle is not inside a Hydra run tree, so ``load_task`` then falls
    back to env vars / built-in defaults."""
    cfg = load_hydra_config(str(pickle_path.parent), max_parent_depth=3)
    if cfg is None:
        print(
            f"[replay] no .hydra/config.yaml found above {pickle_path}; "
            "falling back to SHINKA_LOCOMANIP_* env vars / defaults."
        )
        return {}
    task_cfg = cfg.get("task", {}) or {}
    return {
        "scene": task_cfg.get("scene"),
        "initial_condition": task_cfg.get("initial_condition"),
        "description": task_cfg.get("description"),
    }


def replay(pickle_path: Path, dt: float, loop: bool) -> None:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {pickle_path} (ik_cost={data.get('ik_cost')!r})")

    task = load_task(**_task_kwargs_from_hydra(pickle_path))
    if task.discrete_goal_name:
        task_label = f"{task.scene_name}/{task.ic_name}/{task.discrete_goal_name}"
    else:
        task_label = f"{task.scene_name}/{task.ic_name}"
    desc = task.description or "(no description)"
    print(f"Active task: {task_label} ({desc})")
    mj_model = build_replay_model(task)

    qpos_traj = build_qpos_trajectory(data["results"], task, mj_model)
    T = qpos_traj.shape[0]
    print(
        f"qpos trajectory: {T} frames x {qpos_traj.shape[1]} dofs, "
        f"dt={dt:.3f}s/frame"
    )

    interactive_replay(
        mj_model=mj_model,
        qpos_traj=qpos_traj,
        step_ts=None,
        dt=dt,
        loop=loop,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "pickle_path", type=Path,
        help="Path to an ik_run.pkl produced by evaluate.py",
    )
    p.add_argument(
        "--dt", type=float, default=0.05,
        help=(
            "Seconds between frames (default: 0.05 = time_per_mode / "
            "nodes_per_mode in evaluate.py). KSO has no per-node t_scale, "
            "so this uniform value is real-time playback."
        ),
    )
    p.add_argument(
        "--once", action="store_true",
        help="Play the trajectory once instead of looping",
    )
    args = p.parse_args()
    replay(args.pickle_path, dt=args.dt, loop=not args.once)


if __name__ == "__main__":
    main()
