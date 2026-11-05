"""
Replay a saved TO trajectory (the ``to_run.pkl`` files written by
evaluate.py) in a mujoco viewer.

Usage:

    PYTHONPATH=submodules/faro/src \\
    python examples/loco_manipulation/replay_to_run.py \\
        examples/loco_manipulation/results/<eval_id>/to_run.pkl

Controls (mujoco passive viewer defaults): drag to rotate, scroll to
zoom, ESC to quit. The script loops the trajectory until you close
the window unless ``--once`` is passed.

The active scene is auto-detected from the run's ``.hydra/config.yaml``
(walking up from the pickle path), so no env vars are required when
replaying a pickle produced by ``motiondisco_launch``. For pickles outside a
Hydra run tree, the scripts fall back to ``SHINKA_LOCOMANIP_{SCENE,IC,
DESCRIPTION}`` env vars and finally to built-in defaults; see
``_task_loader.load_task``.

This script intentionally does NOT instantiate a ToTask, so you do
not need ACADOS_SOURCE_DIR set or libhpipm.so on the loader path
just to view a saved run. It only loads the task to grab the
scene's MuJoCo model (``faro.configs.mujoco.build_mujoco``).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

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
    write_trajectory_mp4,
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


def replay(
    pickle_path: Path,
    dt: float,
    loop: bool,
    save_video: bool = True,
    video_size: tuple = (1280, 720),
    fps: int = 30,
) -> None:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    cost = data.get("to_cost")
    conv = data.get("to_converged")
    print(f"Loaded {pickle_path} (to_cost={cost!r}, to_converged={conv!r})")

    task = load_task(**_task_kwargs_from_hydra(pickle_path))
    if task.discrete_goal_name:
        task_label = f"{task.scene_name}/{task.ic_name}/{task.discrete_goal_name}"
    else:
        task_label = f"{task.scene_name}/{task.ic_name}"
    desc = task.description or "(no description)"
    print(f"Active task: {task_label} ({desc})")

    torques = data["results"].get("torques")
    if torques is not None:
        # Keyed by robot instance ID since the faro port; older pickles
        # hold a bare array for the sole robot.
        limits = data["results"].get("torque_limit")
        if not isinstance(torques, dict):
            torques, limits = {task.robot_id: torques}, {task.robot_id: limits}
        for rid, rob_torques in torques.items():
            arr = np.asarray(rob_torques)
            tmax = float(np.max(np.abs(arr))) if arr.size else 0.0
            limit = (limits or {}).get(rid)
            extra = f" (limit={limit})" if limit is not None else ""
            print(f"torque trajectory [{rid}]: {arr.shape}, max|tau|={tmax:.2f}{extra}")

    mj_model = build_replay_model(task)
    qpos_traj = build_qpos_trajectory(data["results"], task, mj_model)
    T = qpos_traj.shape[0]

    # Optimizer-aware per-step durations (dt_traj[i] * t_scale[i]); when
    # absent (older pickles), fall back to the uniform --dt argument.
    step_ts_arr = data["results"].get("step_ts")
    if step_ts_arr is not None:
        step_ts_arr = np.asarray(step_ts_arr, dtype=float).reshape(-1)
        print(
            f"qpos trajectory: {T} frames x {qpos_traj.shape[1]} dofs, "
            f"variable-time (mean dt={float(step_ts_arr.mean()):.4f}s, "
            f"min={float(step_ts_arr.min()):.4f}s, "
            f"max={float(step_ts_arr.max()):.4f}s)"
        )
    else:
        print(
            f"qpos trajectory: {T} frames x {qpos_traj.shape[1]} dofs, "
            f"dt={dt:.3f}s/frame (uniform fallback — pickle has no step_ts)"
        )

    if save_video:
        video_path = pickle_path.with_suffix(".mp4")
        write_trajectory_mp4(
            mj_model, qpos_traj, video_path, dt=dt,
            width=video_size[0], height=video_size[1],
            step_ts=step_ts_arr, fps=fps,
        )

    interactive_replay(
        mj_model=mj_model,
        qpos_traj=qpos_traj,
        step_ts=step_ts_arr,
        dt=dt,
        loop=loop,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "pickle_path", type=Path,
        help="Path to a to_run.pkl produced by evaluate.py",
    )
    p.add_argument(
        "--dt", type=float, default=0.05,
        help=(
            "Uniform fallback dt (seconds/frame) used only when the pickle "
            "has no per-step timing array. Default: 0.05 (= time_per_mode / "
            "nodes_per_mode in evaluate.py)."
        ),
    )
    p.add_argument(
        "--fps", type=int, default=30,
        help=(
            "Constant mp4 fps when variable-time playback is active "
            "(pickle carries step_ts). Each rendered frame is replicated "
            "round(step_ts[k]*fps) times. Default: 30."
        ),
    )
    p.add_argument(
        "--once", action="store_true",
        help="Play the trajectory once instead of looping",
    )
    p.add_argument(
        "--no-video", action="store_true",
        help="Skip writing the mp4 next to the pickle",
    )
    p.add_argument(
        "--video-size", type=str, default="1280x720",
        help="mp4 resolution as WIDTHxHEIGHT (default: 1280x720)",
    )
    args = p.parse_args()
    try:
        w, h = (int(x) for x in args.video_size.lower().split("x"))
    except ValueError:
        p.error(f"--video-size expects WIDTHxHEIGHT, got {args.video_size!r}")
    replay(
        args.pickle_path, dt=args.dt, loop=not args.once,
        save_video=not args.no_video, video_size=(w, h), fps=args.fps,
    )


if __name__ == "__main__":
    main()
