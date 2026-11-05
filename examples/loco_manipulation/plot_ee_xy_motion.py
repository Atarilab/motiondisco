"""Plot EE x/y/z trajectories across the valid contact plans per scene.

Walks ``<results_root>/<scene>/<run>/gen_*/results/to_run.pkl``, runs
forward kinematics on each saved trajectory via pinocchio, and overlays
the world-frame positions of the left / right end-effectors across all
plans. The full trajectory is plotted (contact and free-swing); use
``--converged-only`` to keep only converged TO solves. ``best/`` dirs
are skipped — they are byte-duplicates of some ``gen_N``.

One PNG is written per scene under ``<results_root>/ee_xy_motion/``,
each with three stacked subplots (x, y, z) vs normalized trajectory
progress in [0, 1]. Both EEs share the same axes per coordinate;
linestyle ``-`` = left_ee, ``--`` = right_ee. Color encodes the plan.

Requires the faro src dir on PYTHONPATH so the scene model can
be loaded for FK:

    PYTHONPATH=submodules/faro/src \\
    python examples/loco_manipulation/plot_ee_xy_motion.py \\
        --results-root results/loco_abl3_ik_fb

No acados / mujoco simulation is needed — we only read saved qpos
trajectories and compute pinocchio frame placements.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sibling modules importable regardless of caller cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_pkls(results_root: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """Map scene name → list of ``(run_name, gen_name, pkl_path)``."""
    out: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for scene_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(scene_dir):
            continue
        scene_name = os.path.basename(scene_dir)
        for run_dir in sorted(glob.glob(os.path.join(scene_dir, "*"))):
            if not os.path.isdir(run_dir):
                continue
            if not os.path.isdir(os.path.join(run_dir, ".hydra")):
                continue
            run_name = os.path.basename(run_dir)
            for entry in sorted(os.listdir(run_dir)):
                if not entry.startswith("gen_"):
                    continue
                pkl_path = os.path.join(run_dir, entry, "results", "to_run.pkl")
                if not os.path.isfile(pkl_path):
                    continue
                out[scene_name].append((run_name, entry, pkl_path))
    return out


# ---------------------------------------------------------------------------
# Per-plan EE extraction
# ---------------------------------------------------------------------------


def _ee_xyz(
    pin, pin_model, pin_data, frame_id: int,
    qpos_traj: np.ndarray,
) -> np.ndarray:
    """Return (T, 3) of world XYZ at each frame across the whole trajectory."""
    T = qpos_traj.shape[0]
    out = np.empty((T, 3), dtype=float)
    for k in range(T):
        q = qpos_traj[k]
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacement(pin_model, pin_data, frame_id)
        out[k] = pin_data.oMf[frame_id].translation[:3]
    return out


def _resolve_frame_id(pin, pin_model, name: str) -> Optional[int]:
    """Return the pinocchio frame id for *name*, or None if missing."""
    if hasattr(pin_model, "existFrame") and not pin_model.existFrame(name):
        return None
    fid = pin_model.getFrameId(name)
    if fid >= pin_model.nframes:
        return None
    return int(fid)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _resample_to_grid(arr: np.ndarray, n_grid: int) -> np.ndarray:
    """Linear-interp an (T, 3) trajectory onto a common normalized grid
    of length ``n_grid`` so per-plan trajectories can be stacked for
    mean/variance computation."""
    T = arr.shape[0]
    if T == 0:
        return np.full((n_grid, 3), np.nan)
    if T == 1:
        return np.tile(arr, (n_grid, 1))
    src_t = np.linspace(0.0, 1.0, T)
    dst_t = np.linspace(0.0, 1.0, n_grid)
    out = np.empty((n_grid, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.interp(dst_t, src_t, arr[:, c])
    return out


Panel = Tuple[str, str, str, List[np.ndarray], List[np.ndarray]]
# (column_title, left_name, right_name, left_xyzs, right_xyzs)


# Publication-friendly font sizes for embedded LaTeX figures.
_LATEX_RC = {
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
}


def _plot_scene_mean(
    scene_name: str,
    panels: List[Panel],
    labels: List[str],
    out_path: str,
    n_grid: int = 200,
) -> None:
    """Grid figure (3 rows × n_panels columns) of mean ± 3σ trajectories.
    Rows are x / y / z; columns are panels (e.g. hands, feet). Within
    each panel, left = tab:blue solid, right = tab:orange dashed.
    Variance is the per-step std across plans, shown as a translucent
    band."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not labels or not panels:
        return

    n_rows = len(panels)
    coord_names = ("x", "y", "z")
    t_grid = np.linspace(0.0, 1.0, n_grid)
    left_color = "tab:blue"
    right_color = "tab:orange"

    with plt.rc_context(_LATEX_RC):
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(6 * 3, 4 * n_rows),
            sharex=True, squeeze=False,
        )

        for row, (title, left_name, right_name, lxyzs, rxyzs) in enumerate(panels):
            left_stack = np.stack(
                [_resample_to_grid(a, n_grid) for a in lxyzs], axis=0
            )
            right_stack = np.stack(
                [_resample_to_grid(a, n_grid) for a in rxyzs], axis=0
            )
            mean_l = np.nanmean(left_stack, axis=0)
            std_l = np.nanstd(left_stack, axis=0)
            mean_r = np.nanmean(right_stack, axis=0)
            std_r = np.nanstd(right_stack, axis=0)

            for col in range(3):
                ax = axes[row, col]
                ax.fill_between(
                    t_grid,
                    mean_l[:, col] - 3.0 * std_l[:, col],
                    mean_l[:, col] + 3.0 * std_l[:, col],
                    color=left_color, alpha=0.2, linewidth=0,
                )
                ax.fill_between(
                    t_grid,
                    mean_r[:, col] - 3.0 * std_r[:, col],
                    mean_r[:, col] + 3.0 * std_r[:, col],
                    color=right_color, alpha=0.2, linewidth=0,
                )
                ax.plot(
                    t_grid, mean_l[:, col],
                    color=left_color, linestyle="-", linewidth=2.4,
                )
                ax.plot(
                    t_grid, mean_r[:, col],
                    color=right_color, linestyle="--", linewidth=2.4,
                )

            axes[row, 0].set_ylabel(title)
            axes[row, 0].legend(
                handles=[
                    Line2D([0], [0], color=left_color, linestyle="-",
                           label=f"{left_name} mean"),
                    Patch(facecolor=left_color, alpha=0.2,
                          label=f"{left_name} ±3σ"),
                    Line2D([0], [0], color=right_color, linestyle="--",
                           label=f"{right_name} mean"),
                    Patch(facecolor=right_color, alpha=0.2,
                          label=f"{right_name} ±3σ"),
                ],
                loc="upper left",
            )

        for col, c in enumerate(coord_names):
            axes[0, col].set_title(f"{c} [m]")
            axes[-1, col].set_xlabel("normalized trajectory progress")

        for ax in axes.flat:
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)


def _plot_scene(
    scene_name: str,
    panels: List[Panel],
    labels: List[str],
    out_path: str,
) -> None:
    """Grid figure (3 rows × n_panels columns) overlaying every plan's
    trajectory. Rows are x / y / z; columns are panels (e.g. hands,
    feet). Within each panel left = tab:blue, right = tab:orange,
    solid lines for both, low alpha so the overlay stays readable."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not labels or not panels:
        return

    n_rows = len(panels)
    coord_names = ("x", "y", "z")
    left_color = "tab:blue"
    right_color = "tab:orange"

    with plt.rc_context(_LATEX_RC):
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(6 * 3, 4 * n_rows),
            sharex=True, squeeze=False,
        )

        for row, (title, left_name, right_name, lxyzs, rxyzs) in enumerate(panels):
            for lxyz, rxyz in zip(lxyzs, rxyzs):
                for arr, color in ((lxyz, left_color), (rxyz, right_color)):
                    T = arr.shape[0]
                    t = np.linspace(0.0, 1.0, T) if T > 1 else np.array([0.0])
                    for col in range(3):
                        axes[row, col].plot(
                            t, arr[:, col],
                            color=color, alpha=0.5, linewidth=2.0,
                            linestyle="-",
                        )

            axes[row, 0].set_ylabel(title)
            axes[row, 0].legend(
                handles=[
                    Line2D([0], [0], color=left_color, linestyle="-",
                           label=left_name, linewidth=2.0),
                    Line2D([0], [0], color=right_color, linestyle="-",
                           label=right_name, linewidth=2.0),
                ],
                loc="upper left",
            )

        for col, c in enumerate(coord_names):
            axes[0, col].set_title(f"{c} [m]")
            axes[-1, col].set_xlabel("normalized trajectory progress")

        for ax in axes.flat:
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Directory whose children are <scene>/<run>/ (e.g. "
             "results/loco_abl3_ik_fb).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs (default: <results-root>/ee_xy_motion/).",
    )
    parser.add_argument(
        "--converged-only",
        action="store_true",
        help="Plot only plans where to_converged=True in the pkl. "
             "Default: include any plan that produced a to_run.pkl "
             "(matches the broadened 'valid' definition).",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated subset of scenes to plot (default: all).",
    )
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    if not os.path.isdir(results_root):
        parser.error(f"results-root {results_root!r} is not a directory")

    out_dir = args.out_dir or os.path.join(results_root, "ee_xy_motion")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
    except ImportError as e:
        parser.error(f"matplotlib is required: {e}")
    try:
        import pinocchio as pin
    except ImportError as e:
        parser.error(
            f"pinocchio is required: {e}. "
            "Make sure PYTHONPATH includes submodules/faro/src "
            "and your env has pinocchio installed."
        )

    from _task_loader import load_task  # noqa: E402

    scene_pkls = _discover_pkls(results_root)
    if args.scenes:
        wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scene_pkls = {k: v for k, v in scene_pkls.items() if k in wanted}

    print(f"[scan] {sum(len(v) for v in scene_pkls.values())} to_run.pkl "
          f"file(s) across {len(scene_pkls)} scene(s)")

    for scene_name in sorted(scene_pkls):
        entries = scene_pkls[scene_name]
        try:
            task = load_task(scene=scene_name)
        except Exception as e:
            print(f"[skip] {scene_name}: load_task failed: {type(e).__name__}: {e}")
            continue

        robot = task.scene.robots[task.robot_id]
        patch_names = list(robot.patch_ees)
        required_channels = ("left_ee", "right_ee", "left_sole", "right_sole")
        missing_channels = [ch for ch in required_channels if ch not in patch_names]
        if missing_channels:
            print(
                f"[skip] {scene_name}: patch_ees={patch_names} "
                f"missing {missing_channels}"
            )
            continue

        pin_model = robot.pin_model
        pin_data = pin_model.createData()
        fids: Dict[str, int] = {}
        missing_fids: List[str] = []
        for ch in required_channels:
            fid = _resolve_frame_id(pin, pin_model, ch)
            if fid is None:
                missing_fids.append(ch)
            else:
                fids[ch] = fid
        if missing_fids:
            print(
                f"[skip] {scene_name}: pinocchio frames {missing_fids} not found"
            )
            continue

        xyzs: Dict[str, List[np.ndarray]] = {ch: [] for ch in required_channels}
        labels: List[str] = []
        n_kept = 0
        n_total = 0

        for run_name, gen_name, pkl_path in entries:
            n_total += 1
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[warn] {pkl_path}: {type(e).__name__}: {e}")
                continue

            if args.converged_only and not data.get("to_converged"):
                continue

            plan = data.get("contact_mode_seq") or {}
            results = data.get("results") or {}
            qpos_traj = results.get("robot_xs")
            # Keyed by robot instance ID since the faro port; older
            # pickles hold a bare (T, nx) array for the sole robot.
            if isinstance(qpos_traj, dict):
                qpos_traj = qpos_traj.get(task.robot_id)
            if not isinstance(plan, dict) or qpos_traj is None:
                print(f"[warn] {pkl_path}: missing plan / robot_xs; skipping")
                continue
            missing_plan_channels = [
                ch for ch in required_channels if ch not in plan
            ]
            if missing_plan_channels:
                print(
                    f"[warn] {pkl_path}: plan missing {missing_plan_channels}; "
                    "skipping"
                )
                continue

            qpos_traj = np.asarray(qpos_traj, dtype=float)
            # robot_xs rows carry more columns than pin_model.nq (extras
            # — likely velocities or stacked object state); take only
            # the leading nq columns so FK accepts the configuration.
            if qpos_traj.shape[1] < pin_model.nq:
                print(
                    f"[warn] {pkl_path}: robot_xs has {qpos_traj.shape[1]} "
                    f"columns, pin_model.nq={pin_model.nq}; skipping"
                )
                continue
            qpos_traj = qpos_traj[:, : pin_model.nq]

            for ch in required_channels:
                xyzs[ch].append(
                    _ee_xyz(pin, pin_model, pin_data, fids[ch], qpos_traj)
                )
            labels.append(f"{run_name}/{gen_name}")
            n_kept += 1

        if not labels:
            print(f"[skip] {scene_name}: 0 plans plotted (of {n_total} found)")
            continue

        panels: List[Panel] = [
            ("Hands", "left_ee", "right_ee",
             xyzs["left_ee"], xyzs["right_ee"]),
            ("Feet", "left_sole", "right_sole",
             xyzs["left_sole"], xyzs["right_sole"]),
        ]
        overlay_path = os.path.join(out_dir, f"{scene_name}.png")
        mean_path = os.path.join(out_dir, f"{scene_name}_mean.png")
        _plot_scene(scene_name, panels, labels, overlay_path)
        _plot_scene_mean(scene_name, panels, labels, mean_path)
        print(
            f"[plot] {scene_name}: {n_kept}/{n_total} plans -> "
            f"{os.path.basename(overlay_path)}, "
            f"{os.path.basename(mean_path)}"
        )


if __name__ == "__main__":
    main()
