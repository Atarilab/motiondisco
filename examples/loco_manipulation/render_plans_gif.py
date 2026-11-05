"""
Render the TO trajectories of a run's valid, distinct plans as GIFs.

    python examples/loco_manipulation/render_plans_gif.py \\
        --run-dir results/<exp>/<scene>/<run>

Valid = TO converged with a usable cost (to_cost > 0). Identical contact
plans are kept once (best-scoring generation). Writes, under
``<run>/plan_gifs/`` (or ``--out``):

    gen_<N>.gif     one GIF per plan, played at real time from the
                    optimizer's per-node durations (``step_ts``)
    overview.gif    all plans tiled in a grid, each normalized to the same
                    number of frames, labeled with generation, score and
                    contact patches
    index.txt       the plans in grid order with their patch sets

Rendering is offscreen with MuJoCo (``MUJOCO_GL``, default glfw). The
scene comes from the run's ``.hydra/config.yaml``. No solvers run.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from plan_signature import plan_patches  # noqa: E402
from run_to_all import _discover_plan_dirs, _read_hydra_task  # noqa: E402


def collect_valid_plans(run_dir: str) -> List[Dict[str, Any]]:
    """Converged, distinct plans with a saved TO trajectory, best first."""
    best: Dict[str, Dict[str, Any]] = {}
    for label, results_dir in _discover_plan_dirs(run_dir):
        if label == "best":
            continue
        pkl = os.path.join(results_dir, "to_run.pkl")
        try:
            with open(os.path.join(results_dir, "extra.json")) as f:
                extra = json.load(f)
            with open(os.path.join(results_dir, "metrics.json")) as f:
                score = float(json.load(f).get("combined_score") or 0.0)
        except (OSError, json.JSONDecodeError):
            continue
        plan = extra.get("contact_mode_seq")
        if (not isinstance(plan, dict) or not extra.get("to_converged")
                or not (extra.get("to_cost") or 0) > 0 or not os.path.isfile(pkl)):
            continue
        key = json.dumps(plan, sort_keys=True)
        if key not in best or score > best[key]["score"]:
            best[key] = {"gen": label, "score": score, "pkl": pkl,
                         "patches": plan_patches(plan)}
    return sorted(best.values(), key=lambda p: -p["score"])


def _frames_at(times: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(times, t_query, side="right") - 1, 0, len(times) - 1)


def render_plan(renderer, mj_model, qpos: np.ndarray, frame_ids: np.ndarray,
                camera) -> List[np.ndarray]:
    import mujoco

    data = mujoco.MjData(mj_model)
    out = []
    for k in frame_ids:
        data.qpos[:] = qpos[k]
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=camera)
        out.append(renderer.render().copy())
    return out


def _label(img: np.ndarray, lines: List[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    y = 3
    for line in lines:
        draw.rectangle([2, y - 1, 4 + 6 * len(line), y + 10], fill=(0, 0, 0))
        draw.text((3, y - 1), line, fill=(255, 255, 255))
        y += 12
    return np.asarray(im)


def _short_patches(patches: str) -> List[str]:
    """Hand / object part of a patch string, split for a tile label."""
    keep = [p for p in patches.split("|") if not p.startswith(("left_sole", "right_sole"))]
    return [p[:46] for p in keep]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", help="Output dir (default: <run-dir>/plan_gifs)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--size", default="480x360", help="per-plan GIF size WxH")
    p.add_argument("--tile", default="320x240", help="overview tile size WxH")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--overview-frames", type=int, default=60)
    p.add_argument("--max-seconds", type=float, default=40.0,
                   help="cap per-plan GIF length")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    import imageio.v2 as imageio
    import mujoco

    from _replay_common import build_qpos_trajectory, build_replay_model
    from _task_loader import load_task

    run_dir = os.path.abspath(args.run_dir)
    task_info = _read_hydra_task(run_dir)
    if task_info is None:
        sys.exit(f"{run_dir} has no .hydra/config.yaml with a task.scene")
    scene, ic, desc = task_info
    task = load_task(scene=scene, initial_condition=ic, description=desc)
    mj_model = build_replay_model(task)
    out = os.path.abspath(args.out or os.path.join(run_dir, "plan_gifs"))
    os.makedirs(out, exist_ok=True)

    plans = collect_valid_plans(run_dir)
    print(f"{len(plans)} valid distinct plans in {run_dir}")
    if not plans:
        return

    w, h = (int(x) for x in args.size.split("x"))
    tw, th = (int(x) for x in args.tile.split("x"))
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, w, tw)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, h, th)
    renderer = mujoco.Renderer(mj_model, height=h, width=w)
    tile_renderer = mujoco.Renderer(mj_model, height=th, width=tw)

    # Side view of the whole workspace: robot start, objects and supports.
    pts = [st.q[:3] for st in task.resolved_state.env_objects.values()]
    pts += [st.q[:3] for st in task.resolved_state.fixed_env_objects.values()
            if np.linalg.norm(st.q[:2]) > 1e-6]
    pts += [st.q[:3] for st in task.resolved_state.robots.values()]
    pts = np.array(pts)
    center = (pts.min(0) + pts.max(0)) / 2
    span = float(np.ptp(pts[:, :2], axis=0).max())
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [center[0], center[1], 0.6]
    camera.distance = max(3.0, 1.4 * span + 1.5)
    camera.azimuth = 90.0
    camera.elevation = -25.0

    tiles: List[List[np.ndarray]] = []
    index_lines = []
    for i, plan in enumerate(plans):
        with open(plan["pkl"], "rb") as f:
            data = pickle.load(f)
        qpos = build_qpos_trajectory(data["results"], task, mj_model)
        step_ts = data["results"].get("step_ts")
        if step_ts is not None and len(step_ts):
            dts = np.asarray(step_ts, dtype=float).reshape(-1)
            times = np.concatenate([[0.0], np.cumsum(dts)])[: len(qpos)]
        else:
            times = np.arange(len(qpos)) * 0.05
        dur = min(float(times[-1]), args.max_seconds)
        ids = _frames_at(times, np.arange(0.0, dur + 1e-9, 1.0 / args.fps))
        label = [f"{plan['gen']}  score {plan['score']:.3f}  {dur:.1f}s"]
        frames = render_plan(renderer, mj_model, qpos, ids, camera)
        frames = [_label(fr, label + _short_patches(plan["patches"])) for fr in frames]
        imageio.mimsave(os.path.join(out, f"{plan['gen']}.gif"), frames,
                        duration=1.0 / args.fps, loop=0)

        ov_ids = _frames_at(times, np.linspace(0.0, float(times[-1]), args.overview_frames))
        tframes = render_plan(tile_renderer, mj_model, qpos, ov_ids, camera)
        tiles.append([_label(fr, [f"{plan['gen']} {plan['score']:.3f}"]) for fr in tframes])
        index_lines.append(f"{i:2d}  {plan['gen']:>7}  score {plan['score']:.4f}  {plan['patches']}")
        print(f"[{i + 1}/{len(plans)}] {plan['gen']} score {plan['score']:.4f}", flush=True)

    cols = min(args.cols, len(tiles))
    rows = -(-len(tiles) // cols)
    blank = np.zeros((th, tw, 3), dtype=np.uint8)
    grid = []
    for k in range(args.overview_frames):
        cells = [t[k] for t in tiles] + [blank] * (rows * cols - len(tiles))
        grid.append(np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)]))
    imageio.mimsave(os.path.join(out, "overview.gif"), grid, duration=1.0 / args.fps, loop=0)
    with open(os.path.join(out, "index.txt"), "w") as f:
        f.write("\n".join(index_lines) + "\n")
    renderer.close()
    tile_renderer.close()
    print(f"wrote {len(plans)} per-plan GIFs, overview.gif and index.txt to {out}")


if __name__ == "__main__":
    main()
