"""
Categorize a run's valid plans by how they use contacts and save one
labeled MP4 per category (its best plan) as a visual summary.

    python examples/loco_manipulation/render_categories.py \\
        --run-dir results/<exp>/<scene>/<run>

Categories are read from each plan's contact patches (``plan_patches``)
and assigned by priority, so every plan lands in exactly one:

    push_drag       hands on the object's back face, or the object
                    sliding on its starting surface, before the carry
    env_braced      a hand pressed on a fixed surface (table, wall, ...)
    slide_place     the object slides on its final support surface
    top_assisted    a hand on the object's top face
    sliding_hands   hands slide along the object's faces
    plain_carry     sticking grasp on opposite faces, lift, carry, place

Writes ``<run>/category_videos/<rank>_<category>_<gen>.mp4`` plus
``categories.txt`` (every valid plan with its category).

``--gen gen_N --title "..." [--name file_stem]`` instead renders just
that one plan into the same folder, e.g. to add a hand-picked example. Valid = TO
converged with to_cost > 0; identical plans kept once. Single-object
scenes are assumed (the first movable object is "the object").
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from plan_signature import patch_set  # noqa: E402
from render_plans_gif import _frames_at, _label, collect_valid_plans, render_plan  # noqa: E402
from run_to_all import _read_hydra_task  # noqa: E402

CATEGORIES = [
    ("push_drag", "Push / drag, then carry"),
    ("env_braced", "Hand braced on the environment"),
    ("slide_place", "Carry, then slide into place"),
    ("top_assisted", "Grasp assisted by the box top"),
    ("sliding_hands", "Carry with hands sliding on the box"),
    ("plain_carry", "Plain two-hand carry"),
]


def categorize(patches: str, task: Any) -> str:
    """Category name for a plan's contact patches (see module docstring)."""
    obj = next(iter(task.scene.env_objects))
    faces = {fid: (owner, face) for owner, fl in _faces(task).items() for fid, _sid, face in fl}
    obj_ids = {fid for fid, (owner, _f) in faces.items() if owner == obj}
    start = task.resolved_state.contact_mode
    start_env = next(v[1] for k, v in start.items() if k[-1] == obj and v)
    hand_contacts, obj_contacts = [], []
    for token in patch_set(patches):
        ch, _, c = token.partition(":")
        ids = [int(x) for x in c.split(".")]
        if ch.endswith("_ee"):
            hand_contacts.append(ids)
        elif ch == obj:
            obj_contacts.append(ids)
    hand_faces = {faces.get(env, ("?", "?")) for env, _t in hand_contacts}
    if any(o == obj and f == "back" for o, f in hand_faces) or any(
        env == start_env and t == 2 for _s, env, t in obj_contacts
    ):
        return "push_drag"
    if any(env not in obj_ids for env, _t in hand_contacts):
        return "env_braced"
    if any(env != start_env and t == 2 for _s, env, t in obj_contacts):
        return "slide_place"
    if any(o == obj and f == "top" for o, f in hand_faces):
        return "top_assisted"
    if any(t == 2 for _env, t in hand_contacts):
        return "sliding_hands"
    return "plain_carry"


def _faces(task: Any) -> Dict[str, List]:
    from _task_loader import scene_faces

    return scene_faces(task.scene)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", help="Output dir (default: <run-dir>/category_videos)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--size", default="960x720")
    p.add_argument("--gen", help="render only this plan (e.g. gen_20)")
    p.add_argument("--title", default="", help="label for --gen")
    p.add_argument("--name", help="output file stem for --gen")
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
    out = os.path.abspath(args.out or os.path.join(run_dir, "category_videos"))
    os.makedirs(out, exist_ok=True)

    plans = collect_valid_plans(run_dir)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for plan in plans:
        plan["category"] = categorize(plan["patches"], task)
        by_cat.setdefault(plan["category"], []).append(plan)

    lines = []
    for cat, _title in CATEGORIES:
        for plan in by_cat.get(cat, []):
            lines.append(f"{cat:14s} {plan['gen']:>7} score {plan['score']:.4f}  {plan['patches']}")
    with open(os.path.join(out, "categories.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(plans)} valid plans: " + ", ".join(
        f"{cat}={len(by_cat.get(cat, []))}" for cat, _ in CATEGORIES))

    w, h = (int(x) for x in args.size.split("x"))
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, w)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(mj_model, height=h, width=w)
    camera = mujoco.MjvCamera()
    pts = np.array([st.q[:3] for st in (*task.resolved_state.env_objects.values(),
                                         *task.resolved_state.robots.values())]
                   + [st.q[:3] for st in task.resolved_state.fixed_env_objects.values()
                      if np.linalg.norm(st.q[:2]) > 1e-6])
    center = (pts.min(0) + pts.max(0)) / 2
    camera.lookat[:] = [center[0], center[1], 0.6]
    camera.distance = max(2.5, 1.1 * float(np.ptp(pts[:, :2], axis=0).max()) + 1.2)
    camera.azimuth, camera.elevation = 90.0, -20.0

    def render_video(plan: Dict[str, Any], title: str, path: str) -> None:
        with open(plan["pkl"], "rb") as f:
            data = pickle.load(f)
        qpos = build_qpos_trajectory(data["results"], task, mj_model)
        step_ts = data["results"].get("step_ts")
        dts = np.asarray(step_ts if step_ts is not None else [], dtype=float).reshape(-1)
        times = (np.concatenate([[0.0], np.cumsum(dts)])[: len(qpos)]
                 if dts.size else np.arange(len(qpos)) * 0.05)
        ids = _frames_at(times, np.arange(0.0, float(times[-1]) + 1e-9, 1.0 / args.fps))
        hands = [part for part in plan["patches"].split("|")
                 if not part.startswith(("left_sole", "right_sole"))]
        label = [title, f"{plan['gen']}  score {plan['score']:.3f}  {times[-1]:.1f}s", *hands]
        frames = [_label(fr, label) for fr in render_plan(renderer, mj_model, qpos, ids, camera)]
        imageio.mimsave(path, frames, fps=args.fps, macro_block_size=1, quality=8)
        print(path, flush=True)

    if args.gen:
        plan = next((pl for pl in plans if pl["gen"] == args.gen), None)
        if plan is None:
            sys.exit(f"{args.gen} is not a valid plan of this run")
        stem = args.name or f"{plan['category']}_{plan['gen']}"
        render_video(plan, args.title or plan["category"], os.path.join(out, f"{stem}.mp4"))
        renderer.close()
        return

    rank = 0
    for cat, title in CATEGORIES:
        members = by_cat.get(cat)
        if not members:
            continue
        rank += 1
        plan = members[0]  # collect_valid_plans sorts best first
        render_video(plan, f"{rank}. {title}  ({len(members)} plans)",
                     os.path.join(out, f"{rank}_{cat}_{plan['gen']}.mp4"))
    renderer.close()


if __name__ == "__main__":
    main()
