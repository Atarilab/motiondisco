"""
Shared helpers for ``replay_ik_run.py`` and ``replay_to_run.py``.

Both KSO and TO solvers return the same ``robot_xs`` / ``obj_xs``
layout in ``get_results()`` (``robot_xs`` keyed by robot instance ID,
``obj_xs`` keyed by object name), so the qpos packer is identical
across replay scripts. Keeping it in one place avoids divergence as
scene layouts grow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def build_replay_model(task):
    """Compile the MuJoCo model for *task*'s scene at its resolved state.

    faro scenes no longer carry a compiled ``mj_model``; it is assembled
    on demand from the scene plus an explicit resolved state.
    """
    from faro.configs.mujoco import build_mujoco

    mj_model, _qpos0 = build_mujoco(task.scene, task.resolved_state)
    return mj_model


def build_qpos_trajectory(results: dict, task, mj_model) -> np.ndarray:
    """
    Pack ``robot_xs`` and ``obj_xs`` into a (T, mj_model.nq) qpos
    trajectory ready to feed into ``MjData.qpos``, using faro's
    ``pack_trajectory`` so compiled MuJoCo joint addresses (and the
    xyzw -> wxyz quaternion swap) are respected.

    Pickles written before the faro port stored ``robot_xs`` as a bare
    (T, nq) array; those are assigned to the scene's sole robot.
    """
    from faro.configs.mujoco import pack_trajectory

    robot_xs = results["robot_xs"]
    if not isinstance(robot_xs, dict):
        robot_xs = {task.robot_id: np.asarray(robot_xs)}
    qpos, _qvel = pack_trajectory(task.scene, mj_model, robot_xs, results["obj_xs"])
    return qpos


def write_trajectory_mp4(
    mj_model,
    qpos_traj: np.ndarray,
    out_path: Path,
    dt: float,
    width: int = 1280,
    height: int = 720,
    camera: Optional[str] = None,
    step_ts: Optional[np.ndarray] = None,
    fps: int = 30,
) -> None:
    """
    Render ``qpos_traj`` offscreen with ``mujoco.Renderer`` and write
    an mp4 at ``out_path``. Imports mujoco / imageio lazily so the
    shared helper module stays cheap to import from callers that only
    need ``build_qpos_trajectory``.

    Two timing modes:
      - ``step_ts`` provided: one duration per qpos frame. The mp4 is
        written at constant ``fps``, and each rendered frame is
        replicated ``round(step_ts[k] * fps)`` times. This preserves
        the TO solver's optimized per-node timing (e.g. 10× faster
        steps at contact-switch boundaries).
      - ``step_ts`` is None: uniform timing — one frame per qpos sample
        at ``fps = round(1/dt)``. Used by IK replays and as the
        backward-compatible fallback for old pickles.
    """
    import mujoco
    import imageio.v2 as imageio

    md = mujoco.MjData(mj_model)

    # Clamp to the model's offscreen framebuffer size so Renderer()
    # doesn't blow up with "offwidth too small". The user can raise
    # the ceiling via the robot MJCF's <visual><global offwidth="..."
    # offheight="..."/></visual> (build_mujoco copies the robot spec).
    off_w = int(mj_model.vis.global_.offwidth)
    off_h = int(mj_model.vis.global_.offheight)
    if width > off_w or height > off_h:
        print(
            f"[replay] requested video size {width}x{height} exceeds model "
            f"offscreen buffer {off_w}x{off_h}; clamping. "
            f"Bump <visual><global offwidth=\"...\" offheight=\"...\"/></visual> "
            f"in the scene XML for higher-res output."
        )
        width = min(width, off_w)
        height = min(height, off_h)

    camera_arg = {} if camera is None else {"camera": camera}
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    try:
        rendered = []
        T = qpos_traj.shape[0]
        for k in range(T):
            md.qpos[:] = qpos_traj[k]
            mujoco.mj_fwdPosition(mj_model, md)
            renderer.update_scene(md, **camera_arg)
            rendered.append(renderer.render())
    finally:
        renderer.close()

    if step_ts is not None:
        # Variable-time playback: replicate each rendered frame proportional
        # to its node's true duration so the constant-fps mp4 plays back
        # the optimized motion timing exactly.
        durations = np.asarray(step_ts, dtype=float).reshape(-1)
        out_fps = max(1, int(fps))
        frames = []
        for k in range(T):
            # Pad the trailing frame with the previous duration when
            # step_ts is one shorter than qpos (typical N vs N+1 layout).
            d = float(durations[k]) if k < durations.size else float(durations[-1])
            n_repeat = max(1, int(round(d * out_fps)))
            frames.extend([rendered[k]] * n_repeat)
        encoded_fps = out_fps
    else:
        # Uniform-dt fallback (IK replays, old TO pickles).
        frames = rendered
        encoded_fps = max(1, int(round(1.0 / max(dt, 1e-3))))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 avoids imageio's auto-padding warning for
    # non-multiple-of-16 dimensions; fps= drives the ffmpeg writer.
    imageio.mimsave(str(out_path), frames, fps=encoded_fps, macro_block_size=1)
    print(
        f"video saved: {out_path} ({len(frames)} frames "
        f"from {T} qpos samples, fps={encoded_fps}, {width}x{height})"
    )


# ---------------------------------------------------------------------------
# Interactive replay (pause / scrub / speed controls in mujoco passive viewer)
# ---------------------------------------------------------------------------

# GLFW keycodes (mujoco passes raw GLFW codes to key_callback).
_GLFW_KEY_SPACE = 32
_GLFW_KEY_LBRACKET = 91
_GLFW_KEY_RBRACKET = 93
_GLFW_KEY_L = 76
_GLFW_KEY_RIGHT = 262
_GLFW_KEY_LEFT = 263
_GLFW_KEY_HOME = 268
_GLFW_KEY_END = 269

_INTERACTIVE_HELP = (
    "[replay] interactive controls:\n"
    "  SPACE       pause / resume\n"
    "  LEFT/RIGHT  step -1 / +1 frame\n"
    "  HOME / END  jump to first / last frame\n"
    "  [ / ]       halve / double playback speed\n"
    "  L           toggle loop on / off\n"
    "  ESC         quit (mujoco default)"
)


@dataclass
class _PlaybackState:
    frame_idx: int = 0
    paused: bool = False
    speed: float = 1.0
    loop: bool = True
    step_pending: int = 0
    accumulator: float = 0.0


def interactive_replay(
    mj_model,
    qpos_traj: np.ndarray,
    step_ts: Optional[np.ndarray],
    dt: float,
    loop: bool,
) -> None:
    """Drive a mujoco passive viewer with pause / scrub / speed controls.

    Timing reuses the same per-step durations as the non-interactive
    loop: ``step_ts[k]`` when present (optimizer-aware variable timing),
    otherwise the uniform ``dt`` fallback. A real-wall-clock accumulator
    advances the frame so playback speed stays correct even if a render
    takes longer than the nominal tick.

    When ``loop`` is False and playback reaches the end, the helper
    pauses on the last frame instead of closing — keeping the viewer
    open so the user can scrub back with the arrow keys. ESC closes.
    """
    import mujoco
    import mujoco.viewer

    md = mujoco.MjData(mj_model)
    T = int(qpos_traj.shape[0])
    if T == 0:
        print("[replay] qpos trajectory is empty; nothing to display.")
        return

    step_ts_arr: Optional[np.ndarray] = None
    if step_ts is not None:
        step_ts_arr = np.asarray(step_ts, dtype=float).reshape(-1)

    state = _PlaybackState(loop=loop)

    def step_duration(idx: int) -> float:
        if step_ts_arr is not None and step_ts_arr.size > 0:
            i = idx if idx < step_ts_arr.size else step_ts_arr.size - 1
            return max(float(step_ts_arr[i]), 1e-3)
        return max(float(dt), 1e-3)

    def key_callback(keycode: int) -> None:
        if keycode == _GLFW_KEY_SPACE:
            state.paused = not state.paused
        elif keycode == _GLFW_KEY_RIGHT:
            state.step_pending += 1
        elif keycode == _GLFW_KEY_LEFT:
            state.step_pending -= 1
        elif keycode == _GLFW_KEY_HOME:
            state.frame_idx = 0
            state.accumulator = 0.0
        elif keycode == _GLFW_KEY_END:
            state.frame_idx = T - 1
            state.accumulator = 0.0
        elif keycode == _GLFW_KEY_RBRACKET:
            state.speed = min(state.speed * 2.0, 16.0)
            print(f"[replay] speed = {state.speed:g}x")
        elif keycode == _GLFW_KEY_LBRACKET:
            state.speed = max(state.speed * 0.5, 0.0625)
            print(f"[replay] speed = {state.speed:g}x")
        elif keycode == _GLFW_KEY_L:
            state.loop = not state.loop
            print(f"[replay] loop = {state.loop}")

    print(_INTERACTIVE_HELP)

    tick = 1.0 / 60.0
    with mujoco.viewer.launch_passive(
        mj_model, md, key_callback=key_callback,
    ) as v:
        v.opt.sitegroup[0] = 0
        last_t = time.monotonic()
        while v.is_running():
            now = time.monotonic()
            elapsed = now - last_t
            last_t = now

            if state.step_pending != 0:
                direction = 1 if state.step_pending > 0 else -1
                n_steps = abs(state.step_pending)
                state.frame_idx = max(
                    0, min(T - 1, state.frame_idx + direction * n_steps),
                )
                state.step_pending = 0
                state.accumulator = 0.0
            elif not state.paused:
                state.accumulator += elapsed * state.speed
                while state.accumulator >= step_duration(state.frame_idx):
                    state.accumulator -= step_duration(state.frame_idx)
                    state.frame_idx += 1
                    if state.frame_idx >= T:
                        if state.loop:
                            state.frame_idx = 0
                        else:
                            state.frame_idx = T - 1
                            state.paused = True
                            state.accumulator = 0.0
                            break

            md.qpos[:] = qpos_traj[state.frame_idx]
            mujoco.mj_fwdPosition(mj_model, md)
            v.sync()
            time.sleep(tick)
