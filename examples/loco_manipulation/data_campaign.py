"""
Data campaign: searches over scenes x layouts, then physics-varied TO
re-solves of every plan they found, exported as an RL-tracker dataset.

    python examples/loco_manipulation/data_campaign.py \\
        examples/loco_manipulation/campaigns/example.yaml [--dry-run]

The campaign YAML::

    name: example
    out: datasets/example                  # dataset root (repo-relative ok)
    search:                                # run_search.sh knobs
      diverse: true
      score_mode: quality
    dataset:                               # generate_dataset.py options
      samples: 8
      near_feasible: 0.1
      max_per_family: 3
      jobs: 1
      factors: [mass, size, robot, objects, timing, weights]
      ranges: {mass: [0.5, 2.0]}           # any generate_dataset.Ranges field
    scenes:
      - scene: truck_unload_stack
        description: "..."                 # optional (else the script's table)
        generations: 60                    # optional (else the script's table)
        layouts: [default, default@near]   # IC names; <ic>@<layout> -> layouts.py

For every (scene, IC) it runs one search through ``run_search.sh``
(one at a time), which skips a search whose results dir already
exists, then ``generate_dataset.py`` on that run, which skips samples
already in the manifest. Rerunning the campaign therefore resumes it.
``--skip-search`` / ``--skip-dataset`` run one half only.

Nothing here is light: each search is hours and each sample a TO solve.
Keep ``SHINKA_TO_N_JOB=1`` (the default the script exports) and run the
campaign under ``nice``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
SEARCH_SCRIPT = os.path.join(_THIS_DIR, "run_search.sh")
GENERATOR = os.path.join(_THIS_DIR, "generate_dataset.py")


def load_campaign(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not cfg.get("scenes"):
        raise SystemExit(f"{path}: no scenes")
    for entry in cfg["scenes"]:
        if "scene" not in entry:
            raise SystemExit(f"{path}: scene entry without 'scene': {entry}")
        entry.setdefault("layouts", ["default"])
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg.setdefault("out", os.path.join("datasets", cfg["name"]))
    cfg.setdefault("search", {})
    cfg.setdefault("dataset", {})
    return cfg


def exp_dir(scene: str, ic: str, search: Dict[str, Any]) -> str:
    """results/<exp>/<scene>[/<ic>], as run_search.sh names it."""
    suffix = ""
    if search.get("diverse", True):
        suffix += "_diverse"
    mode = search.get("score_mode", "quality")
    if mode != "legacy":
        suffix += f"_{mode}"
    path = os.path.join(REPO_ROOT, "results", f"loco_abl3_ik_fb{suffix}", scene)
    return path if ic == "default" else os.path.join(path, ic)


def latest_run(path: str) -> Optional[str]:
    if not os.path.isdir(path):
        return None
    runs = sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d, ".hydra")))
    return os.path.join(path, runs[-1]) if runs else None


def search_env(entry: Dict[str, Any], ic: str, search: Dict[str, Any]) -> Dict[str, str]:
    env = {
        **os.environ,
        "SCENES": entry["scene"],
        "ICS": ic,
        "SCORE_MODE": str(search.get("score_mode", "quality")),
        "SHINKA_TO_N_JOB": os.environ.get("SHINKA_TO_N_JOB", "1"),
    }
    if search.get("diverse", True):
        env["DIVERSE"] = "1"
    else:
        env.pop("DIVERSE", None)
    if entry.get("description"):
        env["DESCRIPTION"] = str(entry["description"])
    if entry.get("generations"):
        env["MAX_NODES"] = str(int(entry["generations"]))
    return env


def dataset_args(run_dir: str, cfg: Dict[str, Any]) -> List[str]:
    ds = cfg["dataset"]
    out = cfg["out"] if os.path.isabs(cfg["out"]) else os.path.join(REPO_ROOT, cfg["out"])
    args = [sys.executable, GENERATOR, "--run-dir", run_dir, "--out", out]
    for key, flag in (("samples", "--samples"), ("near_feasible", "--near-feasible"),
                      ("max_per_family", "--max-per-family"), ("jobs", "--jobs"),
                      ("seed", "--seed"), ("frequency", "--frequency")):
        if key in ds:
            args += [flag, str(ds[key])]
    if ds.get("factors"):
        args += ["--factors", ",".join(ds["factors"])]
    for name, (lo, hi) in (ds.get("ranges") or {}).items():
        args += ["--range", f"{name}={lo},{hi}"]
    return args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-search", action="store_true")
    p.add_argument("--skip-dataset", action="store_true")
    args = p.parse_args()
    cfg = load_campaign(args.campaign)

    matrix = [(entry, ic) for entry in cfg["scenes"] for ic in entry["layouts"]]
    print(f"[campaign] {cfg['name']}: {len(matrix)} search(es) -> {cfg['out']}")
    for entry, ic in matrix:
        run = latest_run(exp_dir(entry["scene"], ic, cfg["search"]))
        print(f"  {entry['scene']:<36} {ic:<24} "
              f"{'existing run ' + os.path.basename(run) if run else 'new search'}"
              f" (generations: {entry.get('generations', 'script default')})")
    if args.dry_run:
        print("[dry-run] dataset: " + " ".join(dataset_args("<run>", cfg)[2:]))
        return

    for entry, ic in matrix:
        if not args.skip_search:
            print(f"[search] {entry['scene']} {ic}", flush=True)
            subprocess.run(["bash", SEARCH_SCRIPT], cwd=REPO_ROOT,
                           env=search_env(entry, ic, cfg["search"]), check=False)
        run = latest_run(exp_dir(entry["scene"], ic, cfg["search"]))
        if run is None:
            print(f"[warn] no run dir for {entry['scene']} {ic}; dataset skipped")
            continue
        if not args.skip_dataset:
            print(f"[dataset] {run}", flush=True)
            subprocess.run(dataset_args(run, cfg), cwd=REPO_ROOT, check=False)


if __name__ == "__main__":
    main()
