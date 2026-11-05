"""Count contact plans per scene under a results tree.

Walks ``<results_root>/<scene>/<run>/gen_*/results/extra.json``, hashes
each ``contact_mode_seq``, and reports per-scene how many DISTINCT plans
exist in the tree vs. how many were "valid" (TO actually ran on them,
converged or not). ``best/`` is intentionally skipped — it is a
byte-duplicate of some ``gen_N`` and would inflate the count.

A "valid" contact plan = TO actually ran on it. The verdict is sourced
from ``.to_recompute_cache.jsonl`` (next to ``--results-root``) when
present, since recompute is the authoritative re-run; otherwise it
falls back to the original ``extra.json``. TO is considered to have
run iff ``to_converged is not None`` in extra.json, or the cache row
has a non-None ``to_cost`` (numeric result) or non-None ``error``
(solve attempted and crashed). Skipped rows (``goal_match_false``,
``ik_infeasible``, etc.) do not count as "TO ran".

Uniqueness is scoped per-scene (across all runs of that scene), using
the same canonical hash as ``evaluate.py:_canonical_plan_hash`` so
list/tuple round-trips collide.

Columns:
  - ``n_runs``         number of run dirs under that scene
  - ``n_gens``         extra.json files discovered
  - ``n_unique_all``   distinct contact_mode_seq hashes (any TO state)
  - ``n_to_ran``       gens where TO actually ran (converged or not)
  - ``n_unique_valid`` distinct hashes among TO-ran gens
  - ``n_families_valid`` distinct strategy families among TO-ran gens
  - ``n_patch_cells_valid`` distinct contact-patch sets (the archive key)
                       (plan_signature.plan_signature: same contacts made
                       and broken in the same order, step counts ignored)
  - ``best_to_cost``   lowest to_cost seen anywhere in the scene's tree
                       (to_cost == 0 excluded as acados restoration
                       noise, same rule evaluate.py's combined_score
                       uses). Prefers the lowest CONVERGED cost; falls
                       back to the lowest of any numeric result if no
                       gen converged, flagged "*" in the stdout table
                       (``best_to_cost_converged`` in the CSV)

Output:
  - stdout summary table (one row per scene + a TOTAL row)
  - ``<results_root>/valid_plan_counts.csv`` (one row per scene)

Usage::

    python examples/loco_manipulation/count_valid_plans.py \\
        --results-root results/loco_abl3_ik_fb

This script is pure I/O — no acados, no pinocchio, no PYTHONPATH gymnastics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from plan_signature import plan_patches, plan_signature
from _run_common import canonical_plan_hash  # noqa: E402

CSV_FILENAME = "valid_plan_counts.csv"
PNG_FILENAME = "valid_plan_counts.png"
CACHE_FILENAME = ".to_recompute_cache.jsonl"


# ---------------------------------------------------------------------------
# Plan hashing — mirrors evaluate.py:_canonical_plan_hash so tuples-loaded-
# as-lists (JSON round-trip) collide with live tuple plans.
# ---------------------------------------------------------------------------


_plan_hash = canonical_plan_hash  # same hash as evaluate.py / seen_plans.json


# ---------------------------------------------------------------------------
# Discovery — same conventions as recompute_to_costs.py
# ---------------------------------------------------------------------------


def _discover_runs(results_root: str) -> List[Tuple[str, str, str]]:
    """Return ``[(scene_dir_name, run_dir_name, run_dir_abs), ...]`` for
    every ``<results_root>/<scene>/<run>/`` that contains ``.hydra/``."""
    out: List[Tuple[str, str, str]] = []
    for scene_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(scene_dir):
            continue
        scene_name = os.path.basename(scene_dir)
        for run_dir in sorted(glob.glob(os.path.join(scene_dir, "*"))):
            if not os.path.isdir(run_dir):
                continue
            if not os.path.isdir(os.path.join(run_dir, ".hydra")):
                continue
            out.append((scene_name, os.path.basename(run_dir), run_dir))
    return out


def _discover_gens(run_dir: str) -> List[str]:
    """Sorted ``gen_N`` dir names with a readable ``results/extra.json``."""
    entries: List[Tuple[int, str]] = []
    for entry in os.listdir(run_dir):
        if not entry.startswith("gen_"):
            continue
        path = os.path.join(run_dir, entry, "results", "extra.json")
        if not os.path.isfile(path):
            continue
        try:
            idx = int(entry.split("_", 1)[1])
        except ValueError:
            continue
        entries.append((idx, entry))
    entries.sort()
    return [name for _, name in entries]


def _load_recompute_cache(
    cache_path: str,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Load ``.to_recompute_cache.jsonl`` keyed by ``(scene, run, gen)``."""
    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not os.path.isfile(cache_path):
        return cache
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("scene"), row.get("run"), row.get("gen"))
            if None in key:
                continue
            cache[key] = row
    return cache


def _plot_counts(
    scenes: List[str],
    gens_per_scene: Dict[str, int],
    unique_valid_per_scene: Dict[str, Set[str]],
    png_path: str,
) -> None:
    """Grouped horizontal bar chart: n_gens vs n_unique_valid per scene.
    One pair of bars per scene, blue = n_gens, orange = n_unique_valid.
    Saved as a PNG next to the CSV by default."""
    if not scenes:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_gens = [gens_per_scene[s] for s in scenes]
    n_valid = [len(unique_valid_per_scene[s]) for s in scenes]
    y_pos = list(range(len(scenes)))
    bar_h = 0.4

    rc = {
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(11, max(4.0, 0.7 * len(scenes) + 1.0)))
        ax.barh(
            [y - bar_h / 2 for y in y_pos], n_gens, bar_h,
            color="tab:blue", label="n_gens",
        )
        ax.barh(
            [y + bar_h / 2 for y in y_pos], n_valid, bar_h,
            color="tab:orange", label="n_unique_valid",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scenes)
        ax.invert_yaxis()  # first scene at top
        ax.set_xlabel("count")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="lower right")

        # Numeric value at the end of each bar.
        for i, g in enumerate(n_gens):
            ax.text(g, i - bar_h / 2, f" {g}", va="center", fontsize=11)
        for i, v in enumerate(n_valid):
            ax.text(v, i + bar_h / 2, f" {v}", va="center", fontsize=11)

        fig.tight_layout()
        os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
        fig.savefig(png_path, dpi=140)
        plt.close(fig)


def _to_ran_in_original(extra: Dict[str, Any]) -> bool:
    """True iff the ORIGINAL shinka eval attempted TO (regardless of
    convergence). Used by the time-to-first-solution metric so the
    timing reflects what the search actually surfaced at run time —
    NOT what the offline recompute later salvaged."""
    return extra.get("to_converged") is not None


def _format_duration(seconds: float) -> str:
    """Compact h/m/s rendering for the stdout table."""
    s = int(round(max(0.0, seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _time_to_first_to_in_run(
    run_abs: str, gens: List[str],
) -> Optional[float]:
    """Seconds from the end of the compile gen to the first post-compile
    gen where the ORIGINAL eval attempted TO. Returns None if the run
    has fewer than 3 gens or never reached TO from gen[2] onward.

    Compile-exclusion logic: gen_0 is the initial baseline program
    eval (often trivial; doesn't trigger IK). The acados solver pool
    is built on the first real IK call, which falls in gen_1. So
    gen_1's wall time is dominated by compile, not search. We anchor
    the timer at the END of gen_1 (mtime of gens[1]'s extra.json) and
    look for the first TO success in gens[2:] only. Subsequent gens
    re-use the cached pool, so their elapsed time measures actual
    LLM iteration + plan-eval cost."""
    if len(gens) < 3:
        return None
    baseline = os.path.join(run_abs, gens[1], "results", "extra.json")
    try:
        t_start = os.path.getmtime(baseline)
    except OSError:
        return None
    for gen in gens[2:]:
        extra_path = os.path.join(run_abs, gen, "results", "extra.json")
        try:
            with open(extra_path) as f:
                extra = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not _to_ran_in_original(extra):
            continue
        try:
            t_end = os.path.getmtime(extra_path)
        except OSError:
            return None
        return max(0.0, t_end - t_start)
    return None


def _to_attempted(
    extra: Dict[str, Any],
    cache_row: Dict[str, Any] | None,
) -> bool:
    """True iff TO actually ran for this gen (converged or not).

    Prefers the recompute cache when it has a definitive row:
      - ``to_cost is not None`` → solve produced a numeric result (ran).
      - ``error is not None``   → solve was attempted and crashed (ran).
      - ``skipped`` set         → solve was bypassed (e.g. ik_infeasible);
                                  fall back to extra.json so we still
                                  catch cases where the original
                                  evaluation did invoke TO.

    For ``extra.json``, TO ran iff ``to_converged is not None`` (the
    field is initialized to ``None`` in evaluate.py and only assigned
    when the TO solve completes — pass or fail).
    """
    if cache_row is not None:
        if cache_row.get("to_cost") is not None:
            return True
        if cache_row.get("error") is not None:
            return True
        # Cache row is a "skipped" record; fall through to extra.json.
    return extra.get("to_converged") is not None


def _to_cost_and_converged(
    extra: Dict[str, Any],
    cache_row: Dict[str, Any] | None,
) -> Tuple[Optional[float], bool]:
    """Best-known ``(to_cost, to_converged)`` for this gen.

    Prefers the recompute cache when it produced a numeric ``to_cost``
    (an authoritative re-solve); otherwise falls back to extra.json's
    own fields — same preference order as ``_to_attempted``.
    """
    if cache_row is not None and cache_row.get("to_cost") is not None:
        return cache_row.get("to_cost"), bool(cache_row.get("to_converged"))
    return extra.get("to_cost"), bool(extra.get("to_converged"))


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
        "--csv-path",
        default=None,
        help=f"Override the CSV destination (default: <results-root>/{CSV_FILENAME}).",
    )
    parser.add_argument(
        "--png-path",
        default=None,
        help=f"Override the PNG destination (default: <results-root>/{PNG_FILENAME}).",
    )
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    if not os.path.isdir(results_root):
        parser.error(f"results-root {results_root!r} is not a directory")

    cache_path = os.path.join(results_root, CACHE_FILENAME)
    cache = _load_recompute_cache(cache_path)
    if cache:
        print(f"[scan] using recompute cache at {cache_path} ({len(cache)} rows)")
    else:
        print(
            f"[scan] no recompute cache at {cache_path}; "
            "using to_converged from extra.json"
        )

    runs = _discover_runs(results_root)
    print(f"[scan] {len(runs)} run(s) found under {results_root}")
    if not runs:
        return

    runs_per_scene: Dict[str, int] = defaultdict(int)
    gens_per_scene: Dict[str, int] = defaultdict(int)
    to_ran_per_scene: Dict[str, int] = defaultdict(int)
    unique_all_per_scene: Dict[str, Set[str]] = defaultdict(set)
    unique_valid_per_scene: Dict[str, Set[str]] = defaultdict(set)
    families_valid_per_scene: Dict[str, Set[str]] = defaultdict(set)
    # Distinct contact-patch sets (the archive's descriptor key).
    cells_valid_per_scene: Dict[str, Set[str]] = defaultdict(set)
    # Per-scene fastest "time to first TO" across runs (seconds, compile excluded).
    time_to_first_to_per_scene: Dict[str, Optional[float]] = defaultdict(lambda: None)
    # Per-scene best (lowest) TO cost across the whole tree search, tracked
    # separately for converged and any-numeric-result solves so converged
    # results are preferred (mirrors recompute_to_costs.py's own "best"
    # selection: converged pool wins if non-empty, else any success).
    # to_cost == 0 is excluded everywhere as untrustworthy acados
    # restoration noise (same rule evaluate.py's combined_score uses).
    best_converged_cost_per_scene: Dict[str, Optional[float]] = defaultdict(lambda: None)
    best_any_cost_per_scene: Dict[str, Optional[float]] = defaultdict(lambda: None)

    for scene, run, run_abs in runs:
        runs_per_scene[scene] += 1
        gens = _discover_gens(run_abs)
        gens_per_scene[scene] += len(gens)

        elapsed = _time_to_first_to_in_run(run_abs, gens)
        if elapsed is not None:
            best = time_to_first_to_per_scene.get(scene)
            if best is None or elapsed < best:
                time_to_first_to_per_scene[scene] = elapsed

        for gen in gens:
            extra_path = os.path.join(run_abs, gen, "results", "extra.json")
            try:
                with open(extra_path) as f:
                    extra = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"[warn] {extra_path}: {e}")
                continue

            plan = extra.get("contact_mode_seq")
            if not isinstance(plan, dict) or not plan:
                continue

            plan_hash = _plan_hash(plan)
            unique_all_per_scene[scene].add(plan_hash)

            cache_row = cache.get((scene, run, gen))

            cost, converged = _to_cost_and_converged(extra, cache_row)
            if cost is not None and cost > 0.0:
                cur_any = best_any_cost_per_scene[scene]
                if cur_any is None or cost < cur_any:
                    best_any_cost_per_scene[scene] = cost
                if converged:
                    cur_conv = best_converged_cost_per_scene[scene]
                    if cur_conv is None or cost < cur_conv:
                        best_converged_cost_per_scene[scene] = cost

            if not _to_attempted(extra, cache_row):
                continue

            to_ran_per_scene[scene] += 1
            unique_valid_per_scene[scene].add(plan_hash)
            families_valid_per_scene[scene].add(plan_signature(plan))
            cells_valid_per_scene[scene].add(plan_patches(plan))

    # Best cost per scene: converged pool wins whenever it's non-empty;
    # "*" marks a best that comes from a non-converged solve so the
    # reader knows it's the weaker fallback.
    best_to_cost_per_scene: Dict[str, Optional[float]] = {}
    best_to_cost_converged_per_scene: Dict[str, bool] = {}
    for scene in set(best_any_cost_per_scene) | set(best_converged_cost_per_scene):
        conv_cost = best_converged_cost_per_scene[scene]
        if conv_cost is not None:
            best_to_cost_per_scene[scene] = conv_cost
            best_to_cost_converged_per_scene[scene] = True
        else:
            best_to_cost_per_scene[scene] = best_any_cost_per_scene[scene]
            best_to_cost_converged_per_scene[scene] = False

    def _format_best_cost(scene: str) -> str:
        cost = best_to_cost_per_scene.get(scene)
        if cost is None:
            return "—"
        marker = "" if best_to_cost_converged_per_scene.get(scene) else "*"
        return f"{cost:.4f}{marker}"

    scenes = sorted(
        set(runs_per_scene) | set(to_ran_per_scene) | set(unique_all_per_scene)
    )

    print()
    header_fmt = "{:<36s}  {:>8s}  {:>14s}  {:>10s}  {:>8s}  {:>14s}  {:>14s}"
    row_fmt    = "{:<36s}  {:>8d}  {:>14d}  {:>10d}  {:>8d}  {:>14s}  {:>14s}"
    print(header_fmt.format(
        "scene", "n_gens", "n_unique_valid", "n_families", "n_cells", "t_first_TO",
        "best_TO_cost",
    ))
    print("-" * 108)
    for scene in scenes:
        elapsed = time_to_first_to_per_scene.get(scene)
        elapsed_str = _format_duration(elapsed) if elapsed is not None else "—"
        print(row_fmt.format(
            scene,
            gens_per_scene[scene],
            len(unique_valid_per_scene[scene]),
            len(families_valid_per_scene[scene]),
            len(cells_valid_per_scene[scene]),
            elapsed_str,
            _format_best_cost(scene),
        ))
    print("-" * 108)
    print(row_fmt.format(
        "TOTAL",
        sum(gens_per_scene.values()),
        sum(len(s) for s in unique_valid_per_scene.values()),
        sum(len(s) for s in families_valid_per_scene.values()),
        sum(len(s) for s in cells_valid_per_scene.values()),
        "",
        "",
    ))
    print("\n* = best TO cost did not converge (fallback from the converged pool)")

    csv_path = args.csv_path or os.path.join(results_root, CSV_FILENAME)
    fieldnames = [
        "scene", "n_gens_evaluated", "n_unique_valid", "n_families_valid",
        "n_patch_cells_valid", "time_to_first_to_s", "best_to_cost",
        "best_to_cost_converged",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in scenes:
            elapsed = time_to_first_to_per_scene.get(scene)
            best_cost = best_to_cost_per_scene.get(scene)
            writer.writerow({
                "scene": scene,
                "n_gens_evaluated": gens_per_scene[scene],
                "n_unique_valid": len(unique_valid_per_scene[scene]),
                "n_families_valid": len(families_valid_per_scene[scene]),
                "n_patch_cells_valid": len(cells_valid_per_scene[scene]),
                "time_to_first_to_s": (
                    f"{elapsed:.1f}" if elapsed is not None else ""
                ),
                "best_to_cost": (
                    f"{best_cost:.6f}" if best_cost is not None else ""
                ),
                "best_to_cost_converged": (
                    best_to_cost_converged_per_scene.get(scene, "")
                    if best_cost is not None else ""
                ),
            })
    print(f"\n[csv ] wrote {len(scenes)} rows to {csv_path}")

    png_path = args.png_path or os.path.join(results_root, PNG_FILENAME)
    _plot_counts(scenes, gens_per_scene, unique_valid_per_scene, png_path)
    print(f"[png ] wrote {png_path}")


if __name__ == "__main__":
    main()
