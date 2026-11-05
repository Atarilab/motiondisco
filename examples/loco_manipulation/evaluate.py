"""
Evaluator for loco-manipulation contact-mode-sequence planning.

`combined_score` is a sharp gate with TO cost as the only continuous
signal:

    combined_score = 1 / (1 + to_cost)   if TO ran and to_cost > 0
                   = 0                    otherwise

Stages run in order and each must pass for the next to run: discrete
contact goal on the final mode -> KSO feasibility ("IK" in metric names,
faro's kinematic sequence optimization) -> TO. Duplicate plans
and plans whose expanded length exceeds MAX_NUM_NODES short-circuit to 0
before KSO / TO run.

Structural rules below are diagnostics only — reported in
public / private metrics for the LLM, never fed into the score.

Structural rules (each a fraction in [0, 1]):

    well_formed         per channel-entry: tuple shape and ID membership
    equal_length        sequence-level: every channel list has same length
    at_least_one_sole   per mode: at least one sole is in contact
    no_noop_mode        per transition: consecutive modes differ
    object_passive      per transition: an object cannot make a *new*
                        env contact unless it was () in the previous
                        mode (had been lifted first)
"""

import os
import sys
import json
import pickle
import argparse
import contextlib
import fcntl
import re
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval
from shinka.core.wrap_eval import save_json_results

# Make `from env import ...` resolve whether this file is run as a script
# (`python evaluate.py ...` from the example directory) or imported as a
# module (`examples.loco_manipulation.evaluate` via the Hydra `_target_`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import (
    SceneObject,
    all_surface_ids,
    object_surface_ids,
    movable_object_labels,
)
from _task_loader import CONTACT_TYPES, load_task
from _run_common import (  # noqa: E402
    NONE_STRS, TASK_ENV_VARS, canonical_plan_hash, redirect_fds_to,
)
from plan_signature import (
    DIVERSITY_ENV,
    combined_novelty,
    family_stats,
    patch_novelty,
    plan_novelty,
    plan_patches,
    plan_signature,
    shared_patch_tokens,
)
from scoring import (
    legacy_score,
    quality_score,
    score_mode as active_score_mode,
    task_quality,
    to_quality,
)
from to_eval import incremental_plan, solve_to, to_mode as to_eval_mode
from violation_feedback import (  # noqa: E402
    NameResolver, Violation, kso_feedback, score_feedback, summarize, to_status_text,
)
from plan_checks import event_modes, schedule_errors  # noqa: E402
from scoring import quality_parts  # noqa: E402

MAX_MODES = 12   # efficiency normalizer
# Hard cap on the EXPANDED contact-mode-sequence length (i.e. after
# walk() / append_mode helpers have produced their mode list). Mirrors
# the KSO wrapper's MAX_NUM_NODES (ik_eval.py); plans above this
# horizon cannot be scored by the pre-built solver pool, so the
# evaluator short-circuits before IK / TO are even invoked.
MAX_NUM_NODES = int(os.environ.get("SHINKA_IK_MAX_NUM_NODES", "50"))

SOLE_CHANNELS = ("left_sole", "right_sole")
# 1 = sticking, 2 = sliding (admitted on every face; _task_loader.with_sliding)
VALID_CONTACT_TYPES = CONTACT_TYPES


# ---------------------------------------------------------------------------
# Per-entry well-formedness
# ---------------------------------------------------------------------------

def _check_entry_well_formed(
    channel: str,
    entry: Any,
    is_object_channel: bool,
    valid_env_ids: set,
    self_ids_for_channel: Optional[set],
) -> Tuple[bool, str]:
    if entry == ():
        return True, ""
    if not isinstance(entry, tuple):
        return False, f"{channel}: entry must be a tuple, got {type(entry).__name__}"
    if is_object_channel:
        if len(entry) != 3:
            return False, f"{channel}: object entry must be (self_id, env_id, contact_type), got {entry}"
        self_id, env_id, ctype = entry
        if not isinstance(self_id, int) or self_ids_for_channel is None or self_id not in self_ids_for_channel:
            return False, f"{channel}: self_id {self_id!r} not in object's surfaces {sorted(self_ids_for_channel or [])}"
        if not isinstance(env_id, int) or env_id not in valid_env_ids:
            return False, f"{channel}: env_id {env_id!r} not a known surface id"
        if ctype not in VALID_CONTACT_TYPES:
            return False, f"{channel}: contact_type {ctype!r} not in {VALID_CONTACT_TYPES}"
        return True, ""
    # EE / sole channel
    if len(entry) != 2:
        return False, f"{channel}: ee/sole entry must be (env_id, contact_type), got {entry}"
    env_id, ctype = entry
    if not isinstance(env_id, int) or env_id not in valid_env_ids:
        return False, f"{channel}: env_id {env_id!r} not a known surface id"
    if ctype not in VALID_CONTACT_TYPES:
        return False, f"{channel}: contact_type {ctype!r} not in {VALID_CONTACT_TYPES}"
    return True, ""


# ---------------------------------------------------------------------------
# Plan validation (structural only)
# ---------------------------------------------------------------------------

RunOutput = Tuple[Dict[str, List[Tuple]], List[SceneObject]]


def validate_plan(run_output: RunOutput) -> Tuple[bool, Optional[str]]:
    """Minimal structural check — every deeper rule lives in scoring."""
    if not isinstance(run_output, tuple) or len(run_output) != 2:
        return False, (
            f"run_output must be a 2-tuple (contact_mode_seq, scene), "
            f"got {type(run_output).__name__}"
        )
    plan, scene = run_output
    if not isinstance(plan, dict):
        return False, f"contact_mode_seq must be a dict, got {type(plan).__name__}"
    return True, f"Valid. {len(plan)} channels."


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_contact_plan(
    plan: Dict[str, List[Tuple]],
    scene: List[SceneObject],
) -> Dict[str, Any]:
    """Compute the five structural fractions + efficiency for a plan."""

    valid_env_ids = all_surface_ids(scene)
    obj_self_ids = object_surface_ids(scene)
    movable = set(movable_object_labels(scene))

    # Empty plan: zero structure credit, full efficiency, sum = 1.0.
    if not plan:
        return {
            "num_modes": 0,
            "num_channels": 0,
            "well_formed_fraction": 0.0,
            "equal_length_bool": 0.0,
            "at_least_one_sole_fraction": 0.0,
            "no_noop_fraction": 0.0,
            "object_passive_fraction": 0.0,
            "structure_score": 0.0,
            "efficiency": 1.0,
            "combined_score": 1.0,
            "violations": ["plan is empty"],
        }

    violations: List[str] = []

    # ---- equal_length ------------------------------------------------------
    lengths = {ch: len(seq) if isinstance(seq, list) else -1 for ch, seq in plan.items()}
    bad_length = [ch for ch, L in lengths.items() if L < 0]
    for ch in bad_length:
        violations.append(f"{ch}: value must be a list, got {type(plan[ch]).__name__}")
    valid_lengths = [L for L in lengths.values() if L >= 0]
    if not valid_lengths:
        return {
            "num_modes": 0,
            "num_channels": len(plan),
            "well_formed_fraction": 0.0,
            "equal_length_bool": 0.0,
            "at_least_one_sole_fraction": 0.0,
            "no_noop_fraction": 0.0,
            "object_passive_fraction": 0.0,
            "structure_score": 0.0,
            "efficiency": 0.0,
            "combined_score": 0.0,
            "violations": violations or ["no usable channels"],
        }
    num_modes = max(valid_lengths)
    equal_length_bool = 1.0 if all(L == num_modes for L in valid_lengths) else 0.0
    if equal_length_bool == 0.0:
        violations.append(
            f"channels have unequal lengths: {dict(lengths)}"
        )

    # ---- well_formed -------------------------------------------------------
    well_formed_total = 0
    well_formed_pass = 0
    for ch, seq in plan.items():
        if not isinstance(seq, list):
            continue
        is_obj = ch in movable
        self_ids = obj_self_ids.get(ch) if is_obj else None
        for entry in seq:
            well_formed_total += 1
            ok, err = _check_entry_well_formed(
                ch, entry, is_obj, valid_env_ids, self_ids
            )
            if ok:
                well_formed_pass += 1
            else:
                if len(violations) < 30:
                    violations.append(err)
    well_formed_fraction = (
        well_formed_pass / well_formed_total if well_formed_total else 0.0
    )

    # Build per-mode snapshots over the channels we have, padding short
    # ones with () so per-mode rules can still be evaluated.
    def channel_at(ch: str, k: int) -> Any:
        seq = plan.get(ch)
        if not isinstance(seq, list) or k >= len(seq):
            return ()
        return seq[k]

    # ---- at_least_one_sole (per mode) -------------------------------------
    sole_pass = 0
    for k in range(num_modes):
        if any(channel_at(ch, k) != () for ch in SOLE_CHANNELS):
            sole_pass += 1
        else:
            if len(violations) < 30:
                violations.append(f"mode {k}: no sole in contact")
    at_least_one_sole_fraction = sole_pass / num_modes if num_modes else 0.0

    # ---- no_noop_mode (per transition) ------------------------------------
    if num_modes <= 1:
        no_noop_fraction = 1.0
    else:
        all_channels = list(plan.keys())
        noop_pass = 0
        transitions = num_modes - 1
        for k in range(transitions):
            same = all(
                channel_at(ch, k) == channel_at(ch, k + 1) for ch in all_channels
            )
            if same:
                if len(violations) < 30:
                    violations.append(f"transition {k}->{k+1}: no contact changes (noop)")
            else:
                noop_pass += 1
        no_noop_fraction = noop_pass / transitions

    # ---- object_passive (per transition, per object channel) --------------
    object_channels = [ch for ch in plan.keys() if ch in movable]
    if not object_channels or num_modes <= 1:
        object_passive_fraction = 1.0
    else:
        passive_total = 0
        passive_pass = 0
        for ch in object_channels:
            for k in range(num_modes - 1):
                prev = channel_at(ch, k)
                curr = channel_at(ch, k + 1)
                # Only transitions where the object actually has contact in
                # the new mode and the previous mode also had contact (but
                # different) are violations. Going () -> contact is fine
                # (placement / landing). Going contact -> () is fine (lift).
                # Going contact -> same contact is also fine (no change),
                # and so is switching sticking <-> sliding on the same
                # surfaces (faro keeps the contact).
                if prev != () and curr != () and tuple(prev)[:2] != tuple(curr)[:2]:
                    passive_total += 1
                    if len(violations) < 30:
                        violations.append(
                            f"{ch} transition {k}->{k+1}: object teleported "
                            f"from {prev} to {curr} without going free"
                        )
                else:
                    passive_total += 1
                    passive_pass += 1
        object_passive_fraction = (
            passive_pass / passive_total if passive_total else 1.0
        )

    structure_score = (
        well_formed_fraction
        + equal_length_bool
        + at_least_one_sole_fraction
        + no_noop_fraction
        + object_passive_fraction
    ) / 5.0

    efficiency = max(0.0, min(1.0, 1.0 - num_modes / MAX_MODES))
    combined_score = structure_score + efficiency

    return {
        "num_modes": num_modes,
        "num_channels": len(plan),
        "well_formed_fraction": well_formed_fraction,
        "equal_length_bool": equal_length_bool,
        "at_least_one_sole_fraction": at_least_one_sole_fraction,
        "no_noop_fraction": no_noop_fraction,
        "object_passive_fraction": object_passive_fraction,
        "structure_score": structure_score,
        "efficiency": efficiency,
        "combined_score": combined_score,
        "violations": violations,
    }


def _check_discrete_goal(
    plan: Dict[str, List[Tuple]],
    discrete_goal: Dict[str, Tuple],
) -> Tuple[bool, List[str], int, int]:
    """
    Check if the final mode of *plan* satisfies the discrete contact goal.

    Returns (match, violations, matched_channels, total_channels).
    `matched_channels / total_channels` is the per-channel progress
    fraction in [0, 1] used by the stratified scorer; the boolean stays
    True iff matched == total.
    """
    total_channels = len(discrete_goal)
    if not plan:
        return (
            False,
            ["plan is empty, cannot check discrete goal"],
            0,
            total_channels,
        )

    violations: List[str] = []
    matched = 0
    for channel, target_tuple in discrete_goal.items():
        if channel not in plan:
            violations.append(f"discrete goal: channel '{channel}' not in plan")
            continue
        seq = plan[channel]
        if not seq:
            violations.append(f"discrete goal: channel '{channel}' is empty")
            continue
        final_entry = tuple(seq[-1]) if isinstance(seq[-1], (list, tuple)) else seq[-1]
        if final_entry != target_tuple:
            violations.append(
                f"discrete goal: {channel} final mode is {final_entry}, "
                f"expected {target_tuple}"
            )
            continue
        matched += 1
    return matched == total_channels, violations, matched, total_channels


def aggregate_planning_metrics(
    results: List[RunOutput],
    results_dir: str,
) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate."}

    contact_plan, scene = results[0]

    # Ablation toggles (SHINKA_* env vars, exported by run_evo.py from
    # its YAML task: block or by run_search.sh).
    # _disable_to     — skip the TO solve; with to_cost left None the
    #                   plan scores 0.
    # _hide_ik_fb     — keep the IK-failure diagnostic out of
    #                   public_metrics, so the LLM does not see WHERE
    #                   IK failed (used by the IK-no-feedback ablation).
    _disable_to = os.environ.get("SHINKA_DISABLE_TO") == "1"
    _hide_ik_fb = os.environ.get("SHINKA_HIDE_IK_FEEDBACK") == "1"
    # Opt-in modes (task.score_mode / task.diversity; see scoring.py and
    # plan_signature.py). Defaults keep the CoRL ablation behavior.
    _score_mode = active_score_mode()
    _diversity = os.environ.get(DIVERSITY_ENV) == "1"

    # Duplicate-plan short-circuit: if this exact contact_mode_seq was
    # already evaluated in this run, skip the expensive IK+TO stages
    # and return a zero-score with informative text_feedback. Keeps
    # the DB entry (shinka core isn't touched), but archive selection
    # naturally deprioritizes combined_score=0 nodes.
    plan_hash, prior = _check_plan_duplicate(contact_plan, results_dir)
    signature = plan_signature(contact_plan)
    patches = plan_patches(contact_plan)
    if prior is not None:
        return _with_diversity_fields(_build_duplicate_metrics(
            contact_plan, results_dir, plan_hash, prior,
        ), signature, patches, _diversity)

    # Over-cap short-circuit: if the EXPANDED contact_mode_seq exceeds
    # the KSO wrapper's MAX_NUM_NODES, the solver pool cannot score
    # it. Skip _score_contact_plan, the discrete-goal check, IK, and
    # TO entirely — return immediately with a clear text_feedback so
    # the LLM trims the plan.
    expanded_modes = _count_expanded_modes(contact_plan)
    if expanded_modes is not None and expanded_modes > MAX_NUM_NODES:
        # Record it so a resubmission is caught as a duplicate.
        _record_plan_result(
            plan_hash, results_dir, combined_score=0.0,
            top_violation=f"too many contact modes ({expanded_modes} > {MAX_NUM_NODES})",
            signature=signature, patches=patches,
        )
        return _with_diversity_fields(_build_too_many_modes_metrics(
            contact_plan, results_dir, expanded_modes,
        ), signature, patches, _diversity)

    # Structural metrics are kept as diagnostics only — they no longer
    # contribute to ranking.
    scored = _score_contact_plan(contact_plan, scene)

    task = load_task()
    discrete_goal = task.discrete_goal

    # ------------------------------------------------------------------
    # Stage 1: Discrete goal check
    # ------------------------------------------------------------------
    goal_match, goal_violations, goal_matched, goal_total = _check_discrete_goal(
        contact_plan, discrete_goal,
    )

    # ------------------------------------------------------------------
    # Short-plan guard: IK requires at least 2 modes. If the plan has
    # fewer, emit a text_feedback so the LLM knows to extend the plan.
    # ------------------------------------------------------------------
    text_feedback: Optional[str] = None
    try:
        first_seq = next(iter(contact_plan.values())) if contact_plan else []
        n_modes = len(first_seq) if isinstance(first_seq, list) else 0
    except (StopIteration, TypeError):
        n_modes = 0

    if n_modes < 2:
        text_feedback = (
            f"The plan has {n_modes} mode(s); IK requires at least 2 modes "
            f"(an initial mode and at least one transition). Extend the plan "
            f"by appending at least one more mode that differs from the "
            f"initial one."
        )

    # ------------------------------------------------------------------
    # Stage 2: IK feasibility (only if discrete goal matched AND plan long enough)
    # ------------------------------------------------------------------
    ik_import_error: Optional[str] = None
    ik_results_data: Optional[Dict[str, Any]] = None
    ik_diagnosis_raw: Optional[Dict[str, Any]] = None
    ik_feasible: Optional[bool] = None
    ik_cost: Optional[float] = None
    ik_mode: Optional[str] = None
    ik_hard_failed: bool = False

    # Long plans are evaluated incrementally (to_eval.TO_INCREMENTAL): no
    # full-plan KSO; each window is KSO-checked and solved from the previous
    # window's end, solved windows are cached per run.
    incremental = incremental_plan(n_modes) and not _disable_to

    # faro rejects a contact that changes surface without a free mode in
    # between before solving anything; say so without spending solver time.
    schedule_problems = schedule_errors(contact_plan) if goal_match and n_modes >= 2 else []
    if schedule_problems:
        ik_feasible = False
        m = re.match(r"mode (\d+)->", schedule_problems[0])
        k = int(m.group(1)) if m else 0
        ik_diagnosis_raw = {
            "kind": "schedule_error", "errors": schedule_problems,
            "total_modes": n_modes, "last_feasible_modes": k + 1,
            "first_infeasible_modes": k + 2,
        }

    def _add_events(diag: Dict[str, Any]) -> None:
        # Progress for the quality score: contact events (grasps, releases,
        # placements, steps onto another surface) that lie in the feasible
        # prefix, so padding with walking steps does not raise the score.
        events = event_modes(contact_plan)
        last = diag.get("last_feasible_modes")
        diag["events_total"] = len(events)
        diag["events_feasible"] = (
            len(events) if diag.get("kind") == "goal_missed"
            else sum(1 for mode in events if last is not None and mode <= last - 1)
        )

    # TO state, initialised here because _finalize (below) reads it and runs
    # before TO (and before the KSO diagnosis) for the provisional records.
    # The TO solve may repair the plan (walking steps inserted, to_eval
    # _step_repair): contact_plan then becomes the repaired plan the result
    # describes, original_plan stays what the program proposed.
    to_import_error: Optional[str] = None
    to_mode: Optional[str] = None
    to_solve = None
    original_plan = contact_plan

    def _finalize(
        to_solve: Any,
        to_import_error: Optional[str],
        text_feedback: Optional[str],
        provisional: bool = False,
    ) -> Dict[str, Any]:
        """Score, metrics, feedback, pickles, extra.json and the
        seen_plans record for the current state. Called once with
        ``provisional=True`` before TO (see below) and once at the end."""
        to_results_data = None if to_solve is None else to_solve.results
        to_cost = None if to_solve is None else to_solve.cost
        to_converged = None if to_solve is None else to_solve.converged
        # A usable trajectory: converged, or physically valid without
        # converging (to_eval / trajectory_check); scoring and feedback use it.
        to_valid = None if to_solve is None else getattr(to_solve, "valid", to_solve.converged)
        ik_diagnosis = ik_diagnosis_raw
        kso_task_quality: Optional[float] = None
        to_task_quality: Optional[float] = None
        to_quality_value: Optional[float] = None
        if _score_mode == "legacy":
            combined_score = legacy_score(to_cost)
        else:
            if ik_feasible:
                kso_task_quality = task_quality(task, ik_results_data)
            if to_solve is not None:
                to_task_quality = task_quality(task, to_solve.results)
                to_quality_value = to_quality(
                    _scored_violation(to_solve), to_solve.jerk_cost, to_solve.num_modes,
                )
            solved_modes = None if to_solve is None else getattr(to_solve, "solved_modes", None)
            combined_score = quality_score(
                goal_match=goal_match,
                n_modes=n_modes,
                kso_feasible=ik_feasible,
                kso_diagnosis=ik_diagnosis,
                kso_task_quality=kso_task_quality,
                to_ran=to_solve is not None,
                to_disabled=_disable_to,
                to_converged=to_valid,
                to_cost=to_cost,
                to_task_quality=to_task_quality,
                to_quality_value=to_quality_value,
                to_progress=None if solved_modes is None else solved_modes / max(to_solve.num_modes, 1),
            )

        # Plan-level diversity (task.diversity=true): how far this plan is from
        # every plan already evaluated in the run, both in WHICH contact
        # patches it uses (plan_patches, weighted higher) and in their ORDER
        # (plan_signature). Feeds the archive (plan_novelty criterion,
        # plan_patches cells) and the text feedback.
        novelty: Optional[float] = None
        patch_nov: Optional[float] = None
        order_nov: Optional[float] = None
        families: List[Dict[str, Any]] = []
        patch_families: List[Dict[str, Any]] = []
        if _diversity:
            prior_records = _seen_plan_records(results_dir, exclude=plan_hash)
            order_nov = plan_novelty(signature, (r.get("signature") for r in prior_records))
            patch_nov = patch_novelty(
            patches, (r.get("patches") for r in prior_records if r.get("patches")),
            ignore=_shared_patches(task),
        )
            novelty = combined_novelty(patch_nov, order_nov)
            families = family_stats(prior_records)
            patch_families = family_stats(prior_records, key="patches")

        # Persist the IK trajectory for later visualization.
        ik_run_path: Optional[str] = None
        if ik_results_data is not None:
            ik_run_path = os.path.join(results_dir, "ik_run.pkl")
            try:
                with open(ik_run_path, "wb") as f:
                    pickle.dump(
                        {
                            # KSO solved the plan as proposed (TO repairs
                            # come later; see plan_repairs in extra.json).
                            "contact_mode_seq": original_plan,
                            "ik_cost": ik_cost,
                            "results": ik_results_data,
                        },
                        f,
                    )
                print(f"IK trajectory saved to {ik_run_path}")
            except Exception as e:
                print(f"Error saving ik_run.pkl: {e}")
                ik_run_path = None

        # Persist the TO trajectory separately.
        to_run_path: Optional[str] = None
        if to_results_data is not None:
            to_run_path = os.path.join(results_dir, "to_run.pkl")
            try:
                with open(to_run_path, "wb") as f:
                    pickle.dump(
                        {
                            "contact_mode_seq": contact_plan,
                            "to_cost": to_cost,
                            "to_converged": to_converged,
                            "to_valid": to_valid,
                            "to_iterate": None if to_solve is None else getattr(to_solve, "iterate", "final"),
                            "to_segments": None if to_solve is None else getattr(to_solve, "segments", None),
                            "results": to_results_data,
                        },
                        f,
                    )
                print(f"TO trajectory saved to {to_run_path}")
            except Exception as e:
                print(f"Error saving to_run.pkl: {e}")
                to_run_path = None

        # What the LLM sees (prompts_base.perf_str prints every public key):
        # verdicts and the numbers that drive the score. Diagnostics,
        # descriptors and paths go to private metrics / extra.json.
        public_metrics: Dict[str, Any] = {
            "num_modes": scored["num_modes"],
            "goal_match": goal_match,
            "ik_feasible": ik_feasible,
            "to_converged": to_converged,
            "to_valid": to_valid,
            "to_cost": to_cost,
            "violations": scored["violations"][:20],
        }
        if ik_mode == "hard" or ik_hard_failed:
            public_metrics["kso_goal_mode"] = (
                "hard" if ik_mode == "hard" else "soft (hard target missed)"
            )
        if to_solve is not None:
            public_metrics["to_max_violation"] = to_solve.max_constraint_violation
            public_metrics["to_jerk_per_mode"] = to_solve.jerk_cost / max(to_solve.num_modes, 1)
            if getattr(to_solve, "segments", None) and getattr(to_solve, "solved_modes", None) is not None:
                public_metrics["to_solved_modes"] = to_solve.solved_modes
            if getattr(to_solve, "repairs", None):
                public_metrics["plan_repairs"] = len(to_solve.repairs)
        # Prepend goal violations so the LLM sees them first.
        for v in reversed(goal_violations[:5]):
            public_metrics["violations"].insert(0, v)
        kso_details: List[Dict[str, Any]] = []
        if ik_diagnosis is not None:
            kso_details = ik_diagnosis.get("violation_details") or []
            ik_diagnosis = {k: v for k, v in ik_diagnosis.items() if k != "violation_details"}
            if not _hide_ik_fb:
                last = ik_diagnosis.get("last_feasible_modes")
                first = ik_diagnosis.get("first_infeasible_modes")
                public_metrics["kso_feasible_through_mode"] = None if not last else last - 1
                public_metrics["kso_first_infeasible_mode"] = None if not first else first - 1
                if ik_diagnosis.get("break_event"):
                    public_metrics["kso_break_event"] = ik_diagnosis["break_event"]
        if _score_mode != "legacy":
            public_metrics["task_quality"] = (
                to_task_quality if to_task_quality is not None else kso_task_quality
            )
            public_metrics["to_quality"] = to_quality_value
        if _diversity:
            public_metrics["plan_patches"] = patches
            public_metrics["plan_novelty"] = novelty
            public_metrics["plan_patch_family_count"] = len(patch_families) + (
                0 if any(f["signature"] == patches for f in patch_families) else 1
            )
        # Solver feedback (KSO / TO constraint violations in plan vocabulary,
        # violation_feedback.py) and its one-line headline. Suppressed when
        # SHINKA_HIDE_IK_FEEDBACK=1 (no-natural-language-feedback ablation).
        solver_fb: Optional[str] = None
        if not _hide_ik_fb:
            score_line = None
            if _score_mode != "legacy" and to_solve is not None and to_valid:
                score_line = score_feedback(
                    combined_score, to_task_quality, to_quality_value,
                    quality_parts(task, to_solve.results),
                    _scored_violation(to_solve),
                    to_solve.jerk_cost / max(to_solve.num_modes, 1),
                )
            headline, solver_fb = _solver_feedback(
                task, contact_plan, ik_diagnosis, kso_details, to_solve,
                ik_import_error, to_import_error, results_dir, score_line,
            )
            if headline:
                public_metrics["violations"].insert(0, headline)
            if solver_fb:
                # First problem line also goes into the always-shown metrics.
                first_line = next((ln for ln in solver_fb.splitlines() if ln.startswith("- ")), None)
                if first_line:
                    public_metrics["violations"].insert(1 if headline else 0, first_line[2:])
        private_metrics = {
            # structural diagnostics (no longer used for ranking)
            "well_formed_fraction": scored["well_formed_fraction"],
            "equal_length_bool": scored["equal_length_bool"],
            "at_least_one_sole_fraction": scored["at_least_one_sole_fraction"],
            "no_noop_fraction": scored["no_noop_fraction"],
            "object_passive_fraction": scored["object_passive_fraction"],
            "structure_score": scored["structure_score"],
            "efficiency": scored["efficiency"],
        }
        if ik_mode is not None:
            private_metrics["ik_mode"] = ik_mode
        if ik_hard_failed:
            private_metrics["ik_hard_failed"] = True
        if ik_cost is not None:
            private_metrics["ik_cost"] = ik_cost
        if ik_diagnosis is not None:
            private_metrics["ik_diagnosis"] = ik_diagnosis
        if _diversity:
            private_metrics["plan_signature"] = signature
            private_metrics["plan_patch_novelty"] = patch_nov
            private_metrics["plan_order_novelty"] = order_nov
            private_metrics["plan_family_count"] = len(families) + (
                0 if any(f["signature"] == signature for f in families) else 1
            )
        if to_mode is not None:
            private_metrics["to_mode"] = to_mode
        if ik_import_error is not None:
            private_metrics["ik_import_error"] = ik_import_error
        if to_import_error is not None:
            private_metrics["to_import_error"] = to_import_error

        metrics = {
            "combined_score": combined_score,
            "public": public_metrics,
            "private": private_metrics,
        }
        if solver_fb:
            text_feedback = f"{text_feedback}\n\n{solver_fb}" if text_feedback else solver_fb
        if _diversity:
            family_fb = _family_feedback(
                signature, patches, novelty, families, patch_families,
                _face_names(task.scene),
            )
            text_feedback = (
                f"{text_feedback}\n\n{family_fb}" if text_feedback else family_fb
            )
        if provisional:
            # Written before a slow stage (TO, or the KSO break-point search
            # of a failed plan) so a job killed during it still yields a
            # record (the runner loads whatever metrics.json exists after
            # cancelling); the final write replaces it.
            if incremental and ik_feasible is None:
                public_metrics["to_status"] = "pending"
                pending = (
                    f"The {n_modes}-mode plan is evaluated window by window (KSO and TO "
                    f"per window, left to right). If you are reading this, the "
                    f"evaluation did not finish within the time limit. Solved windows "
                    f"are cached: a plan with the same beginning is evaluated from "
                    f"where this one stopped."
                )
            elif ik_feasible:
                public_metrics["to_status"] = "pending"
                pending = (
                    f"KSO found the {n_modes}-mode plan feasible. If you are reading "
                    f"this, trajectory optimization (TO) did not finish within the "
                    f"evaluation time limit (TO time grows with plan length; "
                    f"{n_modes} modes is about {n_modes} s of motion). The plan is "
                    f"kinematically valid; TO was cut off, not failed, so keep its "
                    f"contacts and strategy (an identical plan would count as a "
                    f"duplicate)."
                )
            else:
                public_metrics["to_status"] = "skipped"
                pending = (
                    f"KSO could not solve the {n_modes}-mode plan. If you are reading "
                    f"this, the search for the exact break point did not finish within "
                    f"the evaluation time limit; the violations below come from the "
                    f"full-plan attempts."
                )
            text_feedback = f"{pending}\n\n{text_feedback}" if text_feedback else pending
        if text_feedback is not None:
            metrics["text_feedback"] = text_feedback

        extra_file = os.path.join(results_dir, "extra.json")
        try:
            payload = {
                "contact_mode_seq": {
                    ch: [list(e) if isinstance(e, tuple) else e for e in seq]
                    if isinstance(seq, list) else seq
                    for ch, seq in contact_plan.items()
                },
                "scored": scored,
                "goal_match": goal_match,
                "ik_feasible": ik_feasible,
                "ik_cost": ik_cost,
                "to_converged": to_converged,
                "to_cost": to_cost,
                "ik_run_path": ik_run_path,
                "to_run_path": to_run_path,
                "ik_import_error": ik_import_error,
                "to_import_error": to_import_error,
                "ik_diagnosis": ik_diagnosis,
                "kso_violations": kso_details or None,
                "to_status": None if to_solve is None else to_solve.status,
                "to_violations": None if to_solve is None else to_solve.violations,
                "to_valid": to_valid,
                "to_physical": None if to_solve is None else _physical_record(to_solve),
                "to_goal": None if to_solve is None else getattr(to_solve, "goal", None),
                "to_fail_reason": None if to_solve is None else getattr(to_solve, "fail_reason", "") or None,
                "to_rescues": None if to_solve is None else getattr(to_solve, "rescues", 0),
                "to_segments": None if to_solve is None else getattr(to_solve, "segments", None),
                "to_polish": None if to_solve is None else getattr(to_solve, "polish", None),
                "to_solved_modes": None if to_solve is None else getattr(to_solve, "solved_modes", None),
                # Walking steps the evaluator inserted during TO (to_eval
                # _step_repair); contact_mode_seq above is the repaired plan
                # the TO result describes, this is the plan as proposed.
                "plan_repairs": None if to_solve is None else (getattr(to_solve, "repairs", None) or None),
                "original_contact_mode_seq": None if contact_plan is original_plan else {
                    ch: [list(e) if isinstance(e, tuple) else e for e in seq] if isinstance(seq, list) else seq
                    for ch, seq in original_plan.items()
                },
                "plan_signature": signature,
                "plan_novelty": novelty,
                "plan_patches": patches,
                "score_mode": _score_mode,
                "task_quality": public_metrics.get("task_quality"),
                "to_quality": to_quality_value,
                "to_converged_raw": to_converged,
                "to_cost_parts": None if to_solve is None else {
                    "constraint_violation_sum": to_solve.constraint_violation,
                    "constraint_violation_max": to_solve.max_constraint_violation,
                    "jerk_cost": to_solve.jerk_cost,
                    "num_modes": to_solve.num_modes,
                    "num_nodes": to_solve.num_nodes,
                },
            }
            with open(extra_file, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Plan details saved to {extra_file}")
        except Exception as e:
            print(f"Error saving extra.json: {e}")
            metrics["extra_save_error"] = str(e)

        # Register this plan's outcome so later generations can detect
        # duplicates. First-write-wins; a tail-race with a concurrent
        # worker evaluating the same plan is a no-op on the loser side.
        top_violation = ""
        if public_metrics.get("violations"):
            top_violation = str(public_metrics["violations"][0])
        _record_plan_result(
            plan_hash,
            results_dir,
            combined_score=float(metrics["combined_score"]),
            top_violation=top_violation,
            signature=signature,
            patches=patches,
            provisional=provisional,
        )
        if contact_plan is not original_plan:
            # The repaired plan was evaluated too: resubmitting it verbatim
            # is a duplicate of this result.
            _record_plan_result(
                _canonical_plan_hash(contact_plan), results_dir,
                combined_score=float(metrics["combined_score"]),
                top_violation=f"evaluated as the repaired form of plan {plan_hash[:8]}: {top_violation}",
                signature=signature, patches=patches, provisional=provisional,
            )
        return metrics

    # ------------------------------------------------------------------
    # Stage 2 (continued): the KSO solve. It runs after _finalize exists
    # because a plan that fails the full-plan solves gets a provisional
    # record (the soft attempt's violations, diagnosis "pending") before
    # the slow break-point search, so a job killed during it still yields
    # a record and feedback (ik_eval.evaluate_ik on_full_failure).
    # ------------------------------------------------------------------
    if not schedule_problems and goal_match and n_modes >= 2 and incremental:
        print(f"[evaluate] {n_modes}-mode plan: incremental evaluation (KSO and TO per window, "
              f"left to right; no full-plan KSO)", flush=True)
    elif not schedule_problems and goal_match and n_modes >= 2:
        def _kso_full_failure(diag: Dict[str, Any]) -> None:
            nonlocal ik_feasible, ik_diagnosis_raw
            ik_feasible, ik_diagnosis_raw = False, dict(diag)
            _add_events(ik_diagnosis_raw)
            try:
                save_json_results(
                    results_dir, _finalize(None, None, text_feedback, provisional=True),
                    correct=True, verbose=False,
                )
            except Exception as e:  # noqa: BLE001 — best effort
                print(f"[evaluate] provisional KSO record not written: {type(e).__name__}: {e}")

        try:
            from ik_eval import evaluate_ik
            (
                ik_result,
                ik_results_data,
                ik_diagnosis_raw,
                ik_mode,
                ik_hard_failed,
            ) = evaluate_ik(contact_plan, terminal_goal=task.terminal_goal,
                            on_full_failure=_kso_full_failure)
        except Exception as e:
            ik_result = False
            ik_import_error = f"{type(e).__name__}: {e}"

        if ik_result is False:
            ik_feasible = False
        else:
            ik_feasible = True
            ik_cost = float(ik_result)
    if ik_diagnosis_raw is not None:
        _add_events(ik_diagnosis_raw)

    # ------------------------------------------------------------------
    # Stage 3: TO cost (only if discrete goal matched AND IK feasible)
    # ------------------------------------------------------------------
    if goal_match and not _disable_to and (ik_feasible or (incremental and ik_feasible is None)):
        to_mode = to_eval_mode(task)
        if incremental:
            run_root = _find_run_root(results_dir)
            if run_root is not None:
                from segment_cache import CACHE_DIR_ENV

                os.environ[CACHE_DIR_ENV] = os.path.join(run_root, "segment_cache")
        # A TO solve can run for many minutes and the job may be killed by
        # the runner's time limit meanwhile. Persist a provisional
        # "KSO feasible, TO pending" record first (same files
        # run_shinka_eval writes at the end).
        try:
            save_json_results(
                results_dir, _finalize(None, None, text_feedback, provisional=True),
                correct=True, verbose=False,
            )
        except Exception as e:  # noqa: BLE001 — best effort
            print(f"[evaluate] provisional metrics not written: {type(e).__name__}: {e}")
        # All TO construction and solve output (CasADi substitution
        # messages, moto iteration prints, etc.) is redirected to a
        # logfile in results_dir instead of the worker's stdout. Keep
        # the broad except: a bad TO run must never kill the LLM
        # evaluation — failures surface as combined_score=0 and the
        # error text lands in private_metrics.to_import_error.
        to_log_path = os.path.join(results_dir, "to_solver.log")
        try:
            with _redirect_fds_to(to_log_path):
                # Seed TO from the KSO solution (to_warm_start.py;
                # SHINKA_TO_WARM_START=0 keeps faro's standing-pose guess).
                to_solve = solve_to(task, contact_plan, kso_results=ik_results_data)
        except Exception as e:
            to_import_error = f"{type(e).__name__}: {e}"
        if to_solve is not None and getattr(to_solve, "plan", None):
            contact_plan = to_solve.plan
            scored = _score_contact_plan(contact_plan, scene)
            n_modes = len(next(iter(contact_plan.values())))
        if to_solve is not None and incremental:
            # The first window's KSO stands in for the full-plan verdict; a
            # later window without a kinematic solution is a partial TO result.
            win = _kso_window_failure(to_solve)
            if getattr(to_solve, "kso_feasible", None) is False and win is not None:
                a, e, details = win
                ik_feasible = False
                ik_diagnosis_raw = {"kind": "window", "total_modes": n_modes, "last_feasible_modes": a or None,
                                    "first_infeasible_modes": e, "violation_details": details}
                _add_events(ik_diagnosis_raw)
            else:
                ik_feasible = True

    return _finalize(to_solve, to_import_error, text_feedback)


def get_planning_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


_TASK_ENV_VARS = TASK_ENV_VARS
_NONE_STRS = NONE_STRS


SEEN_PLANS_FILENAME = "seen_plans.json"


_canonical_plan_hash = canonical_plan_hash


def _find_run_root(results_dir: str) -> Optional[str]:
    """`results_dir` is typically `<run_root>/<gen_N|best>/results`, so
    two levels up is the run root. Returns None if that path isn't a
    directory (e.g. standalone evaluator invocations)."""
    rd = os.path.abspath(results_dir)
    root = os.path.dirname(os.path.dirname(rd))
    return root if os.path.isdir(root) else None


def _current_gen_label(results_dir: str) -> str:
    """Extract `gen_N` / `best` from results_dir for diagnostics."""
    m = re.search(r"(gen_\d+|best)", results_dir)
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(os.path.abspath(results_dir)))


@contextlib.contextmanager
def _locked_seen_plans(path: str):
    """Yield (data_dict, save_fn) under an exclusive file lock.

    Opens the file with mode `"a+"` so it's created-if-missing without
    truncating existing contents. The caller may mutate `data_dict`
    and call `save_fn()` to persist. Lock is held for the entire
    with-block so parallel workers serialize on read-modify-write.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                data = {}

            def save() -> None:
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            yield data, save
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _count_expanded_modes(plan: Dict[str, List[Any]]) -> Optional[int]:
    """Return the longest channel sequence's length (the expanded
    mode count after walk() / append_mode helpers ran), or None if the
    plan has no usable list channels. Used by the over-cap guard."""
    if not plan:
        return 0
    lengths = [
        len(seq) for seq in plan.values() if isinstance(seq, list)
    ]
    if not lengths:
        return None
    return max(lengths)


def _check_plan_duplicate(
    plan: Dict[str, List[Any]],
    results_dir: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return (hash, prior_record) if the plan was already seen in
    this run, else (hash, None). Always returns a usable hash so the
    caller can pass it to ``_record_plan_result`` on the novel path."""
    plan_hash = _canonical_plan_hash(plan)
    run_root = _find_run_root(results_dir)
    if run_root is None:
        return plan_hash, None
    path = os.path.join(run_root, SEEN_PLANS_FILENAME)
    gen = _current_gen_label(results_dir)
    with _locked_seen_plans(path) as (data, _save):
        prior = data.get(plan_hash)
    # This generation's own provisional record (written before TO) is
    # not a duplicate.
    if prior is not None and prior.get("provisional") and prior.get("first_seen_gen") == gen:
        return plan_hash, None
    return plan_hash, prior


def _record_plan_result(
    plan_hash: str,
    results_dir: str,
    combined_score: float,
    top_violation: str,
    signature: Optional[str] = None,
    patches: Optional[str] = None,
    provisional: bool = False,
) -> None:
    """Register this plan's outcome in seen_plans.json. First-write-wins,
    except that a generation's own provisional record (written before
    TO) is replaced by its final one."""
    run_root = _find_run_root(results_dir)
    if run_root is None:
        return
    path = os.path.join(run_root, SEEN_PLANS_FILENAME)
    gen = _current_gen_label(results_dir)
    with _locked_seen_plans(path) as (data, save):
        prior = data.get(plan_hash)
        if prior is not None and not (
            prior.get("provisional") and prior.get("first_seen_gen") == gen
        ):
            return
        data[plan_hash] = {
            "first_seen_gen": gen,
            "combined_score": float(combined_score),
            "top_violation": str(top_violation or ""),
            "signature": signature,
            "patches": patches,
            "provisional": provisional,
        }
        save()


def _seen_plan_records(
    results_dir: str, exclude: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """All seen_plans.json records of this run, minus ``exclude`` (a hash)."""
    run_root = _find_run_root(results_dir)
    if run_root is None:
        return []
    path = os.path.join(run_root, SEEN_PLANS_FILENAME)
    with _locked_seen_plans(path) as (data, save):
        # Records written before plan_patches existed: derive it from the
        # plan saved in that generation's extra.json, once.
        changed = False
        for rec in data.values():
            if "patches" in rec:
                continue
            extra = os.path.join(run_root, rec.get("first_seen_gen", ""), "results", "extra.json")
            try:
                with open(extra) as f:
                    plan = json.load(f).get("contact_mode_seq")
            except (OSError, json.JSONDecodeError):
                plan = None
            rec["patches"] = plan_patches(plan) if isinstance(plan, dict) else None
            changed = True
        if changed:
            save()
        return [rec for h, rec in data.items() if h != exclude]


def _scored_violation(to_solve: Any) -> float:
    """Violation behind to_quality: physical for a valid non-converged
    trajectory (ToSolve.effective_max_violation), else the solver's."""
    return float(getattr(to_solve, "effective_max_violation", to_solve.max_constraint_violation))


def _grasp_lines(task: Any, plan: Dict[str, List[Any]], to_solve: Any) -> List[str]:
    """grasp_probe feedback for a failed segmented TO solve: whether a
    standing pose exists for the failing window's grasp (static IK)."""
    try:
        import grasp_probe

        return grasp_probe.feedback(task, plan, to_solve)
    except Exception:  # noqa: BLE001 - feedback is best effort
        return []


def _goal_lines(goal: Dict[str, Any]) -> List[str]:
    """One line per terminal-goal target the trajectory misses (to_eval.goal_check)."""
    import math

    out: List[str] = []
    for name, t in sorted(goal.get("targets", {}).items(), key=lambda kv: -kv[1].get("excess", 0.0)):
        if t.get("excess", 0.0) <= goal.get("slack", 0.0):
            continue
        parts = [f"{t['position_error']:.2f} m from its target position (tolerance {t['position_tolerance']:.2f} m)"]
        if "rotation_error" in t:
            parts.append(f"{math.degrees(t['rotation_error']):.0f}° from its target orientation "
                         f"(tolerance {math.degrees(t['rotation_tolerance']):.0f}°)")
        out.append(f"{name} ends {' and '.join(parts)}.")
    return out


def _physical_record(to_solve: Any) -> Optional[Dict[str, Any]]:
    """Compact trajectory_check record for extra.json."""
    rep = getattr(to_solve, "physical", None)
    if not rep:
        return None
    return {"pass": bool(rep.get("pass")), "fails": list(rep.get("fails", [])),
            "worst": {k: float(v) for k, v in rep.get("worst", {}).items()},
            "where": rep.get("where", {}), "iterate": getattr(to_solve, "iterate", "final")}


def _solver_feedback(
    task: Any,
    plan: Dict[str, List[Any]],
    ik_diagnosis: Optional[Dict[str, Any]],
    kso_details: List[Dict[str, Any]],
    to_solve: Any,
    ik_import_error: Optional[str] = None,
    to_import_error: Optional[str] = None,
    results_dir: Optional[str] = None,
    score_line: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """(headline, 'Solver feedback' text block): why KSO / TO rejected or
    penalized the plan, in plan vocabulary (violation_feedback.py). The
    headline is the one line kept in the metrics; it and the KSO bullets
    describe the same solve. (None, None) if nothing to say."""
    try:
        resolver = NameResolver(task.scene, plan, resolved_state=getattr(task, "resolved_state", None))
    except Exception:  # noqa: BLE001 — feedback is best effort
        resolver = NameResolver(None, plan)
    headline: Optional[str] = None
    blocks: List[str] = []
    if ik_import_error:
        headline = f"KSO could not run: {ik_import_error}"
        blocks.append(f"{headline} (an evaluator / solver error, not a property of the plan).")
    elif ik_diagnosis is not None:
        diag = dict(ik_diagnosis)
        if kso_details:
            diag["violation_details"] = kso_details
        headline, lines = kso_feedback(diag, resolver)
        best = diag.get("best_max_violation")
        if best is not None and best < 3 * 1e-2 and diag.get("kind") == "prefix":
            lines.append(f"(The best full-plan solve was nearly feasible: max violation "
                         f"{best:.3f}, cutoff 0.01.)")
        blocks.append("\n".join([headline] + list(lines)))  # lines come bulleted
    seg = _segment_failure_text(to_solve)
    kso_win = _kso_window_failure(to_solve)
    prefix = _prefix_lines(to_solve)
    if to_import_error:
        tail = _log_tail(os.path.join(results_dir, "to_solver.log")) if results_dir else ""
        blocks.append(f"Trajectory optimization (TO) crashed: {to_import_error}."
                      + (f" Solver log tail:\n{tail}" if tail else ""))
    elif kso_win is not None and ik_diagnosis is None:
        a, e, details = kso_win
        lines = summarize([Violation(**d) for d in details], resolver, max_items=4) + prefix
        blocks.append(f"KSO found no kinematic solution for modes {a}–{e - 1} from the state TO reached at "
                      f"mode {a}:\n" + "\n".join(_bullet(ln) for ln in lines))
    elif (to_solve is not None and not getattr(to_solve, "goal_met", True)
          and (to_solve.converged or getattr(to_solve, "physically_valid", False))):
        blocks.append(f"Trajectory optimization (TO) {to_status_text(to_solve.status)}{seg} and the trajectory is "
                      "physically valid, but it does not reach the task goal, so it does not count as solved:\n"
                      + "\n".join(_bullet(ln) for ln in _goal_lines(to_solve.goal) + prefix))
    elif to_solve is not None and not to_solve.converged and getattr(to_solve, "valid", False):
        blocks.append("Trajectory optimization (TO) did not reach its solver tolerance, but the "
                      "trajectory is physically valid (dynamics, joint limits, contacts and "
                      "penetration within limits) and reaches the task goal, so it counts as a "
                      "solved trajectory.")
    elif to_solve is not None and not to_solve.converged and getattr(to_solve, "physical", None):
        import trajectory_check

        lines = trajectory_check.describe(to_solve.physical)
        lines = (lines + _grip_lines(task, plan) + _step_lines(task, plan, to_solve, resolver)
                 + _grasp_lines(task, plan, to_solve) + prefix)
        blocks.append(f"Trajectory optimization (TO) {to_status_text(to_solve.status)}{seg}; the trajectory "
                      "is not physically valid:\n" + "\n".join(_bullet(ln) for ln in lines))
    elif to_solve is not None and to_solve.status == 2:
        # A numerical failure leaves no trajectory: the final iterate's
        # violation rows describe the breakdown (shelf_unstack gen_29: soles
        # "needing" 19 N of friction in a walking window), not the plan.
        lines = _grasp_lines(task, plan, to_solve) + _grip_lines(task, plan) + prefix
        blocks.append(f"Trajectory optimization (TO) {to_status_text(2)}{seg}: the solver found no feasible motion "
                      "for these modes and its last iterate is not a trajectory, so its constraint violations are not "
                      "listed. Usual causes: a contact in this window that the robot cannot reach from where it stands, "
                      "or more motion between consecutive modes than the robot can make."
                      + ("\n" + "\n".join(_bullet(ln) for ln in lines) if lines else ""))
    elif to_solve is not None and to_solve.violations:
        violations = [Violation(**d) for d in to_solve.violations]
        if not to_solve.converged:
            head = f"Trajectory optimization (TO) {to_status_text(to_solve.status)}{seg}. Largest violations:"
        else:
            head = ("TO converged; largest remaining violations "
                    "(reducing them raises the score):")
        lines = summarize(violations, resolver, max_items=4)
        if not to_solve.converged:
            lines = lines + _grip_lines(task, plan) + _step_lines(task, plan, to_solve, resolver) + prefix
        blocks.append(head + "\n" + "\n".join(_bullet(ln) for ln in lines))
    elif to_solve is not None and not to_solve.converged:
        blocks.append("\n".join([f"Trajectory optimization (TO) {to_status_text(to_solve.status)}{seg}."]
                                 + [_bullet(ln) for ln in _grip_lines(task, plan)
                                    + _step_lines(task, plan, to_solve, resolver) + prefix]))
    repair_lines = _repair_lines(to_solve)
    if repair_lines:
        blocks.append("Plan repairs made by the evaluator (the scored plan includes them; make the same "
                      "edits in your program):\n" + "\n".join(_bullet(ln) for ln in repair_lines))
    if score_line:
        blocks.append(score_line)
    if not blocks:
        return headline, None
    return headline, "Solver feedback:\n" + "\n".join(blocks)


def _segment_failure_text(to_solve: Any) -> str:
    """Where a segmented TO solve broke: ' in segment i of k (modes a–e;
    modes 0–a-1 were solved)', '' for whole solves and solved plans."""
    segs = getattr(to_solve, "segments", None) or []
    for i, s in enumerate(segs):
        if s.get("status") in (None, 0) or (s.get("physical_pass") and s.get("status") != 2):
            continue
        a, b = s["modes"]
        e = (s.get("extra") or {}).get("window", [a, b])[1]
        solved = f"; modes 0–{a - 1} were solved" if a > 0 else ""
        return f" in segment {i + 1} of {len(segs)} (modes {a}–{e - 1}{solved})"
    return ""


def _kso_window_failure(to_solve: Any) -> Optional[Tuple[int, int, List[Dict[str, Any]]]]:
    """``(a, e, violations)`` of the window whose KSO found no solution in
    an incremental evaluation, None otherwise."""
    for s in getattr(to_solve, "segments", None) or []:
        if s.get("status") == 2 and str(s.get("fail_reason", "")).startswith("kso:"):
            a, b = s["modes"]
            e = (s.get("extra") or {}).get("window", [a, b])[1]
            return int(a), int(e), list(((s.get("extra") or {}).get("kso") or {}).get("violations") or [])
    return None


def _prefix_lines(to_solve: Any) -> List[str]:
    """Incremental evaluation: which leading modes are solved (and cached),
    so the LLM changes the plan from the failing window on."""
    segs = getattr(to_solve, "segments", None) or []
    solved = getattr(to_solve, "solved_modes", None)
    n = int(getattr(to_solve, "num_modes", 0) or 0)
    if not segs or solved is None or solved >= n or getattr(to_solve, "kso_feasible", None) is None:
        return []
    lines = []
    if solved > 0:
        lines.append(f"Modes 0–{solved - 1} are solved and cached: keep them exactly as they are and change "
                     f"the plan from mode {solved} on; a plan with the same first {solved} modes is evaluated "
                     f"from there at no extra cost.")
    cached = int(getattr(to_solve, "cached_modes", 0) or 0)
    if cached:
        lines.append(f"Modes 0–{cached - 1} were reused from an earlier evaluation.")
    return lines


def _repair_lines(to_solve: Any) -> List[str]:
    """One line per walking-step repair TO made to the plan (to_eval
    _step_repair), with the mode in the proposed plan's numbering."""
    lines = []
    for r in getattr(to_solve, "repairs", None) or []:
        mode = r.get("before_mode_original")
        mode = r.get("before_mode") if mode is None else mode
        lines.append(f"walk(plan, {r['steps']}) inserted right before mode {mode} of the proposed plan "
                     f"({r.get('reason', 'too few steps')}); TO then continued from there")
    return lines


def _grip_lines(task: Any, plan: Dict[str, List[Any]]) -> List[str]:
    """Carried objects held on adjacent / the same faces (grip_feedback.py)."""
    from grip_feedback import to_grip_feedback

    return to_grip_feedback(task, plan)


def _step_lines(task: Any, plan: Dict[str, List[Any]], to_solve: Any, resolver: Any) -> List[str]:
    """Too-long walking steps in a failed TO solve (step_feedback.py)."""
    from plan_checks import describe_transition
    from step_feedback import to_step_feedback

    if not getattr(to_solve, "results", None):
        return []
    describe = lambda m: describe_transition(  # noqa: E731
        plan, m, lambda fid: resolver.face_env.get(fid, f"face {fid}"), resolver.objects)
    return to_step_feedback(task, plan, to_solve.results, describe)


def _log_tail(path: str, n: int = 5) -> str:
    try:
        with open(path, errors="replace") as f:
            lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def _bullet(line: str) -> str:
    return line if line.startswith(("Hint", "(", "...")) else f"- {line}"


def _shared_patches(task: Any) -> set:
    """Patch tokens every plan of this task shares (initial contacts and
    the discrete goal), excluded from patch novelty."""
    try:
        from _task_loader import flat_contact_mode

        initial = flat_contact_mode(task.resolved_state.contact_mode)
    except Exception:  # noqa: BLE001 — novelty is diagnostics
        initial = {}
    return shared_patch_tokens(initial, getattr(task, "discrete_goal", None) or {})


def _face_names(scene: Any) -> Dict[str, Any]:
    """Lookup tables for describing contact patches by name."""
    from _task_loader import scene_faces

    env: Dict[int, str] = {}
    self_faces: Dict[str, Dict[int, str]] = {}
    for obj, faces in scene_faces(scene).items():
        for env_id, self_id, face in faces:
            env[env_id] = f"{obj} {face}"
            self_faces.setdefault(obj, {})[self_id] = face
    return {"env": env, "self": self_faces}


def _describe_patches(patches: str, names: Dict[str, Any]) -> str:
    """'left_ee on yellow_box left; yellow_box bottom on floor surface, ...'"""
    out = []
    for part in patches.split("|"):
        ch, _, used = part.partition(":")
        descs = []
        for u in used.split(","):
            if not u or u == "-":
                continue
            ids = [int(x) for x in u.split(".")]
            kind = " (sliding)" if ids[-1] == 2 else ""
            if len(ids) == 3:
                self_face = names["self"].get(ch, {}).get(ids[0], f"face {ids[0]}")
                descs.append(f"{self_face} on {names['env'].get(ids[1], ids[1])}{kind}")
            else:
                descs.append(f"{names['env'].get(ids[0], ids[0])}{kind}")
        if descs:
            out.append(f"{ch}: {', '.join(descs)}")
    return "; ".join(out)


def _family_feedback(
    signature: str,
    patches: str,
    novelty: Optional[float],
    families: List[Dict[str, Any]],
    patch_families: List[Dict[str, Any]],
    names: Dict[str, Any],
    max_listed: int = 25,
    max_orders: int = 8,
) -> str:
    """LLM-facing summary of explored contact patches and orderings
    (diversity mode). Lists what was tried; suggests nothing."""
    lines = [
        f"Novelty of this plan: {novelty:.2f} (0 = already explored, "
        "1 = completely new). Using contact patches no earlier plan used "
        "counts most; a new order of the same contacts counts less. Plans "
        "are ranked on both score and novelty."
    ]
    if patch_families:
        lines.append(
            f"Contact patch sets explored so far ({len(patch_families)}; best "
            "score first):"
        )
        for fam in patch_families[:max_listed]:
            mark = "  <- this plan" if fam["signature"] == patches else ""
            lines.append(
                f"  - [{fam['best_score']:.2f}, {fam['count']} plan(s)] "
                f"{_describe_patches(fam['signature'], names)}{mark}"
            )
    if families:
        lines.append(
            f"Contact orders explored so far ({len(families)}; best score "
            "first; 'walk' = any number of steps):"
        )
        for fam in families[:max_orders]:
            mark = "  <- this plan" if fam["signature"] == signature else ""
            lines.append(
                f"  - [{fam['best_score']:.2f}, {fam['count']} plan(s)] "
                f"{fam['signature']}{mark}"
            )
    return "\n".join(lines)


def _with_diversity_fields(
    metrics: Dict[str, Any], signature: str, patches: str, diversity: bool,
) -> Dict[str, Any]:
    """Give short-circuit results the descriptor fields the diversity
    archive ranks on: their own descriptor cell and zero novelty."""
    if diversity:
        metrics["public"]["plan_signature"] = signature
        metrics["public"]["plan_patches"] = patches
        metrics["public"]["plan_novelty"] = 0.0
    return metrics


def _build_duplicate_metrics(
    contact_plan: Dict[str, List[Any]],
    results_dir: str,
    plan_hash: str,
    prior: Dict[str, Any],
) -> Dict[str, Any]:
    """Short-circuit return value for a duplicate plan — no IK/TO work."""
    first_gen = prior.get("first_seen_gen", "?")
    prior_score = float(prior.get("combined_score", 0.0))
    prior_violation = str(prior.get("top_violation", "")).strip()

    hint = "Propose a plan that differs from it."
    prior_clause = (
        f" Prior outcome: {prior_violation}." if prior_violation else ""
    )
    text_feedback = (
        f"Duplicate plan: this contact_mode_seq is identical to one "
        f"evaluated at {first_gen} (combined_score={prior_score:.2f})."
        f"{prior_clause} {hint}"
    )
    dup_marker = (
        f"duplicate plan (hash {plan_hash[:8]}; first seen {first_gen})"
    )

    # Keep a lean extra.json for human inspection.
    try:
        with open(os.path.join(results_dir, "extra.json"), "w") as f:
            json.dump(
                {
                    "is_duplicate": True,
                    "plan_hash": plan_hash,
                    "first_seen_gen": first_gen,
                    "prior_combined_score": prior_score,
                    "prior_top_violation": prior_violation,
                    "contact_mode_seq": {
                        ch: [list(e) if isinstance(e, tuple) else e for e in seq]
                        if isinstance(seq, list) else seq
                        for ch, seq in contact_plan.items()
                    },
                },
                f,
                indent=2,
            )
    except Exception as e:  # noqa: BLE001 — extra.json is diagnostics only
        print(f"[duplicate] failed to write extra.json: {e}")

    return {
        "combined_score": 0.0,
        "public": {
            "is_duplicate": True,
            "plan_hash": plan_hash,
            "first_seen_gen": first_gen,
            "violations": [dup_marker],
        },
        "private": {
            "prior_combined_score": prior_score,
            "prior_top_violation": prior_violation,
        },
        "text_feedback": text_feedback,
    }


def _build_too_many_modes_metrics(
    contact_plan: Dict[str, List[Any]],
    results_dir: str,
    expanded_modes: int,
) -> Dict[str, Any]:
    """Short-circuit for plans whose expanded mode count exceeds
    MAX_NUM_NODES — no structural scoring, no goal check, no IK, no TO.
    Mirrors `_build_duplicate_metrics`: minimal extra.json + a clear
    text_feedback so the LLM trims the plan in the next generation."""
    overage = expanded_modes - MAX_NUM_NODES
    text_feedback = (
        f"Too many contact modes: the expanded contact_mode_seq has "
        f"{expanded_modes} modes, which exceeds the MAX_NUM_NODES cap "
        f"of {MAX_NUM_NODES} by {overage}. The IK solver only covers "
        f"plans of 2 to {MAX_NUM_NODES} modes, so IK and TO were skipped. "
        f"The plan must be shorter (walk(plan, n) appends n + 1 modes)."
    )
    marker = (
        f"too many contact modes ({expanded_modes} > MAX_NUM_NODES="
        f"{MAX_NUM_NODES}); IK/TO skipped"
    )

    try:
        with open(os.path.join(results_dir, "extra.json"), "w") as f:
            json.dump(
                {
                    "is_too_many_modes": True,
                    "expanded_modes": expanded_modes,
                    "max_num_nodes": MAX_NUM_NODES,
                    "contact_mode_seq": {
                        ch: [list(e) if isinstance(e, tuple) else e for e in seq]
                        if isinstance(seq, list) else seq
                        for ch, seq in contact_plan.items()
                    },
                },
                f,
                indent=2,
            )
    except Exception as e:  # noqa: BLE001 — extra.json is diagnostics only
        print(f"[too-many-modes] failed to write extra.json: {e}")

    return {
        "combined_score": 0.0,
        "public": {
            "is_too_many_modes": True,
            "expanded_modes": expanded_modes,
            "max_num_nodes": MAX_NUM_NODES,
            "violations": [marker],
        },
        "private": {},
        "text_feedback": text_feedback,
    }


def _redirect_fds_to(log_path: str):
    """Fresh solver log per evaluation (see _run_common.redirect_fds_to)."""
    return redirect_fds_to(log_path, mode="w")


def _export_task_env(**fields: Optional[str]) -> None:
    """Push task selection to env vars so the LLM program's imports
    (env.py, ik_eval.py) and cached _task_loader calls see them. A field
    that was not passed leaves its env var as set by the caller (run_evo.py,
    a shell); an explicit "none" clears it."""
    for field, env_name in _TASK_ENV_VARS.items():
        val = fields.get(field)
        if val is None:
            continue
        if str(val).strip() in _NONE_STRS:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = str(val).strip()


def _export_mode_env(score_mode: Optional[str], diversity: Any) -> None:
    """Export task.score_mode / task.diversity when given; leave any
    pre-set SHINKA_SCORE_MODE / SHINKA_DIVERSITY untouched otherwise."""
    from scoring import SCORE_MODE_ENV

    if score_mode is not None and str(score_mode).strip() not in _NONE_STRS:
        os.environ[SCORE_MODE_ENV] = str(score_mode).strip().lower()
    if diversity is not None and str(diversity).strip() not in _NONE_STRS:
        on = str(diversity).strip().lower() in ("1", "true", "yes", "on")
        if on:
            os.environ[DIVERSITY_ENV] = "1"
        else:
            os.environ.pop(DIVERSITY_ENV, None)


def main(
    program_path: str,
    results_dir: str,
    scene: Optional[str] = None,
    initial_condition: Optional[str] = None,
    description: Optional[str] = None,
    score_mode: Optional[str] = None,
    diversity: Any = None,
):
    """Runs the loco-manipulation contact-planning evaluation."""
    _export_task_env(
        scene=scene,
        initial_condition=initial_condition,
        description=description,
    )
    _export_mode_env(score_mode, diversity)
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator(r: List[RunOutput]) -> Dict[str, Any]:
        return aggregate_planning_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_planning",
        num_runs=1,
        get_experiment_kwargs=get_planning_kwargs,
        validate_fn=validate_plan,
        aggregate_metrics_fn=_aggregator,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loco-manipulation contact-mode-sequence evaluator"
    )
    parser.add_argument(
        "--program_path", type=str, default="initial.py",
        help="Path to program to evaluate (must contain 'run_planning')",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Dir to save results (metrics.json, correct.json, extra.json)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
