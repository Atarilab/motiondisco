"""Plan-level proposal filter (loco_manipulation): duplicates and
over-explored strategy families are rejected before evaluation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import plan_precheck  # noqa: E402
from evaluate import _canonical_plan_hash  # noqa: E402
from plan_signature import plan_patches, plan_signature  # noqa: E402

PLAN = {"left_sole": [(1, 1)] * 3, "right_sole": [(1, 1), (), (1, 1)],
        "left_ee": [(), (3, 1), (3, 1)], "box": [(6, 2, 1)] * 3}
LONGER = {"left_sole": [(1, 1)] * 5, "right_sole": [(1, 1), (), (1, 1), (), (1, 1)],
          "left_ee": [(), (3, 1), (3, 1), (3, 1), (3, 1)], "box": [(6, 2, 1)] * 5}
assert plan_signature(PLAN) == plan_signature(LONGER)


def _run(tmp_path, seen, plan):
    results_dir = tmp_path / "gen_9" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "seen_plans.json").write_text(json.dumps(seen))
    return plan_precheck.precheck("prog.py", str(results_dir),
                                  plan_loader=lambda _f, _d: plan, max_family=2)


def _record(plan, score, gen="gen_1", provisional=False):
    return {"first_seen_gen": gen, "combined_score": score, "signature": plan_signature(plan),
            "patches": plan_patches(plan), "provisional": provisional}


def test_exact_duplicate_is_rejected(tmp_path):
    accept, info = _run(tmp_path, {_canonical_plan_hash(PLAN): _record(PLAN, 0.0)}, PLAN)
    assert not accept and info["reason"].startswith("duplicate of gen_1")


def test_own_provisional_record_is_not_a_duplicate(tmp_path):
    seen = {_canonical_plan_hash(PLAN): _record(PLAN, 0.1, provisional=True)}
    accept, info = _run(tmp_path, seen, PLAN)
    assert accept and info["reason"] == "accepted"


def test_saturated_family_is_rejected_but_unexplored_one_accepted(tmp_path):
    seen = {"h1": _record(PLAN, 0.3), "h2": _record(LONGER, 0.2, gen="gen_2"),
            "h3": _record(PLAN, 0.0, gen="gen_3")}  # failed member does not count
    accept, info = _run(tmp_path, seen, LONGER)
    assert not accept and "already has 2 valid plans" in info["reason"]
    other = {**PLAN, "left_ee": [(), (4, 1), (4, 1)]}
    accept, info = _run(tmp_path, seen, other)
    assert accept and info["family_valid_count"] == 0


def test_unextractable_plan_is_left_to_the_evaluator(tmp_path):
    accept, info = _run(tmp_path, {}, None)
    assert accept and "could not be extracted" in info["reason"]
