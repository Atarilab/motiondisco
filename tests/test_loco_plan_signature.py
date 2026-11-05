"""Plan signature / novelty and the level-2 plan edits of the
loco_manipulation example (pure Python; no faro import)."""

import sys
from pathlib import Path

LOCO_DIR = Path(__file__).parents[1] / "examples" / "loco_manipulation"
sys.path.insert(0, str(LOCO_DIR))

from plan_signature import (  # noqa: E402
    family_stats,
    plan_novelty,
    plan_signature,
    token_distance,
)
from sweep_to_variants import edit_plan, variant_catalogue  # noqa: E402

FLOOR = (1, 1)
BOX_ON_FLOOR = (6, 1, 1)


def _walk(plan, n_steps):
    last_l, last_r = plan["left_sole"][-1], plan["right_sole"][-1]
    for i in range(n_steps):
        upd = {"left_sole": last_l, "right_sole": ()} if i % 2 == 0 else {
            "left_sole": (), "right_sole": last_r}
        for ch, seq in plan.items():
            seq.append(upd.get(ch, seq[-1]))
    for ch, seq in plan.items():
        seq.append({"left_sole": last_l, "right_sole": last_r}.get(ch, seq[-1]))


def _append(plan, **upd):
    for ch, seq in plan.items():
        seq.append(upd.get(ch, seq[-1]))


def _pick_plan(n_steps, grasp=(9, 11)):
    plan = {"left_sole": [FLOOR], "right_sole": [FLOOR], "left_ee": [()],
            "right_ee": [()], "box": [BOX_ON_FLOOR]}
    _walk(plan, n_steps)
    _append(plan, left_ee=(grasp[0], 1), right_ee=(grasp[1], 1))
    _append(plan, box=())
    return plan


def test_signature_ignores_step_counts():
    assert plan_signature(_pick_plan(2)) == plan_signature(_pick_plan(5))
    assert plan_signature(_pick_plan(2)).startswith("walk|")


def test_signature_distinguishes_grasp_faces():
    assert plan_signature(_pick_plan(2)) != plan_signature(_pick_plan(2, grasp=(10, 12)))


def test_step_up_is_not_a_walk_step():
    plan = {"left_sole": [FLOOR], "right_sole": [FLOOR]}
    _append(plan, right_sole=())
    _append(plan, right_sole=(2, 1))  # plants on a new surface
    tokens = plan_signature(plan).split("|")
    assert tokens[0] == "walk"  # lifting is a step...
    assert tokens[1] == "right_sole:->2.1"  # ...landing elsewhere is not


def test_novelty_and_distance():
    a = plan_signature(_pick_plan(2))
    b = plan_signature(_pick_plan(2, grasp=(10, 12)))
    assert plan_novelty(a, []) == 1.0
    assert plan_novelty(a, [a]) == 0.0
    assert 0.0 < plan_novelty(b, [a]) <= 1.0
    assert token_distance(["x"], ["x", "y"]) == 0.5


def test_family_stats_groups_and_ranks():
    recs = [
        {"signature": "A", "combined_score": 0.2},
        {"signature": "A", "combined_score": 0.7},
        {"signature": "B", "combined_score": 0.5},
        {"signature": None, "combined_score": 1.0},
    ]
    fams = family_stats(recs)
    assert [(f["signature"], f["count"], f["best_score"]) for f in fams] == [
        ("A", 2, 0.7), ("B", 1, 0.5)]


def test_walk_edits_keep_strategy_and_change_length():
    base = _pick_plan(4)
    n = len(base["left_sole"])
    plus, minus = edit_plan(base, "walk+1"), edit_plan(base, "walk-1")
    assert len(plus["left_sole"]) == n + 2
    assert len(minus["left_sole"]) == n - 2
    for edited in (plus, minus):
        assert plan_signature(edited) == plan_signature(base)
        assert len({len(seq) for seq in edited.values()}) == 1
        for k in range(len(edited["left_sole"]) - 1):  # no no-op modes
            assert any(seq[k] != seq[k + 1] for seq in edited.values())


def test_walk_minus_needs_a_long_enough_walk():
    assert edit_plan(_pick_plan(1), "walk-1") is None


def test_mirror_swaps_left_and_right():
    base = _pick_plan(2)
    mirrored = edit_plan(base, "mirror")
    assert mirrored["left_ee"] == base["right_ee"]
    assert mirrored["right_sole"] == base["left_sole"]
    assert mirrored["box"] == base["box"]


def test_variant_catalogue_groups():
    names = {v["name"] for v in variant_catalogue(["timing", "plan"])}
    assert "reference" in names
    assert {"plan_mirror", "timing_tpm0.7"} <= names
    assert not any(n.startswith("start_") for n in names)


def test_patches_ignore_order_but_not_faces():
    from plan_signature import patch_novelty, plan_patches

    seq = _pick_plan(2)
    simultaneous = {ch: list(v) for ch, v in seq.items()}
    # same faces, hands made contact one after the other instead
    staggered = {"left_sole": [FLOOR], "right_sole": [FLOOR], "left_ee": [()],
                 "right_ee": [()], "box": [BOX_ON_FLOOR]}
    _append(staggered, left_ee=(9, 1))
    _append(staggered, right_ee=(11, 1))
    _append(staggered, box=())
    assert plan_signature(staggered) != plan_signature(simultaneous)
    assert plan_patches(staggered) == plan_patches(simultaneous)
    other_faces = _pick_plan(2, grasp=(10, 12))
    assert plan_patches(other_faces) != plan_patches(simultaneous)
    assert patch_novelty(plan_patches(simultaneous), [plan_patches(staggered)]) == 0.0
    assert patch_novelty(plan_patches(other_faces), [plan_patches(simultaneous)]) > 0.3


def test_combined_novelty_weights_patches_higher():
    from plan_signature import combined_novelty

    assert combined_novelty(1.0, 0.0) > combined_novelty(0.0, 1.0)


def test_hand_change_during_a_step_shares_the_sequential_signature():
    from plan_signature import plan_signature

    # Grasp, then two walking steps.
    sequential = {"left_sole": [(1, 1), (1, 1), (1, 1), (), (1, 1)],
                  "right_sole": [(1, 1), (1, 1), (), (1, 1), (1, 1)],
                  "left_ee": [(), (3, 1), (3, 1), (3, 1), (3, 1)]}
    # The same grasp made in the mode where the right foot lifts.
    simultaneous = {"left_sole": [(1, 1), (1, 1), (), (1, 1)],
                    "right_sole": [(1, 1), (), (1, 1), (1, 1)],
                    "left_ee": [(), (3, 1), (3, 1), (3, 1)]}
    assert plan_signature(sequential) == plan_signature(simultaneous) == "left_ee:->3.1|walk"


def test_patch_novelty_ignores_contacts_every_plan_shares():
    from plan_signature import patch_novelty, plan_patches, shared_patch_tokens

    shared = shared_patch_tokens({"left_sole": (1, 1), "right_sole": (1, 1), "box": (6, 2, 1)},
                                 {"box": (6, 1, 1)})
    prior = plan_patches({"left_sole": [(1, 1)], "right_sole": [(1, 1)], "left_ee": [(3, 1)],
                          "right_ee": [(5, 1)], "box": [(6, 2, 1), (6, 1, 1)]})
    new = plan_patches({"left_sole": [(1, 1)], "right_sole": [(1, 1)], "left_ee": [(4, 1)],
                        "right_ee": [(6, 1)], "box": [(6, 2, 1), (6, 1, 1)]})
    assert patch_novelty(new, [prior]) < 0.6          # diluted by the shared contacts
    assert patch_novelty(new, [prior], ignore=shared) == 1.0
