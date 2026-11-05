"""Registering generated scenes (loco_manipulation/register_generated.py)
and the scene campaign's pure parts (scene_campaign.py): the goals.py
rewrite, scene ordering, unique valid plans, summaries."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import register_generated as rg  # noqa: E402
import scene_campaign as sc  # noqa: E402

GOALS_A = """from faro.search.goal import DiscreteGoal, TerminalGoal
from .initial_conditions import goal_placements

def task_contacts(scene):
    contacts = {('obj', name): (6, scene.face_id(*placement.on), 1)
                for name, placement in goal_placements().items()}
    contacts.update({('ee', 'robot', hand): () for hand in ('left_ee', 'right_ee')})
    return DiscreteGoal(discrete=contacts)

def task_terminal(scene, resolved_state):
    return TerminalGoal.from_placements(scene, resolved_state, goal_placements(),
                                        kind='pose', kso='hard', to='hard')

DISCRETE_GOALS = {'task': task_contacts}
TERMINAL_GOALS = {'task': task_terminal}
"""
GOALS_B = GOALS_A.replace("from faro.search.goal import DiscreteGoal, TerminalGoal\n",
                          "from faro.search.goal import DiscreteGoal, TerminalGoal\nfrom faro.utils.geom_utils import BOX_FACES\n") \
                 .replace("(6, scene.face_id", "(BOX_FACES.index(placement.self_face) + 1, scene.face_id")


@pytest.mark.parametrize("source", [GOALS_A, GOALS_B])
def test_transform_goals_keeps_the_discrete_goal_and_rewrites_the_terminal_one(source):
    out = rg.transform_goals(source)
    assert "from dataclasses import replace" in out and "GoalTolerances" in out
    assert "GOAL_TOLERANCE_M = 0.05" in out and "GOAL_ROTATION_TOLERANCE_RAD = 0.1" in out
    assert "kind='pose', kso='hard', to='hard'" in out
    assert "GoalTolerances(position=GOAL_TOLERANCE_M, rotation=GOAL_ROTATION_TOLERANCE_RAD)" in out
    assert "kso_tolerance=tol, to_tolerance=tol" in out
    assert "def task_score(scene, resolved_state):" in out
    assert "OUTCOME_SCORERS = {'task': task_score}" in out
    assert "TERMINAL_GOALS = {'task': task_terminal}" in out and "DISCRETE_GOALS = {'task': task_contacts}" in out
    # task_contacts is untouched (face index expression included).
    start, end = source.index("def task_contacts"), source.index("def task_terminal")
    assert source[start:end] in out
    if "BOX_FACES" in source:
        assert "from faro.utils.geom_utils import BOX_FACES" in out
    compile(out, "goals.py", "exec")   # valid Python


def test_transform_goals_rejects_other_shapes():
    with pytest.raises(ValueError):
        rg.transform_goals("def task_terminal(scene, resolved_state):\n    return None\n")
    with pytest.raises(ValueError):
        rg.transform_goals(GOALS_A.replace("to='hard'", "to='soft'"))


def test_register_copies_only_the_scene_files(tmp_path):
    examples = tmp_path / "examples"
    src = examples / "demo_001" / "source"
    src.mkdir(parents=True)
    (src / "scene.py").write_text("SCENE = 1\n")
    (src / "initial_conditions.py").write_text("IC = 1\n")
    (src / "goals.py").write_text(GOALS_A)
    (examples / "demo_001" / "spec.json").write_text(json.dumps({"id": "demo_001", "boxes": []}))
    (examples / "demo_001" / "plans").mkdir()
    (examples / "demo_001" / "plans" / "p.json").write_text("{}")
    dest = tmp_path / "scenes"
    assert rg.register(str(examples), "demo_001", dest_root=str(dest)) == ("demo_001", "registered")
    files = sorted(p.name for p in (dest / "demo_001").iterdir())
    assert files == ["goals.py", "initial_conditions.py", "scene.py", "spec.json"]
    assert rg.register(str(examples), "demo_001", dest_root=str(dest)) == ("demo_001", "skipped")
    assert rg.register(str(examples), "demo_001", dest_root=str(dest), force=True, dry_run=True)[1].startswith("registered")
    (src / "goals.py").write_text("nothing here\n")
    assert rg.register(str(examples), "demo_001", dest_root=str(dest), force=True)[1].startswith("invalid")
    assert rg.scene_ids(str(examples), None) == ["demo_001"]
    with pytest.raises(SystemExit):
        rg.scene_ids(str(examples), ["missing_001"])


def test_campaign_orders_easy_first_and_counts_unique_valid_plans(tmp_path):
    specs = [("generated.b", {"id": "b", "boxes": [1, 2], "difficulty": {"terrain": 0, "manipulation": 3}}),
             ("generated.a", {"id": "a", "boxes": [1], "difficulty": {"terrain": 2, "manipulation": 1}}),
             ("generated.c", {"id": "c", "boxes": [1, 2], "difficulty": {"terrain": 0, "manipulation": 1}})]
    assert [s for s, _ in sorted(specs, key=lambda s: sc.order_key(s[1]))] == ["generated.a", "generated.c", "generated.b"]
    run = tmp_path / "run"
    plan = {"left_sole": [[1, 1], [1, 1]]}
    for gen, valid, p in (("gen_1", True, plan), ("gen_2", True, plan), ("gen_3", True, {"left_sole": [[1, 1]]}), ("gen_4", False, plan)):
        d = run / gen / "results"
        d.mkdir(parents=True)
        (d / "extra.json").write_text(json.dumps({"to_valid": valid, "contact_mode_seq": p}))
        (d / "metrics.json").write_text(json.dumps({"combined_score": 0.12}))
    found = sc.valid_plans(str(run))
    assert len(found) == 2 and sorted(v["gen"] for v in found.values()) == ["gen_1", "gen_3"]
    assert sc.evaluations(str(run)) == 4 and sc.nodes(str(run)) == 0
    recs = {"generated.a": {"scene": "generated.a", "boxes": 1, "difficulty": {"terrain": 2, "manipulation": 1},
                            "outcome": "solved", "valid": 5, "plans": [{"gen": "gen_1", "modes": 71, "score": 0.1}],
                            "evaluations": 9, "hours": 2.5}}
    sc.write_csv(recs, str(tmp_path / "s.csv"))
    assert "generated.a,1,2,1,solved,5,71,9,2.5" in (tmp_path / "s.csv").read_text()
    (tmp_path / "s.jsonl").write_text(json.dumps(recs["generated.a"]) + "\n")
    assert sc.load_summary(str(tmp_path / "s.jsonl"))["generated.a"]["valid"] == 5
    assert sc.load_summary(str(tmp_path / "missing.jsonl")) == {}
    # solved / capped / budget scenes are left alone unless --redo; failures without a verdict are retried
    assert not sc.should_run({"outcome": "solved"}) and not sc.should_run({"outcome": "capped"}) and not sc.should_run({"outcome": "budget"})
    assert sc.should_run(None) and sc.should_run({"outcome": "codegen_failed"}) and sc.should_run({"outcome": "capped"}, redo=True)


def test_search_env_sets_the_knobs(monkeypatch):
    monkeypatch.delenv("RESUME", raising=False)
    args = SimpleNamespace(out="/o", nodes=40, eval_jobs=2, job_time="02:00:00", leg_margin=0.02,
                           margin_capsules=sc.LEG_CAPSULES)
    env = sc.search_env("generated.x", "desc", args, None)
    assert env["SCENES"] == "generated.x" and env["DESCRIPTION"] == "desc" and env["MAX_EVAL_JOBS"] == "2"
    # Quotes would end Hydra's task.description='...' override early.
    assert sc.safe_description("the rack's \"upper\" slab\\\n leaves  75 mm") == "the rack’s ”upper” slab/ leaves 75 mm"
    assert "'" not in sc.search_env("generated.x", "it's", args, None)["DESCRIPTION"]
    assert env["SHINKA_COLLISION_MARGIN"] == "0.02" and env["SHINKA_TO_STALL_ITERS"] == "30"
    # Not the feet: a foot margin makes standing on a box support infeasible.
    assert env["SHINKA_COLLISION_MARGIN_CAPSULES"] == "l_shin,r_shin"
    assert "RESUME" not in env
    env = sc.search_env("generated.x", "desc", args, 20)
    assert env["RESUME"] == "1" and env["RESUME_EXTRA"] == "20"
    args.leg_margin = 0.0
    assert "SHINKA_COLLISION_MARGIN" not in sc.search_env("generated.x", "desc", args, None)
    assert sc.codegen_dir("generated.x_001").endswith("moto_gen/generated_x_001")


def test_explicit_scenes_keep_their_order_and_count_movable_boxes(tmp_path):
    tamp, local = tmp_path / "tamp", tmp_path / "local"
    (tamp / "motiondisco" / "long_dist_pick_place").mkdir(parents=True)
    (tamp / "motiondisco" / "long_dist_pick_place" / "scene.py").write_text(
        "objects={'floor': Plane(), 'table': FixedBox(), 'box': MovableBox(half_extents=[1,1,1], mass=1)}\n")
    (local / "floor_stack_carry").mkdir(parents=True)
    (local / "floor_stack_carry" / "scene.py").write_text("MovableBox(...)\nMovableBox(...)\n")
    specs = sc.explicit_specs(["floor_stack_carry", "motiondisco.long_dist_pick_place", "motiondisco.missing"],
                              roots=(str(tamp), str(local)))
    assert [s for s, _ in specs] == ["floor_stack_carry", "motiondisco.long_dist_pick_place", "motiondisco.missing"]
    assert [len(spec["boxes"]) for _, spec in specs] == [2, 1, 0]
    assert all(spec["description"] is None and spec["difficulty"] is None for _, spec in specs)
    assert sc.safe_description(str(specs[0][1]["description"] or "")) == ""     # the ablation script's table applies
    args = SimpleNamespace(out="/o", nodes=40, eval_jobs=2, job_time="02:00:00", leg_margin=0.0,
                           margin_capsules=sc.LEG_CAPSULES, to_goal="hard", to_goal_tol=0.05)
    env = sc.search_env("floor_stack_carry", "", args, None)
    assert env["SHINKA_TO_GOAL"] == "hard" and env["SHINKA_TO_GOAL_TOL"] == "0.05" and env["DESCRIPTION"] == ""
    args.to_goal = None
    assert "SHINKA_TO_GOAL" not in sc.search_env("floor_stack_carry", "", args, None)


def test_dataset_follower_command_and_backlog(tmp_path):
    args = SimpleNamespace(dataset_out="/data", dataset_samples=10, dataset_factors=sc.DATASET_FACTORS, dataset_jobs=2,
                           dataset_range=None)
    cmd = sc.dataset_command(args, "/runs/r1")
    assert cmd[1].endswith("generate_dataset.py") and cmd[cmd.index("--run-dir") + 1] == "/runs/r1"
    assert cmd[cmd.index("--out") + 1] == "/data" and cmd[cmd.index("--samples") + 1] == "10"
    assert cmd.count("--range") == 2 and "mass_kg=0.5,10" in cmd and "size=0.9,1.1" in cmd
    args.dataset_range = ["mass_kg=1,2"]
    assert sc.dataset_command(args, "/runs/r1").count("--range") == 1
    records = {"a": {"outcome": "solved", "run_dir": "/x/a/2026.11.02103400_diverse"},
               "b": {"outcome": "solved", "run_dir": "/x/b/2026.11.01120000_diverse"},
               "c": {"outcome": "capped", "run_dir": "/x/c/2026.11.02110000_diverse"},
               "d": {"outcome": "solved", "run_dir": "/x/d/2026.11.02120000_diverse"}}
    assert sc.dataset_backlog(records, "2026.11.02") == [("a", "/x/a/2026.11.02103400_diverse"), ("d", "/x/d/2026.11.02120000_diverse")]
    assert {s for s, _ in sc.dataset_backlog(records, None)} == {"a", "b", "d"}
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"run_dir": "/x/d/2026.11.02120000_diverse"}) + "\n\n")
    assert sc.manifest_runs(str(tmp_path)) == ["/x/d/2026.11.02120000_diverse"]
    assert sc.manifest_runs(str(tmp_path / "missing")) == []
    assert "SHINKA_TO_INCREMENTAL" in sc.DATASET_ENV and sc.DATASET_ENV["SHINKA_TO_INCREMENTAL"] == "0"
    assert sc.DATASET_ENV["SHINKA_TO_SEGMENT_MODES"] == "13" and "DIVERSE" not in sc.DATASET_ENV
    assert sc.scoped(["x"], SimpleNamespace(cpus=8, mem="32G"), cpus=3, mem="24G")[4] == "CPUQuota=300%"


def test_warm_codegen_retries_a_crashed_compile(monkeypatch, tmp_path):
    """A gcc internal compiler error fails one attempt; the next resumes."""
    monkeypatch.setattr(sc, "MOTO_GEN", str(tmp_path))
    args = SimpleNamespace(cpus=2, mem="1G")
    outputs = iter(["", "WARM_DONE status=0,0 time=1s\n"])
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=next(outputs), stderr="internal compiler error\n")

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    lines = []
    assert sc.warm_codegen("generated.x", args, lines.append)
    assert len(calls) == 2 and any("retrying codegen (2/3)" in ln for ln in lines)
    outputs = iter([""] * sc.CODEGEN_ATTEMPTS)
    assert not sc.warm_codegen("generated.x", args, lines.append)
    assert len(calls) == 2 + sc.CODEGEN_ATTEMPTS
    (tmp_path / "generated_x" / ".moto_artifacts").mkdir(parents=True)
    assert sc.warm_codegen("generated.x", args, lines.append) and len(calls) == 2 + sc.CODEGEN_ATTEMPTS
