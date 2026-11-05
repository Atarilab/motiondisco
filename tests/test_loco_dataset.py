"""RL-tracker dataset generation (loco_manipulation): deterministic
sampling, plan collection, export bookkeeping (solver and faro stubbed)."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import dataset_export as dx  # noqa: E402
import generate_dataset as gd  # noqa: E402

PLAN = {"left_sole": [[1, 1]] * 3, "right_sole": [[1, 1], [], [1, 1]],
        "left_ee": [[], [3, 1], [3, 1]], "right_ee": [[]] * 3,
        "box": [[6, 2, 1], [], [6, 1, 1]], "lid": [[6, 9, 1]] * 3}


def test_samples_are_deterministic_and_in_range():
    r = gd.Ranges()
    a = gd.plan_samples("run", "h" * 64, ["box", "lid"], 3, 7, gd.ALL_FACTORS, r)
    b = gd.plan_samples("run", "h" * 64, ["box", "lid"], 3, 7, gd.ALL_FACTORS, r)
    assert a == b
    assert a[0] == {"sample_id": "s000_reference", "seed": None, "factors": {}}
    assert [s["sample_id"] for s in a] == ["s000_reference", "s001", "s002", "s003"]
    other = gd.plan_samples("run", "g" * 64, ["box", "lid"], 3, 7, gd.ALL_FACTORS, r)
    assert other[1]["factors"] != a[1]["factors"]
    for s in a[1:]:
        f = s["factors"]
        assert set(f) == {"mass_scale", "mass_kg", "size_scale", "robot_offset", "object_offsets",
                          "time_per_mode", "weight_scale"}
        assert all(r.mass[0] <= v <= r.mass[1] for v in f["mass_scale"].values())
        assert all(r.mass_kg[0] <= v <= r.mass_kg[1] for v in f["mass_kg"].values())
        assert all(r.size[0] <= v <= r.size[1] for vs in f["size_scale"].values() for v in vs)
        assert r.time_per_mode[0] <= f["time_per_mode"] <= r.time_per_mode[1]
        assert all(r.weights[0] <= v <= r.weights[1] for v in f["weight_scale"].values())
    json.dumps(a)  # job files must be JSON


def test_factor_selection_and_objects():
    assert gd.plan_objects(PLAN) == ["box", "lid"]
    s = gd.sample_factors(np.random.default_rng(0), ["box"], ["mass", "timing"], gd.Ranges())
    assert set(s) == {"mass_scale", "time_per_mode"}
    s = gd.sample_factors(np.random.default_rng(0), ["box"], ["mass_kg"], gd.Ranges(mass_kg=(0.5, 10.0)))
    assert set(s) == {"mass_kg"} and 0.5 <= s["mass_kg"]["box"] <= 10.0


def test_parse_ranges():
    r = gd._parse_ranges(["mass=0.8,1.2", "time-per-mode=1,2", "mass_kg=0.5,10"])
    assert r.mass == (0.8, 1.2) and r.time_per_mode == (1.0, 2.0) and r.mass_kg == (0.5, 10.0)
    with pytest.raises(SystemExit):
        gd._parse_ranges(["friction=0.3,0.8"])


def _write_gen(run, label, plan, converged, max_vio):
    d = run / label / "results"
    d.mkdir(parents=True)
    (d / "extra.json").write_text(json.dumps({
        "contact_mode_seq": plan, "to_converged": converged, "to_cost": 1.0,
        "to_cost_parts": {"constraint_violation_max": max_vio}}))


def test_collect_plans_filters_dedupes_and_caps_families(tmp_path):
    other = {**PLAN, "left_ee": [[], [4, 1], [4, 1]]}
    _write_gen(tmp_path, "gen_1", PLAN, False, 0.05)       # near-feasible
    _write_gen(tmp_path, "gen_2", PLAN, True, 0.01)        # same plan, converged: wins
    _write_gen(tmp_path, "gen_3", other, False, 0.5)       # too infeasible
    _write_gen(tmp_path, "gen_4", {**other, "lid": [[6, 9, 2]] * 3}, True, 0.02)
    plans = gd.collect_plans(str(tmp_path), near_feasible=0.1, max_per_family=5)
    by_gen = {p["gen"]: p for p in plans}
    assert set(by_gen) == {"gen_2", "gen_4"}
    assert by_gen["gen_2"]["converged"]
    assert gd.collect_plans(str(tmp_path), near_feasible=0.01, max_per_family=5)[0]["gen"] == "gen_2"


def test_classify():
    assert dx.classify(True, 5.0, 0.1) == "converged"
    assert dx.classify(False, 0.05, 0.1) == "near_feasible"
    assert dx.classify(False, 0.5, 0.1) is None
    assert dx.classify(None, None, 0.1) is None
    # with a physical check, it replaces the solver-residual rule
    assert dx.classify(False, 3.0, 0.1, physically_valid=True) == "physical"
    assert dx.classify(False, 0.05, 0.1, physically_valid=False) is None
    assert dx.classify(True, 3.0, 0.1, physically_valid=False) == "converged"


def _stub_scene():
    box = types.SimpleNamespace(mass=2.0, inertia=np.array([0.1, 0.1, 0.1]),
                                half_extents=(0.2, 0.2, 0.15))
    table = types.SimpleNamespace(half_extents=(0.5, 0.5, 0.4))
    robot = types.SimpleNamespace(joint_names=["j1", "j2"])
    scene = types.SimpleNamespace(env_objects={"box": box}, fixed_env_objects={"table": table},
                                  robots={"robot": robot})
    st = lambda q: types.SimpleNamespace(q=np.array(q, float), v=np.zeros(2))  # noqa: E731
    resolved = types.SimpleNamespace(
        env_objects={"box": st([0.5, 0, 0.15, 0, 0, 0, 1])},
        fixed_env_objects={"table": st([2.5, 0, 0.4, 0, 0, 0, 1])},
        robots={"robot": st([0, 0, 0.77, 0, 0, 0, 1, 0.1, 0.2])})
    return scene, resolved


def test_physics_record_and_export(tmp_path):
    scene, resolved = _stub_scene()
    phys = dx.physics_record(scene, resolved)
    assert phys["objects"]["box"]["mass"] == 2.0
    assert phys["objects"]["box"]["half_extents"] == [0.2, 0.2, 0.15]
    assert phys["fixed_objects"]["table"]["q_xyzw"][0] == 2.5
    assert phys["robots"]["robot"]["joint_names"] == ["j1", "j2"]
    assert phys["friction_mu"] == dx.FRICTION_MU

    results = {"step_ts": np.full(4, 0.05), "t_scale": np.ones(4),
               "robot_xs": {"robot": np.zeros((5, 3))}, "torques": {"robot": np.zeros((4, 2))},
               "rob_Ws": {"robot": {"left_ee": np.zeros((4, 6))}},
               "obj_xs": {"box": np.zeros((5, 13))}, "obj_Ws": {"box": np.zeros((4, 6))}}
    seen = {}

    def to_mj(scene_, results_, time_, cms, num_modes, nodes_per_mode, frequency):
        seen.update(time=np.asarray(time_), num_modes=num_modes, frequency=frequency)
        return {"time": np.linspace(0, 0.2, 11), "schema_version": np.array(2)}

    def scene_xml(scene_, resolved_, path):
        Path(path).write_text("<scene/>")

    params = {"scene": "s", "physics": phys}
    paths = dx.export_sample(str(tmp_path / "s001"), scene, resolved, {}, results, params,
                             robot_id="robot", num_modes=2, nodes_per_mode=2, frequency=50,
                             to_mj=to_mj, scene_xml=scene_xml)
    assert seen["time"] == pytest.approx([0, 0.05, 0.1, 0.15, 0.2])
    with np.load(paths["npz"], allow_pickle=False) as z:
        keys = set(z.files)
        assert {"time", "solver_time", "solver_torques", "solver_wrench_left_ee",
                "solver_obj_xs_box", "solver_obj_wrench_box", "params_json"} <= keys
        assert json.loads(str(z["params_json"]))["scene"] == "s"
    assert json.loads(Path(paths["params"]).read_text())["physics"]["objects"]["box"]["mass"] == 2.0
    assert Path(paths["scene_xml"]).read_text() == "<scene/>"


def test_manifest_resume_and_summary(tmp_path):
    out = str(tmp_path)
    base = {"run_id": "r", "plan_hash": "h", "scene": "s", "initial_condition": "default"}
    dx.append_manifest(out, {**base, "sample_id": "s000_reference", "quality": "converged",
                             "skipped": "", "error": None})
    dx.append_manifest(out, {**base, "sample_id": "s001", "quality": None,
                             "skipped": "KSO infeasible", "error": None})
    dx.append_manifest(out, {**base, "sample_id": "s002", "quality": "near_feasible",
                             "skipped": "", "error": None})
    assert dx.done_keys(out) == {("r", "h", "s000_reference"), ("r", "h", "s001"), ("r", "h", "s002")}
    # a sample that ended in an exception is retried; its retry's row wins
    dx.append_manifest(out, {**base, "sample_id": "s003", "quality": None, "skipped": "", "error": "ValueError: x"})
    assert ("r", "h", "s003") not in dx.done_keys(out)
    dx.append_manifest(out, {**base, "sample_id": "s003", "quality": None, "skipped": "KSO infeasible", "error": None})
    assert ("r", "h", "s003") in dx.done_keys(out)
    (row,) = dx.write_summary(out)
    assert (row["attempted"], row["exported"], row["converged"], row["near_feasible"],
            row["kso_infeasible"]) == (4, 2, 1, 1, 2)
    assert (tmp_path / dx.SUMMARY).read_text().startswith("scene,initial_condition")
