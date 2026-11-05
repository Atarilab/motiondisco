"""Per-run cache of solved TO windows (loco_manipulation/segment_cache.py):
keys that depend on the plan prefix, the start state and the settings,
and atomic put / get."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "loco_manipulation"))

import segment_cache as sc  # noqa: E402
import to_segments as ts  # noqa: E402

PLAN = {"left_sole": [(1, 1), (), (1, 1), (1, 1)], "right_sole": [(1, 1), (1, 1), (), (1, 1)],
        "box": [(6, 1, 1), (6, 1, 1), (6, 1, 1), ()]}


def _start(x=0.0):
    return ts.SegmentStart(robot_x=np.r_[x, np.zeros(9)], obj_x={"box": np.arange(13.0)})


def test_key_depends_on_the_prefix_the_start_and_the_settings():
    settings = {"npm": 20, "tol": 0.1}
    k = sc.key(PLAN, 3, None, settings)
    other_suffix = {ch: seq[:3] + [()] for ch, seq in PLAN.items()}
    assert sc.key(other_suffix, 3, None, settings) == k                   # the suffix is not part of the key
    assert sc.key(other_suffix, 4, None, settings) != sc.key(PLAN, 4, None, settings)
    assert sc.key(PLAN, 3, _start(), settings) != k                        # a start state is
    assert sc.key(PLAN, 3, _start(1e-9), settings) == sc.key(PLAN, 3, _start(), settings)   # within rounding
    assert sc.key(PLAN, 3, _start(1e-3), settings) != sc.key(PLAN, 3, _start(), settings)
    assert sc.key(PLAN, 3, None, {"npm": 20, "tol": 0.05}) != k            # and the settings
    listed = {ch: [list(e) for e in seq] for ch, seq in PLAN.items()}     # lists hash like tuples
    assert sc.key(listed, 3, None, settings) == k


def test_put_and_get_round_trip_and_missing_entries(tmp_path):
    cache = sc.Cache(str(tmp_path / "c"))
    assert cache.get("nope") is None and len(cache) == 0
    rec = {"outcome": ts.SegmentOutcome((0, 4), status=0), "part": {"robot_xs": {"r": np.ones((3, 2))}},
           "rows": [{"node": 1}], "end": _start(2.0)}
    cache.put("abc", rec)
    back = cache.get("abc")
    assert back["outcome"].modes == (0, 4) and back["end"].robot_x[0] == 2.0 and back["rows"] == [{"node": 1}]
    assert len(cache) == 1 and not [p for p in (tmp_path / "c").iterdir() if p.name.startswith(".tmp_")]
    (tmp_path / "c" / "bad.pkl").write_bytes(b"not a pickle")
    assert cache.get("bad") is None


def test_open_cache_uses_the_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(sc.CACHE_DIR_ENV, raising=False)
    assert sc.open_cache() is None
    monkeypatch.setenv(sc.CACHE_DIR_ENV, str(tmp_path / "run" / "segment_cache"))
    cache = sc.open_cache()
    assert cache is not None and Path(cache.dir).is_dir()
    assert sc.settings_digest({"a": 1, "b": [2]}) == sc.settings_digest({"b": [2], "a": 1})
    with pytest.raises(AttributeError):      # a start must carry robot_x / obj_x
        sc.key(PLAN, 3, object(), {})
