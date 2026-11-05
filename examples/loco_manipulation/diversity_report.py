"""
Post-run diversity report for a loco_manipulation search.

    python examples/loco_manipulation/diversity_report.py results/<exp>/<scene>/<run>

Reads the run's SQLite program archive read-only and prints, per island
and overall: programs, valid plans (score > 0), distinct contact-patch
cells (``plan_patches``, the archive's descriptor key), distinct strategy
orders (``plan_signature``), and how many children were mutations of a
generation-0 copy; the archive's occupancy and any score <= 0 members;
and proposal-filter rejections recorded in the program metadata. Writes
``<run>/diversity_by_gen.csv`` (cumulative distinct cells / orders by
generation) unless ``--no-csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List


def _load(run_dir: str) -> Dict[str, Any]:
    db_path = next(
        (os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith(".sqlite")), None,
    )
    if db_path is None:
        raise SystemExit(f"no *.sqlite in {run_dir}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    programs: List[Dict[str, Any]] = []
    for row in conn.execute(
        "SELECT id, generation, island_idx, combined_score, correct, parent_id, "
        "public_metrics, private_metrics, metadata FROM programs ORDER BY generation"
    ):
        rec = dict(row)
        for key in ("public_metrics", "private_metrics", "metadata"):
            try:
                rec[key] = json.loads(rec[key]) if rec[key] else {}
            except json.JSONDecodeError:
                rec[key] = {}
        programs.append(rec)
    archive = {r[0] for r in conn.execute("SELECT program_id FROM archive")}
    conn.close()
    return {"programs": programs, "archive": archive}


def _metric(rec: Dict[str, Any], key: str) -> Any:
    value = (rec.get("public_metrics") or {}).get(key)
    if value is None:
        value = (rec.get("private_metrics") or {}).get(key)
    return value


def report(run_dir: str, write_csv: bool = True) -> None:
    data = _load(run_dir)
    programs, archive = data["programs"], data["archive"]
    by_id = {p["id"]: p for p in programs}
    gen0 = {p["id"] for p in programs if p["generation"] == 0}

    islands: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"programs": 0, "valid": 0, "cells": set(), "orders": set(), "gen0_children": 0}
    )
    for p in programs:
        isl = islands[p["island_idx"]]
        isl["programs"] += 1
        score = float(p["combined_score"] or 0.0)
        if p["parent_id"] in gen0:
            isl["gen0_children"] += 1
        if score > 0:
            isl["valid"] += 1
            if _metric(p, "plan_patches"):
                isl["cells"].add(_metric(p, "plan_patches"))
            if _metric(p, "plan_signature"):
                isl["orders"].add(_metric(p, "plan_signature"))

    print(f"{run_dir}\n{len(programs)} programs, {len(islands)} islands\n")
    print(f"{'island':>6}  {'programs':>8}  {'valid':>5}  {'cells':>5}  {'orders':>6}  {'from gen0':>9}")
    for idx in sorted(islands):
        isl = islands[idx]
        print(f"{idx:>6}  {isl['programs']:>8}  {isl['valid']:>5}  {len(isl['cells']):>5}  "
              f"{len(isl['orders']):>6}  {isl['gen0_children']:>9}")
    all_cells = set().union(*(i["cells"] for i in islands.values())) if islands else set()
    all_orders = set().union(*(i["orders"] for i in islands.values())) if islands else set()
    valid = sum(i["valid"] for i in islands.values())
    print(f"\nvalid plans: {valid}; distinct contact-patch cells: {len(all_cells)}; "
          f"distinct orders: {len(all_orders)}")

    members = [by_id[i] for i in archive if i in by_id]
    cells = defaultdict(int)
    for p in members:
        if _metric(p, "plan_patches"):
            cells[_metric(p, "plan_patches")] += 1
    low = [p["id"][:8] for p in members if float(p["combined_score"] or 0.0) <= 0]
    print(f"archive: {len(members)} members, {len(cells)} cells, "
          f"{sum(1 for c in cells.values() if c > 1)} cells with >1 member, "
          f"{len(low)} members with score <= 0{': ' + ', '.join(low) if low else ''}")

    rejections = defaultdict(int)
    for p in programs:
        info = (p.get("metadata") or {}).get("proposal_filter")
        if isinstance(info, dict) and info.get("reason") and info["reason"] != "accepted":
            rejections[info["reason"].split(" (")[0].split(";")[0]] += 1
    if rejections:
        print("proposal filter rejections:")
        for reason, n in sorted(rejections.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3}  {reason}")

    if write_csv:
        path = os.path.join(run_dir, "diversity_by_gen.csv")
        seen_cells, seen_orders, n_valid = set(), set(), 0
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["generation", "valid_cum", "cells_cum", "orders_cum"])
            for p in sorted(programs, key=lambda r: r["generation"]):
                if float(p["combined_score"] or 0.0) > 0:
                    n_valid += 1
                    if _metric(p, "plan_patches"):
                        seen_cells.add(_metric(p, "plan_patches"))
                    if _metric(p, "plan_signature"):
                        seen_orders.add(_metric(p, "plan_signature"))
                w.writerow([p["generation"], n_valid, len(seen_cells), len(seen_orders)])
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()
    report(os.path.abspath(args.run_dir), write_csv=not args.no_csv)


if __name__ == "__main__":
    main()
