"""
Reconstruct the feedback each LLM proposal saw during evolution.

Usage:
    python inspect_feedback.py <path/to/programs.sqlite>                    # all programs
    python inspect_feedback.py <path/to/programs.sqlite> --id <program_id>  # one program
    python inspect_feedback.py <path/to/programs.sqlite> --gen 5            # one generation
    python inspect_feedback.py <path/to/programs.sqlite> --top 10           # top-N by score
    python inspect_feedback.py <path/to/programs.sqlite> --show-code        # also print code

The "feedback" shown is the exact string injected into the prompt template
as {performance_metrics} + {text_feedback_section} for each program when
it was used as a parent in a later iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.prompts.prompts_base import perf_str, format_text_feedback_section


def load_extra_for_program(run_root: Path, prog) -> Optional[dict]:
    """Load the per-generation extra.json written by evaluate.py."""
    extra = run_root / f"gen_{prog.generation}" / "results" / "extra.json"
    if not extra.exists():
        return None
    try:
        return json.loads(extra.read_text())
    except Exception:
        return None


def format_contact_mode_seq(cms: dict) -> str:
    """
    Render a contact_mode_seq dict as Python-source style:

        mode_interface_dict = {
            "left_ee": [(), (9, 1), ()],
            "box1":    [(6, 1, 1), (), (6, 2, 1)],
        }

    Empty list entries become () tuples; non-empty ones become (a, b, ...).
    """
    lines = ["mode_interface_dict = {"]
    for channel, seq in cms.items():
        entries = []
        for entry in seq:
            if not entry:
                entries.append("()")
            else:
                entries.append("(" + ", ".join(str(x) for x in entry) + ")")
        lines.append(f'    "{channel}": [{", ".join(entries)}],')
    lines.append("}")
    return "\n".join(lines)


def render(prog, run_root: Path, show_code: bool = False) -> str:
    header = (
        f"\n{'=' * 72}\n"
        f"Program id:   {prog.id}\n"
        f"Generation:   {prog.generation}\n"
        f"Parent id:    {prog.parent_id}\n"
        f"Correct:      {prog.correct}\n"
        f"Score:        {prog.combined_score:.4f}\n"
        f"{'-' * 72}"
    )
    feedback = perf_str(prog.combined_score, prog.public_metrics)
    text_fb = format_text_feedback_section(prog.text_feedback)
    body = f"\n[performance_metrics block]\n{feedback}{text_fb}"

    extra = load_extra_for_program(run_root, prog)
    if extra is not None and "contact_mode_seq" in extra:
        body += "\n\n[contact_mode_seq fed to IK]\n"
        body += format_contact_mode_seq(extra["contact_mode_seq"])
    else:
        body += "\n\n[contact_mode_seq: extra.json not found for this program]"

    if show_code:
        body += f"\n\n[code]\n{prog.code}\n"
    return header + body


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("db_path", help="Path to programs.sqlite")
    p.add_argument("--id", help="Show a single program by id")
    p.add_argument("--gen", type=int, help="Show all programs from a generation")
    p.add_argument("--top", type=int, help="Show top-N by combined_score")
    p.add_argument("--show-code", action="store_true", help="Also print the code")
    args = p.parse_args()

    run_root = Path(args.db_path).resolve().parent

    db = ProgramDatabase(
        config=DatabaseConfig(db_path=args.db_path),
        embedding_model=None,  # read-only: no embedding client is built
        read_only=True,
    )

    if args.id:
        prog = db.get(args.id)
        if prog is None:
            print(f"No program found with id {args.id!r}", file=sys.stderr)
            sys.exit(1)
        progs = [prog]
    elif args.gen is not None:
        progs = db.get_programs_by_generation(args.gen)
    elif args.top is not None:
        progs = db.get_top_programs(n=args.top)
    else:
        progs = db.get_all_programs()

    if not progs:
        print("No programs matched the filters.", file=sys.stderr)
        sys.exit(1)

    for prog in progs:
        print(render(prog, run_root=run_root, show_code=args.show_code))

    print(f"\n{'=' * 72}\n{len(progs)} program(s) shown.\n")


if __name__ == "__main__":
    main()
