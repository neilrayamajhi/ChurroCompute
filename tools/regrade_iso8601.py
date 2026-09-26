"""Re-score cached iso8601-recurrence rollouts with the env's own grade().

The env's _completion_text() only reads dict messages, but verifiers 0.3.0
passes AssistantMessage objects, so every rollout was scored 0 at run time.
The saved results.jsonl has the messages as plain dicts, so feeding the
content text to grade() gives the score the env intended. No inference is
re-run. Writes corrected copies under <run_root>/regraded/ with identical
metadata.json, leaving the originals untouched.

Needs iso8601_recurrence importable (install it into the project venv with
`uv pip install iso8601-recurrence --extra-index-url
https://hub.primeintellect.ai/joeljose/iso8601-recurrence/install/simple/`).

    uv run --no-sync python tools/regrade_iso8601.py outputs/runpod/<pod-id>
    uv run python tools/collect_run.py <regraded run dir> --out-root data/regraded
    uv run python tools/report_difficulty.py data/regraded/iso8601-recurrence.jsonl
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import iso8601_recurrence as env


def regrade_run(run_dir: Path, out_dir: Path) -> tuple[int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "metadata.json", out_dir / "metadata.json")
    changed = correct = total = 0
    with (run_dir / "results.jsonl").open(encoding="utf-8") as src, (
        out_dir / "results.jsonl"
    ).open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stop_condition") != "timeout_reached":
                completion = row.get("completion") or []
                text = completion[-1].get("content") or "" if completion else ""
                new_reward = env.grade(text, row["answer"])
                total += 1
                correct += new_reward == 1.0
                changed += new_reward != row["reward"]
                row["reward"] = new_reward
                row["exact_match"] = new_reward
            dst.write(json.dumps(row) + "\n")
    return total, correct, changed


def main() -> None:
    run_root = Path(sys.argv[1])
    for metadata in sorted(run_root.glob("outputs/evals/*/*/metadata.json")):
        run_dir = metadata.parent
        out_dir = run_root / "regraded" / run_dir.parent.name / run_dir.name
        total, correct, changed = regrade_run(run_dir, out_dir)
        print(f"{run_dir.parent.name}: {correct}/{total} correct after regrade ({changed} rewards changed) -> {out_dir}")


if __name__ == "__main__":
    main()
