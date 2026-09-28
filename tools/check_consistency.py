from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import verifiers as vf

from churro.metrics.consistency import consistency
from churro.rescore import score_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score saved rollouts several times to test grader consistency (PRD Part 9)"
    )
    parser.add_argument("env_name", help="installed environment name")
    parser.add_argument("results", type=Path, nargs="+", help="vf-eval results.jsonl file(s)")
    parser.add_argument("--items", type=int, default=20, help="saved rollouts to re-score")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    env = vf.load_environment(args.env_name)
    dataset = env.eval_dataset or env.dataset
    task_rows = {row["example_id"]: row for row in dataset}

    items = []
    for path in args.results:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            saved = json.loads(line)
            row = task_rows.get(saved["example_id"])
            completion = saved.get("completion") or []
            if row is None or not completion:
                continue
            items.append((row, completion[-1].get("content") or ""))
            if len(items) >= args.items:
                break

    if not items:
        sys.exit("no saved rollouts matched this environment's tasks")
    report = consistency(lambda item: score_text(env, *item), items, n_repeats=args.repeats)
    print(
        f"{args.env_name}: re-scored {report.n_items_tested} saved answers x{report.n_repeats}: "
        f"disagreement_rate={report.disagreement_rate:.0%} max_spread={report.max_spread} "
        f"-> {'deterministic' if report.deterministic else 'INCONSISTENT'}"
    )


if __name__ == "__main__":
    main()
