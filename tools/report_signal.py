from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from churro.collect import load_rollouts
from churro.metrics.signal import signal_rate
from churro.schema import ModelId


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Signal Rate for a normalized data/raw/<env>.jsonl file"
    )
    parser.add_argument("data_path", type=Path, help="path to a normalized JSONL file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="restrict to rollouts from this model id",
    )
    args = parser.parse_args()

    rollouts = load_rollouts(args.data_path)
    model = ModelId(args.model) if args.model else None
    report = signal_rate(rollouts, model=model)

    if report is None:
        print(f"no rollouts in {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(report.to_json_dict(), indent=2))
    print(
        f"\n  {report.signal_rate:.0%} of tasks produce learning signal "
        f"(model={report.model or 'all'}, group_size={report.group_size}, "
        f"n={report.n_tasks}, ±{report.signal_rate_ci95:.0%})"
    )
    print(f"  {report.dead_too_easy} dead: too easy")
    print(f"  {report.dead_too_hard} dead: too hard or broken grader")
    print(f"  mean spread on live groups: {report.mean_spread:.3f}")


if __name__ == "__main__":
    main()
