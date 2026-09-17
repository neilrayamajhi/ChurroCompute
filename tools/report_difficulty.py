from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from churro.collect import load_rollouts
from churro.config import LADDER, LADDER_VERSION
from churro.metrics.difficulty import difficulty_curve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Difficulty Curve across the model ladder for a data/raw/<env>.jsonl file"
    )
    parser.add_argument("data_path", type=Path, help="path to a normalized JSONL file")
    parser.add_argument(
        "--ladder-version",
        type=str,
        default=LADDER_VERSION,
        help=f"stamp on the report (default: {LADDER_VERSION})",
    )
    args = parser.parse_args()

    rollouts = load_rollouts(args.data_path)
    report = difficulty_curve(rollouts, LADDER, args.ladder_version)

    if report is None:
        print(f"no rollouts in {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(report.to_json_dict(), indent=2))
    print()
    print(f"env: {report.env_id}   ladder: {report.ladder_version}")
    print("-" * 64)
    for rung in report.rungs:
        if rung.signal is None:
            print(f"  {rung.model:<14} (no data)")
            continue
        sig = rung.signal
        print(
            f"  {rung.model:<14} "
            f"pass={rung.pass_rate:.2f}  "
            f"signal={sig.signal_rate:.2f} ±{sig.signal_rate_ci95:.2f}  "
            f"(easy={sig.dead_too_easy}, hard={sig.dead_too_hard}, n={sig.n_tasks})"
        )
    print("-" * 64)
    print(f"  best_signal_model: {report.best_signal_model or '(none)'}")
    print(f"  floor:             {report.floor or '(none)'}")
    print(f"  ceiling:           {report.ceiling or '(none)'}")
    print(f"  slope:             {report.slope:.4f} pass_rate/rung")
    print(f"  saturated:         {report.saturated}")
    print(f"  possibly_broken:   {report.possibly_broken}")


if __name__ == "__main__":
    main()
