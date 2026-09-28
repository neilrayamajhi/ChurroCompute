from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churro.collect import load_rollouts
from churro.config import LADDER
from churro.metrics.shape import ShapeReport, reward_shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward Shape for a normalized data/raw/<env>.jsonl file, overall and per model"
    )
    parser.add_argument("data_path", type=Path, help="path to a normalized JSONL file")
    args = parser.parse_args()

    rollouts = load_rollouts(args.data_path)
    overall = reward_shape(rollouts)
    if overall is None:
        print(f"no rollouts in {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"env: {rollouts[0].env_id}")
    print(f"{'model':<14}{'n':>6}{'distinct':>10}{'eff_bins':>10}{'binary':>8}{'at 0/1':>8}  top values")
    print("-" * 80)
    for model in LADDER:
        report = reward_shape(rollouts, model=model)
        if report is not None:
            print(_row(str(model), report))
    print("-" * 80)
    print(_row("all", overall))


def _row(label: str, r: ShapeReport) -> str:
    top = sorted(r.histogram.items(), key=lambda kv: -kv[1])[:4]
    top_text = ", ".join(f"{value:g}×{count}" for value, count in top)
    return (
        f"{label:<14}{r.n_rollouts:>6}{r.distinct_values:>10}{r.effective_bins:>10.2f}"
        f"{'yes' if r.effectively_binary else 'no':>8}{r.frac_at_extremes:>8.0%}  {top_text}"
    )


if __name__ == "__main__":
    main()
