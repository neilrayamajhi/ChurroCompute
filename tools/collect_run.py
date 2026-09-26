from __future__ import annotations

import argparse
from pathlib import Path

from churro.collect import (
    collect_run,
    count_timed_out,
    load_rollouts,
    merge_rollouts,
    write_rollouts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize one vf-eval run into data/raw/<env_id>.jsonl (append + dedupe)"
    )
    parser.add_argument("run_dir", type=Path, help="vf-eval output directory")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/raw"),
        help="root directory for normalized JSONL files",
    )
    args = parser.parse_args()

    new_rollouts = collect_run(args.run_dir)
    if not new_rollouts:
        print(
            f"no usable rollouts in {args.run_dir}: "
            f"{count_timed_out(args.run_dir)} were cut off by the vf-eval timeout"
        )
        return
    env_id = new_rollouts[0].env_id
    out_path = args.out_root / f"{env_id}.jsonl"

    existing = load_rollouts(out_path) if out_path.is_file() else []
    n = write_rollouts(merge_rollouts(existing, new_rollouts), out_path)
    added = n - len(existing)
    print(f"wrote {n} rollouts to {out_path} (+{added} new, {len(new_rollouts) - added} deduped)")
    timed_out = count_timed_out(args.run_dir)
    if timed_out:
        print(f"skipped {timed_out} rollout(s) cut off by the vf-eval timeout")


if __name__ == "__main__":
    main()
