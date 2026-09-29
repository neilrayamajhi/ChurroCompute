from __future__ import annotations

import argparse
import sys

import verifiers as vf

from churro.grader_check import check_environment

EXPLANATIONS = {
    "ok": "the grader scores the reference answer above a wrong and a blank one",
    "rejects_correct_answer": "the grader gives the reference answer 0",
    "cannot_tell_right_from_wrong": "a wrong or blank answer scores as well as the reference",
    "inconclusive": "the reference scores 0 in every form, so it may not be a literal answer; "
    "check this env by hand",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check an installed environment's grader before spending GPU time on it"
    )
    parser.add_argument("env_name", help="installed environment name, e.g. iso8601-recurrence")
    parser.add_argument("--rows", type=int, default=5, help="how many tasks to check")
    args = parser.parse_args()

    report = check_environment(vf.load_environment(args.env_name), max_rows=args.rows)
    for i, row in enumerate(report.rows):
        print(
            f"  task {i}: reference={row.reference:.2f}  wrong={row.wrong:.2f}  "
            f"blank={row.blank:.2f}  -> {row.verdict}"
        )
    print(f"eval tasks: {report.n_tasks}")
    print(f"grader check for {args.env_name}: {report.verdict} ({EXPLANATIONS[report.verdict]})")
    sys.exit(0 if report.verdict == "ok" else 1)


if __name__ == "__main__":
    main()
