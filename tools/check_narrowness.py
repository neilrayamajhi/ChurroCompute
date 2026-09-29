from __future__ import annotations

import argparse

import verifiers as vf

from churro.grader_check import reference_forms
from churro.metrics.narrowness import narrowness
from churro.rescore import score_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grader narrowness (PRD Part 12): does the grader reject correct answers "
        "that are merely worded differently?"
    )
    parser.add_argument("env_name", help="installed environment name")
    parser.add_argument("--rows", type=int, default=10, help="tasks to test")
    args = parser.parse_args()

    env = vf.load_environment(args.env_name)
    dataset = env.eval_dataset or env.dataset
    rows = [dataset[i] for i in range(min(args.rows, len(dataset)))]
    # Rephrase the reference in the format the env asks for (e.g. \boxed{}),
    # so a grader isn't blamed for rejecting answers that ignore its format.
    # On a tie, prefer a formatted form (e.g. \boxed{18} over 18): it's the
    # format the env asks for, and the one models are told to wrap.
    def best_form(row: dict[str, object]) -> str:
        answer = str(row.get("answer", ""))
        forms = reference_forms(env, answer)
        return max(forms, key=lambda text: (score_text(env, row, text), text != answer))

    report = narrowness(lambda text, row: score_text(env, row, text), rows, answer_of=best_form)
    print(
        f"{args.env_name}: rephrased the reference answer of {report.n_answers_tested} tasks "
        f"({report.n_skipped} skipped: reference scored 0) -> "
        f"{report.rejection_rate:.0%} of correct rephrasings scored below the reference itself"
    )
    for template, count in sorted(report.rejected_phrasings.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {count}x: {template!r}")


if __name__ == "__main__":
    main()
