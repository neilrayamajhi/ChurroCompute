from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import verifiers as vf

from churro.rescore import score_text

Verdict = Literal[
    "ok", "rejects_correct_answer", "cannot_tell_right_from_wrong", "inconclusive"
]

WRONG_ANSWER = "I do not know."


@dataclass(frozen=True, slots=True)
class RowCheck:
    reference: float
    reference_as_dict: float
    wrong: float
    blank: float
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class GraderReport:
    rows: list[RowCheck]
    verdict: Verdict
    n_tasks: int


def grader_verdict(
    reference: float, reference_as_dict: float, wrong: float, blank: float
) -> Verdict:
    if reference <= 0.0:
        # Scoring only when handed a plain dict is the iso8601 bug; scoring 0
        # in every shape means the answer column isn't a usable answer.
        return "rejects_correct_answer" if reference_as_dict > 0.0 else "inconclusive"
    if reference <= max(wrong, blank):
        return "cannot_tell_right_from_wrong"
    return "ok"


def check_environment(env: vf.Environment, max_rows: int = 5) -> GraderReport:
    dataset = env.eval_dataset or env.dataset
    rows = [dataset[i] for i in range(min(max_rows, len(dataset)))]
    checks = []
    for row in rows:
        forms = reference_forms(env, str(row.get("answer", "")))
        reference = max(score_text(env, row, text) for text in forms)
        reference_as_dict = max(score_text(env, row, text, as_dict=True) for text in forms)
        wrong = score_text(env, row, WRONG_ANSWER)
        blank = score_text(env, row, "")
        verdict = grader_verdict(reference, reference_as_dict, wrong, blank)
        checks.append(RowCheck(reference, reference_as_dict, wrong, blank, verdict))
    return GraderReport(rows=checks, verdict=_majority(checks), n_tasks=len(dataset))


def _answer_tags(env: vf.Environment) -> set[str]:
    parsers = [env.parser, getattr(env.rubric, "parser", None)]
    return {"answer"} | {
        tag for p in parsers if (tag := getattr(p, "answer_field", None))
    }


def reference_forms(env: vf.Environment, answer: str) -> list[str]:
    """The reference answer in each format the env's parser might expect."""
    tagged = [f"<{tag}>{answer}</{tag}>" for tag in sorted(_answer_tags(env))]
    return [answer, f"\\boxed{{{answer}}}", *tagged]


def _majority(checks: list[RowCheck]) -> Verdict:
    counts = Counter(c.verdict for c in checks)
    if not counts:
        return "inconclusive"
    top = max(counts.values())
    tied = [v for v, n in counts.items() if n == top]
    return next((v for v in tied if v != "ok"), tied[0])
