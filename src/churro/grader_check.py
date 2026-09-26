from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import verifiers as vf
from verifiers.types import AssistantMessage, State

Verdict = Literal["ok", "rejects_correct_answer", "cannot_tell_right_from_wrong"]

WRONG_ANSWER = "I do not know."


@dataclass(frozen=True, slots=True)
class RowCheck:
    reference: float
    wrong: float
    blank: float
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class GraderReport:
    rows: list[RowCheck]
    verdict: Verdict


def grader_verdict(reference: float, wrong: float, blank: float) -> Verdict:
    if reference <= 0.0:
        return "rejects_correct_answer"
    if reference <= max(wrong, blank):
        return "cannot_tell_right_from_wrong"
    return "ok"


def check_environment(env: vf.Environment, max_rows: int = 5) -> GraderReport:
    dataset = env.eval_dataset or env.dataset
    rows = [dataset[i] for i in range(min(max_rows, len(dataset)))]
    checks = []
    for row in rows:
        answer = str(row.get("answer", ""))
        reference = max(_score(env, row, text) for text in _reference_forms(answer))
        wrong = _score(env, row, WRONG_ANSWER)
        blank = _score(env, row, "")
        verdict = grader_verdict(reference, wrong, blank)
        checks.append(RowCheck(reference, wrong, blank, verdict))
    return GraderReport(rows=checks, verdict=_majority(checks))


def _reference_forms(answer: str) -> list[str]:
    return [answer, f"\\boxed{{{answer}}}", f"<answer>{answer}</answer>"]


def _score(env: vf.Environment, row: dict[str, object], text: str) -> float:
    # Build the state the way a real rollout does: the completion is a list of
    # AssistantMessage objects, not dicts, and row columns arrive as-is.
    state = State(
        input=dict(row),
        prompt=row["prompt"],
        completion=[AssistantMessage(content=text)],
        answer=row.get("answer", ""),
        info=row.get("info") or {},
        trajectory=[],
    )
    asyncio.run(env.rubric.score_group([state]))
    return float(state["reward"])


def _majority(checks: list[RowCheck]) -> Verdict:
    counts = Counter(c.verdict for c in checks)
    if not counts:
        return "rejects_correct_answer"
    top = max(counts.values())
    tied = [v for v, n in counts.items() if n == top]
    return next((v for v in tied if v != "ok"), tied[0])
