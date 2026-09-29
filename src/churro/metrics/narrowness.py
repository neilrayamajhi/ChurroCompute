from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


# Ways a model naturally wraps a correct answer. A grader that scores these
# below the bare answer punishes phrasing rather than correctness.
_TEMPLATES = (
    "Answer: {answer}",
    "The answer is {answer}.",
    "**{answer}**",
    "{answer}.",
    "Final answer: {answer}",
)


@dataclass(frozen=True, slots=True)
class NarrownessReport:
    n_answers_tested: int
    n_skipped: int
    rejection_rate: float
    rejected_phrasings: dict[str, int]


def phrasings(answer: str) -> list[str]:
    return [t.format(answer=answer) for t in _TEMPLATES]


def narrowness(
    score_fn: Callable[[str, T], float],
    items: Sequence[T],
    answer_of: Callable[[T], str] = str,
) -> NarrownessReport:
    rejected: dict[str, int] = {}
    tested = skipped = rejections = 0
    for item in items:
        answer = answer_of(item)
        plain = score_fn(answer, item)
        if plain <= 0.0:
            skipped += 1
            continue
        tested += 1
        for template, text in zip(_TEMPLATES, phrasings(answer)):
            if score_fn(text, item) < plain:
                rejections += 1
                rejected[template] = rejected.get(template, 0) + 1
    checks = tested * len(_TEMPLATES)
    return NarrownessReport(
        n_answers_tested=tested,
        n_skipped=skipped,
        rejection_rate=round(rejections / checks, 4) if checks else 0.0,
        rejected_phrasings=rejected,
    )
