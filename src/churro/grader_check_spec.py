from __future__ import annotations

import verifiers as vf
from datasets import Dataset

from churro.grader_check import (
    GraderReport,
    RowCheck,
    check_environment,
    grader_verdict,
)

REFERENCE = "[a-z]+\\d{3}"


def _text(completion: object) -> str:
    last = completion[-1]  # type: ignore[index]
    return last.get("content") or ""


def _working_grader(completion: object, answer: str, **kwargs: object) -> float:
    return 1.0 if _text(completion).strip() == answer else 0.0


def _boxed_only_grader(completion: object, answer: str, **kwargs: object) -> float:
    return 1.0 if _text(completion) == f"\\boxed{{{answer}}}" else 0.0


def _dict_only_grader(completion: object, answer: str, **kwargs: object) -> float:
    last = completion[-1]  # type: ignore[index]
    text = last.get("content") if isinstance(last, dict) else ""
    return 1.0 if text == answer else 0.0


def _info_grader(completion: object, answer: str, **kwargs: object) -> float:
    info = kwargs.get("info") or {}
    assert isinstance(info, dict)
    expected = info.get("expected")
    if expected is None:
        return 0.5 if _text(completion) else 0.0
    return 1.0 if _text(completion) == expected else 0.0


def _env(reward_func: object, rows: int = 3) -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": f"task {i}"}], "answer": REFERENCE}
            for i in range(rows)
        ]
    )
    return vf.SingleTurnEnv(
        dataset=dataset, rubric=vf.Rubric(funcs=[reward_func], weights=[1.0])
    )


class TestGraderVerdict:
    def test_reference_scoring_above_wrong_and_blank_is_ok(self) -> None:
        assert grader_verdict(reference=1.0, wrong=0.0, blank=0.0) == "ok"

    def test_reference_scoring_zero_like_everything_else_rejects_correct(
        self,
    ) -> None:
        assert grader_verdict(reference=0.0, wrong=0.0, blank=0.0) == (
            "rejects_correct_answer"
        )

    def test_reference_tied_with_wrong_cannot_tell_right_from_wrong(self) -> None:
        assert grader_verdict(reference=0.5, wrong=0.5, blank=0.0) == (
            "cannot_tell_right_from_wrong"
        )

    def test_wrong_answer_beating_reference_cannot_tell_right_from_wrong(
        self,
    ) -> None:
        assert grader_verdict(reference=0.2, wrong=0.9, blank=0.0) == (
            "cannot_tell_right_from_wrong"
        )


class TestCheckEnvironment:
    def test_working_grader_passes(self) -> None:
        report = check_environment(_env(_working_grader), max_rows=2)

        assert report == GraderReport(
            rows=[
                RowCheck(reference=1.0, wrong=0.0, blank=0.0, verdict="ok"),
                RowCheck(reference=1.0, wrong=0.0, blank=0.0, verdict="ok"),
            ],
            verdict="ok",
        )

    def test_grader_expecting_boxed_answers_passes(self) -> None:
        report = check_environment(_env(_boxed_only_grader), max_rows=2)

        assert report.verdict == "ok"

    def test_dict_only_grader_is_caught_rejecting_correct_answers(self) -> None:
        report = check_environment(_env(_dict_only_grader), max_rows=2)

        assert report.verdict == "rejects_correct_answer"

    def test_grader_reading_missing_info_is_caught_scoring_wrong_like_right(
        self,
    ) -> None:
        report = check_environment(_env(_info_grader), max_rows=2)

        assert report.verdict == "cannot_tell_right_from_wrong"

    def test_checks_at_most_max_rows(self) -> None:
        report = check_environment(_env(_working_grader, rows=5), max_rows=3)

        assert len(report.rows) == 3
