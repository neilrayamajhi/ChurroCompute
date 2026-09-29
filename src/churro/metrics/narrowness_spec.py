from __future__ import annotations

from churro.metrics.narrowness import NarrownessReport, narrowness, phrasings

ANSWER = "neutral"


def _exact(text: str, answer: str) -> float:
    return 1.0 if text.strip().strip(".").lower() == answer else 0.0


def _lenient(text: str, answer: str) -> float:
    return 1.0 if answer in text.lower() else 0.0


class TestPhrasings:
    def test_are_natural_rewordings_that_keep_the_answer(self) -> None:
        variants = phrasings(ANSWER)

        assert variants == [
            "Answer: neutral",
            "The answer is neutral.",
            "**neutral**",
            "neutral.",
            "Final answer: neutral",
        ]


class TestNarrowness:
    def test_lenient_grader_accepts_every_rephrasing(self) -> None:
        report = narrowness(_lenient, [ANSWER, "positive"])

        assert report == NarrownessReport(
            n_answers_tested=2,
            n_skipped=0,
            rejection_rate=0.0,
            rejected_phrasings={},
        )

    def test_exact_match_grader_rejects_rephrasings(self) -> None:
        report = narrowness(_exact, [ANSWER])

        assert report == NarrownessReport(
            n_answers_tested=1,
            n_skipped=0,
            rejection_rate=0.8,
            rejected_phrasings={
                "Answer: {answer}": 1,
                "The answer is {answer}.": 1,
                "**{answer}**": 1,
                "Final answer: {answer}": 1,
            },
        )

    def test_answers_whose_plain_form_scores_zero_are_skipped(self) -> None:
        report = narrowness(lambda _text, _answer: 0.0, [ANSWER])

        assert (report.n_answers_tested, report.n_skipped, report.rejection_rate) == (0, 1, 0.0)

    def test_rephrasing_that_scores_less_than_plain_counts_as_rejected(self) -> None:
        def partial(text: str, answer: str) -> float:
            return 1.0 if text == answer else 0.5

        report = narrowness(partial, [ANSWER])

        assert report.rejection_rate == 1.0
