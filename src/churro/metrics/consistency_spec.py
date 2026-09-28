from __future__ import annotations

from collections.abc import Callable
from itertools import cycle

from hypothesis import given
from hypothesis import strategies as st

from churro.metrics.consistency import ConsistencyReport, consistency

REPEATS = 5


def _flaky(values: list[float]) -> Callable[[object], float]:
    stream = cycle(values)
    return lambda _item: next(stream)


class TestConsistency:
    def test_no_items_reports_nothing_tested(self) -> None:
        assert consistency(lambda _item: 1.0, [], n_repeats=REPEATS) == ConsistencyReport(
            n_items_tested=0,
            n_repeats=REPEATS,
            disagreement_rate=0.0,
            max_spread=0.0,
            deterministic=True,
        )

    def test_grader_returning_the_same_score_every_time_is_deterministic(self) -> None:
        items = ["a", "b", "c"]

        report = consistency(lambda item: float(len(item)), items, n_repeats=REPEATS)

        assert report == ConsistencyReport(
            n_items_tested=3,
            n_repeats=REPEATS,
            disagreement_rate=0.0,
            max_spread=0.0,
            deterministic=True,
        )

    def test_grader_that_wobbles_on_every_item_disagrees_everywhere(self) -> None:
        report = consistency(_flaky([0.0, 1.0]), ["a", "b"], n_repeats=REPEATS)

        assert (report.disagreement_rate, report.max_spread, report.deterministic) == (
            1.0,
            1.0,
            False,
        )

    def test_scores_each_item_n_repeats_times(self) -> None:
        calls: list[object] = []

        consistency(lambda item: calls.append(item) or 0.0, ["x", "y"], n_repeats=3)

        assert calls == ["x", "x", "x", "y", "y", "y"]

    @given(st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=20))
    def test_constant_per_item_scores_are_always_deterministic(
        self, per_item: list[float]
    ) -> None:
        items = list(range(len(per_item)))

        report = consistency(lambda i: per_item[i], items, n_repeats=REPEATS)

        assert report.deterministic
