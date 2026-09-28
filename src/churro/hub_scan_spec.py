from __future__ import annotations

from churro.hub_scan import classify_check, is_scan_candidate

ENV = "some-env"


class TestClassifyCheck:
    def test_passing_grader_check_is_ok(self) -> None:
        stdout = f"  task 0: ...\ngrader check for {ENV}: ok (the grader scores ...)\n"

        assert classify_check(0, stdout, "") == "ok"

    def test_failing_grader_check_reports_its_verdict(self) -> None:
        stdout = f"grader check for {ENV}: rejects_correct_answer (the grader ...)\n"

        assert classify_check(1, stdout, "") == "rejects_correct_answer"

    def test_inconclusive_grader_check_is_reported_as_such(self) -> None:
        stdout = f"grader check for {ENV}: inconclusive (the reference ...)\n"

        assert classify_check(1, stdout, "") == "inconclusive"

    def test_unresolvable_dependencies_are_install_failures(self) -> None:
        stderr = "  x No solution found when resolving `--with` dependencies:\n"

        assert classify_check(1, "", stderr) == "install_failed"

    def test_package_missing_from_index_is_an_install_failure(self) -> None:
        stderr = f"error: Because {ENV} was not found in the package registry"

        assert classify_check(2, "", stderr) == "install_failed"

    def test_crash_while_loading_the_env_is_a_load_failure(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            f"RuntimeError: Failed to load environment '{ENV}': boom\n"
        )

        assert classify_check(1, "", stderr) == "load_failed"

    def test_unrecognised_crash_is_a_load_failure(self) -> None:
        assert classify_check(1, "", "Traceback (most recent call last):\n") == (
            "load_failed"
        )


class TestIsScanCandidate:
    def test_single_turn_env_is_a_candidate(self) -> None:
        assert is_scan_candidate(["math", "single-turn", "train"]) is True

    def test_env_without_single_turn_tag_is_not(self) -> None:
        assert is_scan_candidate(["math", "train"]) is False

    def test_single_turn_env_needing_a_sandbox_is_not(self) -> None:
        assert is_scan_candidate(["single-turn", "sandbox"]) is False

    def test_single_turn_env_graded_by_an_llm_judge_is_not(self) -> None:
        assert is_scan_candidate(["single-turn", "llm-judge"]) is False

    def test_env_built_for_verifiers_v1_is_not(self) -> None:
        assert is_scan_candidate(["single-turn", "v1"]) is False

    def test_tag_matching_ignores_case(self) -> None:
        assert is_scan_candidate(["Single-Turn", "Math"]) is True
