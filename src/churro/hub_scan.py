from __future__ import annotations

from typing import Literal

ScanOutcome = Literal[
    "ok",
    "rejects_correct_answer",
    "cannot_tell_right_from_wrong",
    "inconclusive",
    "install_failed",
    "load_failed",
    "timeout",
]


_VERDICTS: tuple[ScanOutcome, ...] = (
    "ok",
    "rejects_correct_answer",
    "cannot_tell_right_from_wrong",
    "inconclusive",
)
_INSTALL_ERRORS = ("No solution found", "not found in the package registry", "Failed to build")


# Things our local Ollama + verifiers 0.3.0 stack can't run cheaply or at all.
_UNSUPPORTED_TAGS = frozenset(
    {
        "multi-turn", "tool-use", "tool-agent-user", "agent", "agentic", "mcp",
        "multi-agent", "sandbox", "browser", "cua", "gpu", "cuda", "vision",
        "multimodal", "rag", "llm-judge", "v1",
    }
)


def is_scan_candidate(tags: list[str]) -> bool:
    lowered = {t.lower() for t in tags}
    return "single-turn" in lowered and not lowered & _UNSUPPORTED_TAGS


def classify_check(returncode: int, stdout: str, stderr: str) -> ScanOutcome:
    for line in stdout.splitlines():
        if line.startswith("grader check for "):
            verdict = line.split(": ", 1)[1].split(" ", 1)[0]
            if verdict in _VERDICTS:
                return verdict  # type: ignore[return-value]
    if any(marker in stderr for marker in _INSTALL_ERRORS):
        return "install_failed"
    return "load_failed"
