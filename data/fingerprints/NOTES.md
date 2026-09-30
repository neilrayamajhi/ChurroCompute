
## Known answer-key problems

These affect how to read the rows above; full evidence is in `FINDINGS.md`.

- **iso8601-recurrence:** as shipped, the grader scores every answer 0 under verifiers 0.3.0 (it only reads dict messages). The row above comes from re-scoring the cached rollouts with the env's own `grade()` (`tools/regrade_iso8601.py`).
- **fingpt-sentiment:** passes the answer-key check but rejects 100% of correctly reworded answers. `Answer: neutral` scores 0, below the wrong bare `positive` (0.1). Its falling pass rate with model size reflects formatting, not sentiment skill.
- **regex-craft:** not listed. Its grader can't tell right from wrong (any valid regex scores 0.5), so no report card is published.
- **legalbench:** not listed. Its default config evaluates on a single task.
