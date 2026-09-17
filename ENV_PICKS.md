# Environment Picks — Which RL envs we ran and why

_Written 2026-08-30. Read this if you're wondering "wait what environments are these and why did we pick them."_

---

## The short version

We need to run Churro's Signal Rate metric on **more than one environment** to prove it actually discriminates between "good training data" and "bad training data." A metric that reports the same number for every env would be useless — like a thermometer that reads 70°F everywhere.

So the plan: pick 2-3 more environments alongside gsm8k, run each through the same `vf-eval -n 30 -r 8` sample, and see if the signal_rate numbers come out **different in ways that match intuition**.

---

## What went wrong first

I initially picked `primeintellect/apex-shortlist` (competition math) and `primeintellect/aime2026` (AIME 2026 problems). **Both failed to install-and-run.** They were built against a newer version of the `verifiers` library than the one we have installed (0.3.0). Specifically they import `vf.Tasks` (plural, doesn't exist in 0.3.0) and `vf.MaybeThinkParser` (doesn't exist).

This is itself a Churro-relevant finding: on a small n=3 sample, **~67% of the environments I picked from the Prime Intellect Hub were broken.** ROADMAP Phase 7 predicts this ("broken packages, missing deps... failure rate is itself a finding") — we're just hitting it earlier than expected.

---

## What we're actually running

I switched strategy: instead of picking envs by task-flavor first and hoping they install, I now check the source code for the pattern that works with our verifiers 0.3.0:

- Uses `import verifiers as vf` (old-style)
- Exports a top-level `def load_environment(**kwargs) -> vf.Environment`

Three envs passed both checks and smoke-tested successfully (produced real rewards, not error stacks):

### 1. `liuliu/regex-craft` — write a regex to match given strings

- **Task shape:** model is given a set of strings and asked to produce a regex that matches them all. Grader runs the regex, checks matches.
- **Why we picked it:** completely different flavor from gsm8k (math word problems). Regex is a **programming skill** — tests a different part of the model.
- **Speed:** ~8 seconds per rollout (short outputs). n=30 × r=8 = ~32 minutes total. **Fast.**
- **Hypothesis:** medium signal rate. qwen3:1.7b is small enough that regex will be mixed — some easy patterns it nails, harder ones it doesn't.

### 2. `joeljose/iso8601-recurrence` — parse/format datetime recurrence strings

- **Task shape:** deal with ISO 8601 recurrence rules (things like `R5/PT10M` = repeat 5 times every 10 minutes). Model outputs a structured answer.
- **Why we picked it:** **deterministic, exact-match grading.** No fuzzy "close enough" — either the model produces the exact right string or scores 0. Different from gsm8k's numeric answers.
- **Speed:** ~35 seconds per rollout. n=30 × r=8 = ~2 hours.
- **Hypothesis:** high dead_too_hard. Small models are notoriously bad at exact-format tasks — they hallucinate delimiters, miscount digits, etc. Likely most rollouts fail.

### 3. `halfounce/exact-determinant` — compute the determinant of a matrix

- **Task shape:** model is given a small integer matrix, must output the exact determinant.
- **Why we picked it:** **numeric-only output**, no reasoning artifact to score. Pure math capability check.
- **Speed:** ~131 seconds per rollout (surprisingly slow for a numeric answer — model probably shows work). n=30 × r=8 = ~4-5 hours.
- **Hypothesis:** high dead_too_hard for 3x3+ matrices. Determinant calculation is where small models fall over.

---

## Why these three together tell a story

The four envs we'll have signal rates for span very different task shapes:

| Env | Flavor | Grader | Hypothesis |
|---|---|---|---|
| `gsm8k` | grade-school math word problems | numeric | **Saturated** (mostly dead_too_easy) — 1.7B is too strong here |
| `regex-craft` | programming (pattern matching) | functional | **Live** — mixed difficulty |
| `iso8601-recurrence` | exact-format parsing | exact-string | **Dead_too_hard** — small models fail exact-match |
| `exact-determinant` | matrix arithmetic | exact-numeric | **Dead_too_hard** — 1.7B can't do multi-step linear algebra reliably |

**If the metric works, we should see a visible spread** — gsm8k with low signal (too easy), regex-craft with real signal (mixed), and the two exact-match envs with low signal (too hard). Different **failure modes** should show up in `dead_too_easy` vs `dead_too_hard`.

**If the metric doesn't work,** all four will report similar numbers regardless of what we intuit, which would tell us the metric isn't measuring what we think.

---

## What we skipped and why

- **Anything multi-turn, sandbox, agent, tool-use, or Docker-requiring** — needs infrastructure we don't have on this Mac. Would spend hours setting up and still might not work.
- **LLM-judge envs** — the grader itself is a model call, doubles cost per rollout.
- **Coding envs with test suites** — often need Docker containers to safely run untrusted code.
- **`primeintellect/aime2026` and `apex-shortlist`** — broken (see "What went wrong first").
- **`hugoabonizio/if-rlvr`** — worked but generated 4,000-token outputs. A full 30×8 run would take 8+ hours. Skipped for time budget.
- **`nike47/html-extraction`** — imported `verifiers.v1` and didn't expose top-level `load_environment`. Doesn't fit our stack.

---

## What happens next

1. Kick off `regex-craft` first (~30 min).
2. Then `iso8601-recurrence` (~2 hr).
3. Then `exact-determinant` (~4-5 hr).
4. Each finishes → normalize with `collect_run.py` → compute signal rate with `report_signal.py` → append to `FINDINGS.md`.
5. Cross-env comparison table in `FINDINGS.md` once all four numbers are in.

Total wall-clock estimate: **~7-8 hours** of Mac-awake time (use `caffeinate -i` so it doesn't sleep).
